#![allow(unused)]

extern crate proc_macro;

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

use syn::{
    Attribute, ConstParam, Expr, ExprCall, ExprPath, FnArg, GenericArgument, GenericParam, Ident, ImplItem, ImplItemFn,
    Item, ItemFn, ItemImpl, ItemMod, ItemTrait, Lifetime, LifetimeParam, Pat, Path, PathArguments, PathSegment, QSelf,
    ReturnType, Signature, Token, TraitItem, Type, TypeParamBound, WhereClause, WherePredicate,
    parse::{Parse, ParseStream, Parser as _},
    punctuated::Punctuated,
    visit_mut::VisitMut,
};

mod late_bound;

const SKIP_DISPATCH: &str = "skip_dispatch";

/// Describes a single dispatch backend used by both `#[dispatch]` and `dispatch_dyn!`.
struct Backend {
    /// Name of the corresponding `InstructionSet` variant (e.g. `"X86V3"`).
    isa: &'static str,

    /// Comma-separated CPU feature string passed to `#[target_feature(enable = "...")]`.
    /// Empty string means no additional target features are required (e.g. scalar).
    target_feature: &'static str,

    /// Thermite-crate-relative path to the concrete type that implements `Simd` for
    /// this backend (e.g. `"backend::x86_v3::X86V3"`).
    ///
    /// `None` when the backend does not yet have a complete, runtime-dispatchable `Simd`
    /// implementation (e.g. x86-v1 whose `registers/` submodule is still WIP).
    /// Such backends are skipped by `dispatch_dyn!` and their hardware falls through to
    /// the scalar fallback.
    simd_type: Option<&'static str>,
}

static BACKENDS: &[Backend] = cfg_select! {
    feature = "x86" => &[
        Backend { isa: "Scalar", target_feature: "",         simd_type: Some("backend::scalar::Scalar")   },
        //Backend { isa: "X86V1",  target_feature: "sse2",     simd_type: None                              },
        Backend { isa: "X86V2",  target_feature: "sse4.2",   simd_type: Some("backend::x86_v2::X86V2")   },
        Backend { isa: "X86V3",  target_feature: "avx2,fma", simd_type: Some("backend::x86_v3::X86V3")   },
    ],
    feature = "neon" => &[
        Backend { isa: "Scalar", target_feature: "",     simd_type: Some("backend::scalar::Scalar") },
        Backend { isa: "NEON",   target_feature: "neon", simd_type: None                            },
    ],
    feature = "wasm" => &[
        Backend { isa: "Scalar", target_feature: "",        simd_type: Some("backend::scalar::Scalar") },
        Backend { isa: "WASM32", target_feature: "simd128", simd_type: Some("backend::wasm::Wasm")     },
    ],
    feature = "spirv" => &[
        Backend { isa: "SPIRV", target_feature: "", simd_type: None },
    ],
    _ => &[
        Backend { isa: "Scalar", target_feature: "", simd_type: Some("backend::scalar::Scalar") },
    ]
};

/// Associated type names declared on the `Simd` trait that correspond to `Vector<R>`
/// user-facing types.  A bare, unqualified occurrence of any of these names as a type
/// inside a `dispatch_dyn!` body is automatically rewritten to
/// `#thermite::Vector<S::name>`.
const SIMD_VECTOR_TYPES: &[&str] = &[
    "f32x2", "f32x4", "f32x8", "f32x16", "i32x2", "i32x4", "i32x8", "i32x16", "u32x2", "u32x4", "u32x8", "u32x16",
    "f64x2", "f64x4", "f64x8", "f64x16", "i64x2", "i64x4", "i64x8", "i64x16", "u64x2", "u64x4", "u64x8", "u64x16",
    "usizex2", "usizex4", "usizex8", "usizex16", "f32xN", "i32xN", "u32xN", "f64xN", "i64xN", "u64xN", "usizexN",
    "usizex3A", "f32x3A", "i32x3A", "u32x3A", "f64x3A", "i64x3A", "u64x3A",
];

/// Holds the directly parsed attributes (no intermediate Punctuated tree).
struct DispatchAttributes {
    simd: TokenStream,
    thermite: TokenStream,
}

/// Compile-time ISA dispatch for functions, `impl` blocks, traits, and modules.
///
/// Rewrites the annotated item so that every method/function body is wrapped in an
/// `#[inline(always)]` inner copy and then called through a per-backend
/// `#[target_feature(enable = "…")]` trampoline, selected at compile time by matching
/// on `<S as HasIsa>::ISA` — a const that is resolved when `S` is monomorphized.
///
/// # Syntax
///
/// ```rust,ignore
/// // `S` is the default SIMD type parameter name; override with a positional ident:
/// #[dispatch]
/// fn my_fn<S: HasIsa>(…) { … }
///
/// // Explicit SIMD parameter name:
/// #[dispatch(V)]
/// fn my_fn<V: HasIsa>(…) { … }
///
/// // Override the thermite crate path (needed when calling from inside thermite itself):
/// #[dispatch(thermite = "crate")]
/// fn my_fn<S: HasIsa>(…) { … }
///
/// // Both together:
/// #[dispatch(V, thermite = "crate")]
/// fn my_fn<V: HasIsa>(…) { … }
/// ```
///
/// # Supported items
///
/// | Item | Effect |
/// |------|--------|
/// | `fn` | Wraps the body; the function gains per-backend trampolines. |
/// | `impl` block | Every method in the block is wrapped individually. |
/// | `trait` definition | Strips `#[skip_dispatch]` markers from trait methods (no-op otherwise). |
/// | `mod` | Recursively applies `#[dispatch]` to every `fn` and `impl` inside. |
///
/// # `#[skip_dispatch]`
///
/// Place `#[skip_dispatch]` on any individual `fn` or `impl` item (or on the
/// `impl` block itself) to opt it out of dispatch generation entirely.
///
/// # How it works
///
/// For a function `fn foo<S: HasIsa>(args…)`:
///
/// 1. The original body is moved into an `#[inline(always)]` copy named `foo`.
/// 2. For each backend a `#[target_feature(enable = "…")] unsafe fn __dispatch_<backend>`
///    is generated that calls `foo` under the appropriate CPU feature flags.
/// 3. The outer body becomes a `match <S as HasIsa>::ISA { … }` that selects the
///    right trampoline.  Because `ISA` is a const, LLVM folds the match away at
///    monomorphization time — there is no runtime branch.
///
/// `impl` blocks use a private helper trait to allow the trampolines to call back into
/// `Self` without recursion.
#[proc_macro_attribute]
pub fn dispatch(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut simd = quote! { S };
    let mut thermite = quote! { ::thermite };

    let attr_parser = syn::meta::parser(|meta| {
        if meta.path.is_ident("thermite") {
            let s: syn::LitStr = meta.value()?.parse()?;
            let path = quote::format_ident!("{}", s.value());
            thermite = quote! { #path };
            Ok(())
        } else if meta.path.get_ident().is_some() {
            let path = &meta.path;
            simd = quote! { #path };
            Ok(())
        } else {
            Err(meta.error("unsupported dispatch attribute"))
        }
    });

    if let Err(err) = attr_parser.parse(attr) {
        return err.into_compile_error().into();
    }

    let attr_data = DispatchAttributes { simd, thermite };

    let mut item = syn::parse_macro_input!(item as Item);

    match item {
        Item::Fn(ref mut fn_item) => gen_function(&attr_data, fn_item),
        Item::Impl(ref mut impl_block) => gen_impl_block(&attr_data, impl_block),
        Item::Trait(ref mut trait_item) => gen_trait_def(&attr_data, trait_item),
        Item::Mod(ref mut module) => gen_mod_def(&attr_data, module),
        _ => unimplemented!("#[dispatch] is only supported on naked functions, impl blocks or trait definitions!"),
    }

    item.into_token_stream().into()
}

// -----------------------------------------------------------------------------
// AST Traversal & Mutators
// -----------------------------------------------------------------------------

/// Helper to check for and remove specific internal attributes.
fn take_attribute(attrs: &mut Vec<Attribute>, name: &str) -> bool {
    let len = attrs.len();
    attrs.retain(|attr| !attr.path().is_ident(name));
    attrs.len() < len
}

struct TypeVisitor {
    self_ty: Box<Type>,
}

impl VisitMut for TypeVisitor {
    #[rustfmt::skip]
    fn visit_type_mut(&mut self, i: &mut Type) {
        if let Type::Path(p) = i && p.qself.is_none() && p.path.segments.len() > 1
            && let Some(first) = p.path.segments.first_mut()
            && first.ident == "Self"
        {
            let mut path = Punctuated::new();
            let old_path = std::mem::take(&mut p.path.segments);
            for segment in old_path.into_iter().skip(1) {
                path.push(segment);
            }
            p.path.segments = path;
            p.path.leading_colon = Some(Default::default());
            p.qself = Some(QSelf {
                lt_token: Default::default(),
                ty: self.self_ty.clone(),
                position: 0,
                as_token: None,
                gt_token: Default::default(),
            });
        }

        syn::visit_mut::visit_type_mut(self, i);
    }
}

struct SelfTraitVisitor {
    depth: u32,
    method: Ident,
    self_ty: Box<Type>,
    trait_: Path,
    qself: QSelf,
}

impl SelfTraitVisitor {
    fn new(trait_: Path, self_ty: Box<Type>, method: Ident) -> SelfTraitVisitor {
        let qself = QSelf {
            lt_token: Default::default(),
            ty: self_ty.clone(),
            position: trait_.segments.len(),
            as_token: Some(Default::default()),
            gt_token: Default::default(),
        };

        SelfTraitVisitor {
            depth: 0,
            method,
            self_ty,
            trait_,
            qself,
        }
    }
}

impl VisitMut for SelfTraitVisitor {
    fn visit_expr_mut(&mut self, i: &mut Expr) {
        if self.depth == 0 {
            match i {
                Expr::MethodCall(m) if m.method == self.method => {
                    if let Expr::Path(p) = &mut *m.receiver
                        && p.path.is_ident("self")
                    {
                        let mut path = self.trait_.clone();

                        // syn 2.0: turbofish is directly AngleBracketedGenericArguments
                        let arguments = match &m.turbofish {
                            None => PathArguments::None,
                            Some(tf) => PathArguments::AngleBracketed(tf.clone()),
                        };

                        path.segments.push(PathSegment {
                            ident: m.method.clone(),
                            arguments,
                        });

                        let call_attrs = std::mem::take(&mut m.attrs);
                        let path_attrs = std::mem::take(&mut p.attrs);

                        *i = Expr::Call(ExprCall {
                            attrs: call_attrs,
                            func: Box::new(Expr::Path(ExprPath {
                                attrs: path_attrs,
                                qself: Some(self.qself.clone()),
                                path,
                            })),
                            paren_token: m.paren_token,
                            args: {
                                let mut args = Punctuated::new();

                                // Reuse the exact receiver expression from the AST to preserve hygiene
                                args.push((*m.receiver).clone());

                                for arg in m.args.iter() {
                                    args.push(arg.clone());
                                }
                                args
                            },
                        });
                    }
                }
                Expr::Call(c) => {
                    if let Expr::Path(p) = &mut *c.func
                        && p.path.segments.len() == 2
                        && p.path.segments.first().unwrap().ident == "Self"
                        && p.path.segments.last().unwrap().ident == self.method
                    {
                        p.qself = Some(self.qself.clone());
                        // start off new path with trait_
                        let old_path = std::mem::replace(&mut p.path, self.trait_.clone());
                        // skip 1st `Self` segment, then append the rest
                        for segment in old_path.segments.into_iter().skip(1) {
                            p.path.segments.push(segment);
                        }
                    }
                }
                _ => {}
            }
        }

        syn::visit_mut::visit_expr_mut(self, i);
    }

    // Track scope changes to avoid rewriting non-associated scopes.
    fn visit_item_fn_mut(&mut self, i: &mut syn::ItemFn) {
        self.depth += 1;
        syn::visit_mut::visit_item_fn_mut(self, i);
        self.depth -= 1;
    }

    fn visit_impl_item_fn_mut(&mut self, i: &mut syn::ImplItemFn) {
        self.depth += 1;
        syn::visit_mut::visit_impl_item_fn_mut(self, i);
        self.depth -= 1;
    }

    fn visit_expr_closure_mut(&mut self, i: &mut syn::ExprClosure) {
        self.depth += 1;
        syn::visit_mut::visit_expr_closure_mut(self, i);
        self.depth -= 1;
    }

    fn visit_trait_item_fn_mut(&mut self, i: &mut syn::TraitItemFn) {
        self.depth += 1;
        syn::visit_mut::visit_trait_item_fn_mut(self, i);
        self.depth -= 1;
    }
}

struct DemutSelfVisitor;

impl VisitMut for DemutSelfVisitor {
    fn visit_fn_arg_mut(&mut self, i: &mut FnArg) {
        if let FnArg::Receiver(rcv) = i
            && rcv.reference.is_none()
        {
            rcv.mutability = None;
        }
        syn::visit_mut::visit_fn_arg_mut(self, i);
    }
}

// -----------------------------------------------------------------------------
// Generators
// -----------------------------------------------------------------------------

fn gen_mod_def(attr: &DispatchAttributes, mod_item: &mut ItemMod) {
    if take_attribute(&mut mod_item.attrs, SKIP_DISPATCH) {
        return;
    }

    if let Some((_, ref mut items)) = mod_item.content {
        for item in items {
            match item {
                Item::Fn(fn_item) => gen_function(attr, fn_item),
                Item::Impl(impl_block) => gen_impl_block(attr, impl_block),
                Item::Mod(module) => gen_mod_def(attr, module),
                _ => {}
            }
        }
    }
}

fn gen_impl_block(attr: &DispatchAttributes, item_impl: &mut ItemImpl) {
    if take_attribute(&mut item_impl.attrs, SKIP_DISPATCH) {
        return;
    }

    let simd = &attr.simd;
    let thermite = &attr.thermite;

    let mut tyv = TypeVisitor {
        self_ty: item_impl.self_ty.clone(),
    };

    let (impl_generics, type_generics, where_clause) = item_impl.generics.split_for_impl();
    let self_ty = &item_impl.self_ty;

    let extra_bounds = where_clause.map(|wc| {
        let mut extra_bounds = Vec::new();
        for pred in wc.predicates.iter() {
            if let WherePredicate::Type(ty) = pred
                && ty.bounded_ty == **self_ty
            {
                extra_bounds.push(&ty.bounds);
            }
        }
        quote! { : #(#extra_bounds +)* }
    });

    for item in &mut item_impl.items {
        if let ImplItem::Fn(f) = item {
            if take_attribute(&mut f.attrs, SKIP_DISPATCH) {
                continue;
            }

            let sig = &f.sig;
            let asyncness = &sig.asyncness;
            let abi = &sig.abi;
            let ident = &sig.ident;
            let fn_generics = &sig.generics;
            let inputs = &sig.inputs;
            let output = &sig.output;
            let defaultness = &f.defaultness;

            let helper_trait_name = quote::format_ident!("__DispatchHelper_{}", ident);

            let mut decl_sig = sig.clone();
            DemutSelfVisitor.visit_signature_mut(&mut decl_sig);

            tyv.visit_return_type_mut(&mut decl_sig.output);
            for arg in decl_sig.inputs.iter_mut() {
                tyv.visit_fn_arg_mut(arg);
            }
            if let Some(wc) = &mut decl_sig.generics.where_clause {
                tyv.visit_where_clause_mut(wc);
            }

            // Disambiguate recursive calls if implementing a trait
            if let Some((_, trait_, _)) = &item_impl.trait_ {
                SelfTraitVisitor::new(trait_.clone(), self_ty.clone(), ident.clone()).visit_block_mut(&mut f.block);
            }

            let (fn_impl_generics, _, fn_where_clause) = fn_generics.split_for_impl();

            let branch_defs = BACKENDS.iter().map(|b| {
                let dispatch_ident = format_backend(b.isa);
                let decl_inputs = &decl_sig.inputs;
                let decl_output = &decl_sig.output;
                let decl_wc = &decl_sig.generics.where_clause;

                quote! {
                    #asyncness unsafe #abi fn #dispatch_ident #fn_impl_generics(#decl_inputs) #decl_output #decl_wc;
                }
            });

            let dispatch_trait = quote! {
                #[allow(non_camel_case_types)] #[allow(clippy::missing_safety_doc)]
                unsafe trait #helper_trait_name #impl_generics #extra_bounds #where_clause {
                    #defaultness #decl_sig;
                    #(#branch_defs)*
                }
            };

            let forward_args = forward_args(inputs.iter(), false);
            let forward_tys = forward_tys(
                fn_generics.params.iter(),
                inputs.iter(),
                fn_generics.where_clause.as_ref(),
            );

            let tf = quote! { ::<#(#forward_tys),*> };

            let branch_impls = BACKENDS.iter().map(|b| {
                let dispatch_ident = format_backend(b.isa);

                let target_feature_attr = if b.target_feature.is_empty() {
                    quote! {}
                } else {
                    let feat = b.target_feature;
                    quote! { #[target_feature(enable = #feat)] }
                };

                quote! {
                    #[inline] #[allow(clippy::missing_safety_doc)]
                    #target_feature_attr
                    #asyncness unsafe #abi fn #dispatch_ident #fn_impl_generics(#inputs) #output #fn_where_clause {
                        <Self as #helper_trait_name #type_generics>::#ident #tf(#(#forward_args,)*)
                    }
                }
            });

            let original_block = &f.block;

            let dispatch_impl = quote! {
                unsafe impl #impl_generics #helper_trait_name #type_generics for #self_ty #where_clause {
                    #[inline(always)]
                    #defaultness #sig #original_block
                    #(#branch_impls)*
                }
            };

            let branches = BACKENDS.iter().map(|b| {
                let dispatch_ident = format_backend(b.isa);
                let backend_ident = quote::format_ident!("{}", b.isa);

                quote! {
                    #thermite::InstructionSet::#backend_ident => unsafe {
                        <Self as #helper_trait_name #type_generics>::#dispatch_ident #tf(#(#forward_args,)*)
                    }
                }
            });

            f.block = syn::parse_quote! {{
                #dispatch_trait
                #dispatch_impl

                match const { <#simd as #thermite::HasIsa>::ISA } {
                    #(#branches)*
                    _ => unsafe { ::core::hint::unreachable_unchecked() }
                }
            }};
        }
    }
}

fn gen_function(attr: &DispatchAttributes, f: &mut ItemFn) {
    if take_attribute(&mut f.attrs, SKIP_DISPATCH) {
        return;
    }

    let simd = &attr.simd;
    let thermite = &attr.thermite;

    let sig = &f.sig;
    let asyncness = &sig.asyncness;
    let unsafety = &sig.unsafety;
    let abi = &sig.abi;
    let ident = &sig.ident;
    let generics = &sig.generics;
    let inputs = &sig.inputs;
    let output = &sig.output;

    let (impl_generics, _, where_clause) = generics.split_for_impl();

    let forward_args = forward_args(inputs.iter(), false);
    let forward_tys = forward_tys(generics.params.iter(), inputs.iter(), generics.where_clause.as_ref());

    let original_block = &f.block;

    let inner = quote! {
        #[inline(always)]
        #asyncness #unsafety #abi fn #ident #impl_generics(#inputs) #output #where_clause #original_block
    };

    let tf = quote! { ::<#(#forward_tys),*> };

    let mut branches = Vec::new();

    for b in BACKENDS {
        let dispatch_ident = format_backend(b.isa);
        let backend_ident = quote::format_ident!("{}", b.isa);

        let target_feature = if b.target_feature.is_empty() {
            quote! {}
        } else {
            let feat = b.target_feature;
            quote! { #[target_feature(enable = #feat)] }
        };

        branches.push(quote! {
            #thermite::InstructionSet::#backend_ident => {
                #[inline] #target_feature
                #asyncness unsafe fn #dispatch_ident #impl_generics(#inputs) #output #where_clause {
                    #ident #tf (#(#forward_args,)*)
                }

                unsafe { #dispatch_ident #tf (#(#forward_args,)*) }
            }
        });
    }

    *f.block = syn::parse_quote! {{
        #inner

        match const { <#simd as #thermite::HasIsa>::ISA } {
            #(#branches,)*
            _ => unsafe { ::core::hint::unreachable_unchecked() }
        }
    }};
}

fn gen_trait_def(_attr: &DispatchAttributes, trait_item: &mut ItemTrait) {
    for item in &mut trait_item.items {
        if let TraitItem::Fn(f) = item {
            take_attribute(&mut f.attrs, SKIP_DISPATCH);
        }
    }
}

fn format_backend(backend: &str) -> Ident {
    quote::format_ident!("__dispatch_{}", backend.to_lowercase())
}

fn forward_tys<'a>(
    fn_generics: impl IntoIterator<Item = &'a GenericParam>,
    inputs: impl IntoIterator<Item = &'a FnArg> + Clone,
    where_clause: Option<&'a WhereClause>,
) -> Vec<TokenStream> {
    let inputs: Vec<&'a FnArg> = inputs.into_iter().collect();
    let mut forward_tys = Vec::new();

    for generic in fn_generics.into_iter() {
        forward_tys.push(match generic {
            GenericParam::Type(ty) => {
                let ident = &ty.ident;
                quote! { #ident }
            }
            GenericParam::Lifetime(lf) => {
                if late_bound::is_late_bound(lf, where_clause, inputs.iter().copied()) {
                    continue;
                }

                let lifetime = &lf.lifetime;
                quote! { #lifetime }
            }
            GenericParam::Const(c) => {
                let ident = &c.ident;
                quote! { #ident }
            }
        });
    }

    forward_tys
}

fn forward_args<'a>(inputs: impl IntoIterator<Item = &'a FnArg>, _inner: bool) -> Vec<TokenStream> {
    let mut forward_args = Vec::new();

    for input in inputs {
        forward_args.push(match input {
            FnArg::Receiver(rcv) => {
                // Extract the exact self_token to preserve its hygiene span
                let self_token = &rcv.self_token;
                quote! { #self_token }
            }

            FnArg::Typed(arg) => {
                if let Pat::Ident(ref param) = *arg.pat {
                    let ident = &param.ident;
                    quote! { #ident }
                } else {
                    // Bubble up a spanned compiler error instead of panicking the rustc process
                    syn::Error::new_spanned(arg, "Unsupported pattern type in arguments. Expected Ident.")
                        .into_compile_error()
                }
            }
        })
    }

    forward_args
}

/// Rewrites bare `f32x4`-style type references inside a `dispatch_dyn!` body to their
/// fully-qualified form `#thermite::Vector<S::f32x4>`.
///
/// Only bare, single-segment, unqualified paths with **no** generic arguments are
/// matched. Multi-segment paths (`my_mod::f32x4`), paths that already carry generic
/// arguments (`f32x4::<Something>`), and qualified self-types (`<f32x4 as Trait>`) are
/// all left untouched.
struct SimdTypeReplacer<'a> {
    thermite: &'a TokenStream,
    /// The identifier bound to the dispatch type parameter (e.g. `S` from `for<S>`).
    dispatch_ident: &'a Ident,
}

impl SimdTypeReplacer<'_> {
    fn is_simd_vector_type(ident: &Ident) -> bool {
        SIMD_VECTOR_TYPES.iter().any(|&name| ident == name)
    }
}

impl VisitMut for SimdTypeReplacer<'_> {
    /// Handles type positions: `let x: f32x4`, `-> f32x4`, function parameters, etc.
    fn visit_type_mut(&mut self, ty: &mut Type) {
        if let Type::Path(p) = &*ty
            && p.qself.is_none()
            && p.path.leading_colon.is_none()
            && p.path.segments.len() == 1
        {
            let seg = &p.path.segments[0];
            if matches!(seg.arguments, PathArguments::None) && Self::is_simd_vector_type(&seg.ident) {
                // Clone the ident before we drop the shared borrow on `ty`.
                let ident = &seg.ident;
                let span = ident.span();
                let thermite = &self.thermite;
                let s = &self.dispatch_ident;

                *ty = syn::parse_quote_spanned! { span => #thermite::Vector<#s::#ident> };

                // Do not recurse — the replacement is already fully expanded.
                return;
            }
        }

        syn::visit_mut::visit_type_mut(self, ty);
    }

    /// Handles expression positions: `f32x4::splat(1.0)`, `f32x4::ZERO`, etc.
    ///
    /// `f32x4::method` is an `Expr::Path` with two segments; the SIMD name is NOT in a
    /// type position so `visit_type_mut` never sees it.  We rewrite
    /// `f32x4::rest` → `<#thermite::Vector<S::f32x4>>::rest`.
    ///
    /// A bare `f32x4` with no further segments is not a valid standalone expression and
    /// is left alone (it would be a compile error regardless).
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Path(p) = &*expr
            && p.qself.is_none()
            && p.path.leading_colon.is_none()
            && p.path.segments.len() >= 2
        {
            let first = &p.path.segments[0];
            if matches!(first.arguments, PathArguments::None) && Self::is_simd_vector_type(&first.ident) {
                let ident = &first.ident;
                let span = ident.span();
                let thermite = &self.thermite;
                let s = &self.dispatch_ident;
                let simd_ty: Type = syn::parse_quote_spanned! { span => #thermite::Vector<#s::#ident> };

                // Collect everything after the SIMD type name and build
                // `<::thermite::Vector<S::f32x4>>::remaining::path` via parse_quote!.
                let remaining: Punctuated<PathSegment, Token![::]> = p.path.segments.iter().skip(1).cloned().collect();

                *expr = syn::parse_quote_spanned! { span => <#simd_ty>::#remaining };

                // Do not recurse into the replacement.
                return;
            }
        }

        syn::visit_mut::visit_expr_mut(self, expr);
    }
}

/// Input syntax for `dispatch_dyn!`:
///
/// ```text
/// dispatch_dyn!(
///     [thermite = "path";]
///     [for<Ident [: Bound [+ Bound]*]>]
///     [<ExtraGenericParams>]
///     |arg: Type, ...|
///     [-> ReturnType]
///     [where ExtraWherePredicates]
///     { body }
/// )
/// ```
///
/// `body` may reference the `for<Ident>` binding as a generic type satisfying the stated
/// bound (default `Simd3A`), as well as any extra generic parameters listed in
/// `<ExtraGenericParams>` (assumed to be in scope at the call site).
struct DispatchDynInput {
    /// Path to the thermite crate root (defaults to `::thermite`).
    thermite: TokenStream,
    /// The identifier bound to the runtime-dispatched `Simd` type (from `for<S>`).
    /// Defaults to `S` if the `for<…>` clause is omitted.
    dispatch_ident: Ident,
    /// Explicit trait bounds on the dispatch type parameter (from `for<S: Bound + …>`).
    /// If empty, the generated code defaults to `Simd3A`.
    dispatch_bounds: Punctuated<TypeParamBound, Token![+]>,
    /// Zero or more extra generic parameters (`<T: Bound, const N: usize>`, etc.) that
    /// are threaded through the generated inner function and per-backend wrappers.
    /// The `where_clause` field on this `Generics` carries any `where` predicates.
    extra_generics: syn::Generics,
    inputs: Punctuated<FnArg, Token![,]>,
    output: ReturnType,
    body: syn::Block,
}

impl Parse for DispatchDynInput {
    fn parse(stream: ParseStream) -> syn::Result<Self> {
        let mut thermite = quote! { ::thermite };

        // Optional `thermite = "some::path";` prefix.
        // Detected unambiguously: if the stream starts with `Ident` then `=`, it must
        // be the config prefix because the only other valid first token is `|` or `<`.
        if stream.peek(syn::Ident) && stream.peek2(Token![=]) {
            let id: syn::Ident = stream.parse()?;
            if id != "thermite" {
                return Err(syn::Error::new(
                    id.span(),
                    "expected `thermite = \"crate-path\";` or `(args...)`",
                ));
            }
            stream.parse::<Token![=]>()?;
            let lit: syn::LitStr = stream.parse()?;
            stream.parse::<Token![;]>()?;
            let path: syn::Path = syn::parse_str(&lit.value()).map_err(|e| syn::Error::new(lit.span(), e))?;
            thermite = quote! { #path };
        }

        // Optional `for<S>` or `for<S: Bound + Bound2>` — the dispatch type binding.
        // Detected unambiguously: `for` keyword followed by `<`.
        // Defaults to the identifier `S` with no explicit bounds (→ `Simd3A` at codegen time).
        let (dispatch_ident, dispatch_bounds) = if stream.peek(Token![for]) && stream.peek2(Token![<]) {
            stream.parse::<Token![for]>()?;
            stream.parse::<Token![<]>()?;
            let ty_param: syn::TypeParam = stream.parse()?;
            if ty_param.eq_token.is_some() {
                return Err(syn::Error::new(
                    ty_param.ident.span(),
                    "default types (`= Type`) are not allowed in `for<…>` dispatch binding",
                ));
            }
            stream.parse::<Token![>]>()?;
            (ty_param.ident, ty_param.bounds)
        } else {
            (Ident::new("S", proc_macro2::Span::call_site()), Punctuated::new())
        };

        // Optional `<ExtraGenericParams>`.
        let mut extra_generics: syn::Generics = if stream.peek(Token![<]) {
            stream.parse()?
        } else {
            syn::Generics::default()
        };

        // |arg: Type, ...|
        stream.parse::<Token![|]>()?;
        let mut inputs = Punctuated::<FnArg, Token![,]>::new();
        while !stream.peek(Token![|]) {
            inputs.push_value(stream.parse()?);
            if stream.peek(Token![|]) {
                break;
            }
            inputs.push_punct(stream.parse()?);
        }
        stream.parse::<Token![|]>()?;

        // optional `-> ReturnType`
        let output: ReturnType = stream.parse()?;

        // Optional `where ExtraWherePredicates` — attached to `extra_generics`.
        if stream.peek(Token![where]) {
            extra_generics.where_clause = Some(stream.parse()?);
        }

        // `{ body }`
        let body: syn::Block = stream.parse()?;

        Ok(Self {
            thermite,
            dispatch_ident,
            dispatch_bounds,
            extra_generics,
            inputs,
            output,
            body,
        })
    }
}

/// Returns `#thermite::#path_str` as a `TokenStream`, where `path_str` is a
/// `"::"` separated path relative to the thermite crate root.
fn backend_type_path(thermite: &TokenStream, path_str: &str) -> TokenStream {
    let path: syn::Path = syn::parse_str(path_str).expect("invalid backend simd_type path");
    quote! { #thermite::#path }
}

/// Runtime ISA-dispatched expression.
///
/// Wraps the body in an `#[inline(always)]` inner function generic over `S: Simd` (plus
/// any caller-supplied extra generics), creates `#[target_feature]`-annotated wrappers
/// for each backend that has a complete [`Simd`] implementation, then dispatches at
/// runtime via `InstructionSet::get()`.
///
/// The syntax is similar to closures, but captures are done via arguments, and must be typed.
///
/// # Syntax
///
/// ```rust,ignore
/// // Basic form — no explicit dispatch binding, SIMD type rewriting only:
/// dispatch_dyn!(|arg: Type, ...| -> ReturnType { body })
///
/// // Explicit dispatch binding (recommended): `for<Ident>` names the backend type.
/// // Default bound is `Simd3A`:
/// dispatch_dyn!(for<S> |arg: f32x4| -> f32x4 { arg })
///
/// // Custom bound — restrict or widen the set of usable Simd traits:
/// dispatch_dyn!(for<S: Simd> |arg: f32x4| -> f32x4 { arg })
/// dispatch_dyn!(for<S: Simd3A + MyCustomTrait> |arg: f32x4| -> f32x4 { arg })
///
/// // With extra caller-provided generics and a where clause:
/// dispatch_dyn!(for<S> <T: Clone, const N: usize> |arg: T| -> T where T: Debug { body })
///
/// // Override the thermite crate path (needed when calling from inside thermite itself):
/// dispatch_dyn!(thermite = "crate"; for<S> |arg: f32x4| -> f32x4 { arg })
/// ```
///
/// When `for<Ident>` is present, `Ident` is in scope inside `body` as a generic type
/// satisfying the stated bound (or `Simd3A` by default).  When omitted, no explicit
/// dispatch binding is in scope — rely on the automatic SIMD type rewriting below.
///
/// Any extra generic parameters from `<…>` are assumed to be in scope at the macro call
/// site; the macro passes them through as explicit turbofish arguments.
///
/// # Automatic SIMD type rewriting
///
/// Before code generation the macro rewrites every **bare, unqualified** reference to a
/// known `Simd` associated-type name into its fully-qualified `Vector<S::…>` form.
/// For example, `f32x4` becomes `::thermite::Vector<S::f32x4>`.
///
/// The full set of names that trigger rewriting is every associated type declared on the
/// `Simd` trait: `{f32,f64,i32,u32,i64,u64,usize}x{N,2,3A,4,8,16}`
///
/// The rewriting fires in all standard Rust type positions (annotations, return types,
/// generic arguments, `as` casts, trait bounds, `where` clauses, fn-pointer types) as
/// well as in **expression paths** such as `f32x4::splat(1.0)` or `f32x4::ZERO`, which
/// become `<::thermite::Vector<S::f32x4>>::splat(1.0)` and
/// `<::thermite::Vector<S::f32x4>>::ZERO` respectively.
///
/// **Explicit references are left untouched.** `S::f32x4`, `<S as Simd>::f32x4`, and
/// any multi-segment path (`my_mod::f32x4`) are not rewritten, so you can always opt
/// out by being explicit.
///
/// Macro invocations (`some_macro!(f32x4)`) are opaque to the rewriter and are also
/// left untouched.
#[proc_macro]
pub fn dispatch_dyn(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let DispatchDynInput {
        thermite,
        dispatch_ident,
        dispatch_bounds,
        extra_generics,
        inputs,
        output,
        mut body,
    } = syn::parse_macro_input!(input as DispatchDynInput);

    // Rewrite bare SIMD type names (e.g. `f32x4`) in the body to their fully-qualified
    // `Vector<dispatch_ident::…>` form before any code generation happens.
    SimdTypeReplacer {
        thermite: &thermite,
        dispatch_ident: &dispatch_ident,
    }
    .visit_block_mut(&mut body);

    // Build the bound for the dispatch type parameter.
    // If the user wrote `for<S: Bound>` use that; otherwise default to `Simd3A`.
    let dispatch_bound: TokenStream = if dispatch_bounds.is_empty() {
        quote! { #thermite::simd::Simd3A }
    } else {
        quote! { #dispatch_bounds }
    };

    let fwd_args = forward_args(inputs.iter(), false);

    let extra_params = &extra_generics.params;
    let where_clause = &extra_generics.where_clause;

    // Forwarding turbofish arguments (type, lifetime, const) for the extra generic params.
    let extra_fwd_tys = forward_tys(
        extra_generics.params.iter(),
        inputs.iter(),
        extra_generics.where_clause.as_ref(),
    );

    // Turbofish for calling `__dispatch_dyn_inner::<BackendType [, T, N, ...]>(args)`
    let inner_tf = |simd_ty: &TokenStream| -> TokenStream {
        quote! { ::<#simd_ty, #(#extra_fwd_tys),*> }
    };

    // For each backend that has a concrete Simd type, generate a target-feature-annotated
    // wrapper that instantiates the inner function for that backend's concrete Simd type.
    let branches = BACKENDS.iter().filter_map(|b| {
        let simd_path_str = b.simd_type?;
        // Skip the scalar backend — it will be emitted as the `_ =>` fallback arm.
        if b.target_feature.is_empty() {
            return None;
        }
        let dispatch_ident = format_backend(b.isa);
        let isa_ident = quote::format_ident!("{}", b.isa);
        let simd_ty = backend_type_path(&thermite, simd_path_str);
        let feat = b.target_feature;
        let itf = inner_tf(&simd_ty);
        Some(quote! {
            #thermite::isa::InstructionSet::#isa_ident => {
                #[inline]
                #[target_feature(enable = #feat)]
                unsafe fn #dispatch_ident <#extra_params> (#inputs) #output #where_clause {
                    __dispatch_dyn_inner #itf (#(#fwd_args,)*)
                }
                unsafe { #dispatch_ident ::<#(#extra_fwd_tys),*> (#(#fwd_args,)*) }
            }
        })
    });

    // Scalar fallback used for the `_ =>` arm.
    let scalar_ty = BACKENDS
        .iter()
        .find(|b| b.target_feature.is_empty() && b.simd_type.is_some())
        .and_then(|b| b.simd_type)
        .unwrap_or("backend::scalar::Scalar");
    let fallback_ty = backend_type_path(&thermite, scalar_ty);
    let fallback_tf = inner_tf(&fallback_ty);

    quote! {{
        use #thermite::prelude::*;

        #[inline(always)]
        fn __dispatch_dyn_inner <#dispatch_ident: #dispatch_bound, #extra_params> (#inputs) #output #where_clause {
            #body
        }

        match #thermite::isa::InstructionSet::get() {
            #(#branches,)*
            _ => __dispatch_dyn_inner #fallback_tf (#(#fwd_args,)*)
        }
    }}
    .into()
}
