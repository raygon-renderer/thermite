#![allow(unused)]

extern crate proc_macro;

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

use syn::{
    Attribute, ConstParam, Expr, ExprCall, ExprPath, FnArg, GenericArgument, GenericParam, Ident, ImplItem, ImplItemFn,
    Item, ItemFn, ItemImpl, ItemMod, ItemTrait, Lifetime, LifetimeParam, Pat, Path, PathArguments, PathSegment, QSelf,
    ReturnType, Signature, Token, TraitItem, Type, WhereClause, WherePredicate,
    parse::{Parse, ParseStream, Parser as _},
    punctuated::Punctuated,
    visit_mut::VisitMut,
};

mod late_bound;

const SKIP_DISPATCH: &'static str = "skip_dispatch";

cfg_if::cfg_if! {
    if #[cfg(feature = "x86")] {
        static BACKENDS: &[(&str, &str)] = &[
            ("Scalar", ""),
            ("X86V1", "sse2"),
            ("X86V2", "sse4.2"),
            ("X86V3", "avx2,fma"),
        ];
    } else if #[cfg(feature = "neon")] {
        static BACKENDS: &[(&str, &str)] = &[("Scalar", ""), ("NEON", "neon")];
    } else if #[cfg(feature = "wasm")] {
        static BACKENDS: &[(&str, &str)] = &[("Scalar", ""), ("WASM32", "simd128")];
    } else if #[cfg(feature = "spirv")] {
        static BACKENDS: &[(&str, &str)] = &[("SPIRV", "")];
    } else {
        static BACKENDS: &[(&str, &str)] = &[("Scalar", "")];
    }
}

/// Holds the directly parsed attributes (no intermediate Punctuated tree).
struct DispatchAttributes {
    simd: TokenStream,
    thermite: TokenStream,
}

/// Generates monomorphized backend `target_feature` function calls to the annotated function or `impl` block.
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

            let branch_defs = BACKENDS.iter().map(|(backend, _)| {
                let dispatch_ident = format_backend(backend);
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

            let branch_impls = BACKENDS.iter().map(|(backend, instrset)| {
                let dispatch_ident = format_backend(backend);

                let target_feature = if instrset.is_empty() {
                    quote! {}
                } else {
                    quote! { #[target_feature(enable = #instrset)] }
                };

                quote! {
                    #[inline] #[allow(clippy::missing_safety_doc)]
                    #target_feature
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

            let branches = BACKENDS.iter().map(|(backend, _)| {
                let dispatch_ident = format_backend(backend);
                let backend = quote::format_ident!("{}", backend);

                quote! {
                    #thermite::InstructionSet::#backend => unsafe {
                        <Self as #helper_trait_name #type_generics>::#dispatch_ident #tf(#(#forward_args,)*)
                    }
                }
            });

            f.block = syn::parse_quote! {{
                #dispatch_trait
                #dispatch_impl

                match <#simd as #thermite::HasIsa>::ISA {
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

    for (backend, instrset) in BACKENDS {
        let dispatch_ident = format_backend(backend);
        let backend = quote::format_ident!("{}", backend);

        let target_feature = if instrset.is_empty() {
            quote! {}
        } else {
            quote! { #[target_feature(enable = #instrset)] }
        };

        branches.push(quote! {
            #thermite::InstructionSet::#backend => {
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

        match <#simd as #thermite::HasIsa>::ISA {
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
