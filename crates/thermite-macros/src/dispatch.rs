use proc_macro2::{Group, TokenStream, TokenTree};
use quote::{ToTokens, quote};

use syn::{
    Attribute, Expr, ExprCall, ExprPath, FnArg, GenericParam, Ident, ImplItem, Item, ItemFn, ItemImpl, ItemMod,
    ItemTrait, Pat, Path, PathArguments, PathSegment, QSelf, ReceiverKind, ReturnType, Signature, Token, TraitItem,
    Type, TypeParamBound, WhereClause, WherePredicate,
    parse::{Parse, ParseStream, Parser as _},
    punctuated::Punctuated,
    visit_mut::VisitMut,
};

use super::late_bound;

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
    /// implementation (e.g. NEON). Such backends are skipped by `dispatch_dyn!` and
    /// their hardware falls through to the scalar fallback.
    simd_type: Option<&'static str>,
}

/// Target features for the x86-v3 (AVX2 + FMA) backend.
///
/// Every CPU with AVX2 (Haswell, 2013) also has F16C (introduced one generation earlier
/// with Ivy Bridge), so the `avx2-f16c` feature lets us assume F16C is present whenever the
/// AVX2 backend is selected and unconditionally enable the half-precision conversion
/// intrinsics in dispatched code without a separate runtime check. However, there are
/// some AVX2-capable CPUs that do not have F16C, so this remains optional.
/// `avx2-pclmul` additionally assumes PCLMULQDQ (present on every AVX2 CPU - it
/// shipped with Westmere, three years before Haswell), enabling the CLMUL-based 2D
/// Morton fast path on u64-lane registers.
#[cfg(feature = "x86")]
const X86V3_TARGET_FEATURE: &str = cfg_select! {
    all(feature = "avx2-f16c", feature = "avx2-pclmul") => "avx2,fma,popcnt,f16c,pclmulqdq", feature = "avx2-f16c" => "avx2,fma,popcnt,f16c", feature = "avx2-pclmul" => "avx2,fma,popcnt,pclmulqdq",
    _ => "avx2,fma,popcnt",
};

/// Target features for the x86-v4 (AVX-512) backend, per compiled tier. Kept in
/// lockstep BY HAND with the `Avx512Features` consts in `backend::x86_v4`: a
/// const being `true` there does not make an intrinsic callable here. Every tier
/// includes the full v3 set plus `pclmulqdq` (the u64x2 Morton path uses the
/// legacy `_mm_clmulepi64_si128`) and `bmi2` (the opmask interleave uses `pdep`).
#[cfg(all(feature = "x86", feature = "avx512-tier1"))]
const X86V4_TARGET_FEATURE: &str = cfg_select! {
    feature = "avx512-tier3" => "avx2,fma,popcnt,f16c,pclmulqdq,bmi2,avx512f,avx512cd,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,avx512vnni,avx512bitalg,avx512vpopcntdq,avx512ifma,gfni,vaes,vpclmulqdq,avx512bf16",
    feature = "avx512-tier2" => "avx2,fma,popcnt,f16c,pclmulqdq,bmi2,avx512f,avx512cd,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,avx512vnni,avx512bitalg,avx512vpopcntdq,avx512ifma,gfni,vaes,vpclmulqdq",
    _ => "avx2,fma,popcnt,f16c,pclmulqdq,bmi2,avx512f,avx512cd,avx512vl,avx512bw,avx512dq",
};

static BACKENDS: &[Backend] = cfg_select! {
    feature = "x86" => &[
        Backend { isa: "Scalar", target_feature: "", simd_type: Some("backend::scalar::Scalar") },
        Backend { isa: "X86V1", target_feature: "sse2", simd_type: Some("backend::x86_v1::X86V1")  },
        Backend { isa: "X86V2", target_feature: "sse4.2,popcnt", simd_type: Some("backend::x86_v2::X86V2")  },

        // See `X86V3_TARGET_FEATURE` for what the AVX2 rung assumes about F16C/PCLMULQDQ.
        Backend {
            isa: "X86V3", target_feature: X86V3_TARGET_FEATURE, simd_type: Some("backend::x86_v3::X86V3")
        },

        // x86-v4 (AVX-512): the real backend when a tier feature is on. The
        // detector only reports `X86V4` when the CPU satisfies the COMPILED tier,
        // so the trampoline's feature set is always satisfiable there.
        #[cfg(feature = "avx512-tier1")]
        Backend {
            isa: "X86V4", target_feature: X86V4_TARGET_FEATURE, simd_type: Some("backend::x86_v4::X86V4Default")
        },

        // Without a tier feature the x86-v4 registers are not compiled, so V4
        // hardware deliberately maps to the _x86-v3_ backend. Without an arm here
        // it would fall into the `_ =>` scalar fallback and the entire library
        // would run one lane wide: correct, and roughly an order of magnitude
        // slower (verified under Intel SDE on skx/icx/spr/gnr/dmr: `f32xN::LANES`
        // came back 1). Every AVX-512F part is strictly newer than Haswell, so
        // the v3 trampoline's feature set is always satisfiable here.
        #[cfg(not(feature = "avx512-tier1"))]
        Backend {
            isa: "X86V4", target_feature: X86V3_TARGET_FEATURE, simd_type: Some("backend::x86_v3::X86V3")
        },
    ], feature = "neon" => &[
        Backend { isa: "Scalar", target_feature: "", simd_type: Some("backend::scalar::Scalar") },
        // NEON/AdvSIMD is mandatory on aarch64 (the only ARM target the backend
        // supports), so `InstructionSet::get()` is constant and the `neon`
        // target feature is already in the target baseline - the trampoline
        // attribute is a stable no-op.
        Backend { isa: "NEON", target_feature: "neon", simd_type: Some("backend::neon::Neon")     },
    ], feature = "wasm" => &[
        Backend { isa: "Scalar", target_feature: "", simd_type: Some("backend::scalar::Scalar") },
        Backend { isa: "WASM32", target_feature: "simd128", simd_type: Some("backend::wasm::Wasm")     },
    ], feature = "spirv" => &[
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
    "usizex3A", "f32x3A", "i32x3A", "u32x3A", "f64x3A", "i64x3A", "u64x3A", "usizex3", "f32x3", "i32x3", "u32x3",
    "f64x3", "i64x3", "u64x3", "i16x2", "u16x2", "i16x4", "u16x4", "i16x8", "u16x8", "i16x16", "u16x16", "i8x2",
    "u8x2", "i8x4", "u8x4", "i8x8", "u8x8", "i8x16", "u8x16", "i16xN", "u16xN", "i8xN", "u8xN",
];

/// Holds the directly parsed attributes (no intermediate Punctuated tree).
struct DispatchAttributes {
    simd: TokenStream,
    /// `true` when the user did not explicitly specify a dispatch type - `S` is just the default.
    simd_is_default: bool,
    thermite: TokenStream,
}

pub fn dispatch_inner(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut simd = quote! { S };
    let mut simd_is_default = true;
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
            simd_is_default = false;
            Ok(())
        } else {
            Err(meta.error("unsupported dispatch attribute"))
        }
    });

    if let Err(err) = attr_parser.parse(attr) {
        return err.into_compile_error().into();
    }

    let attr_data = DispatchAttributes {
        simd,
        simd_is_default,
        thermite,
    };

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
    #[rustfmt::skip]    fn visit_type_mut(&mut self, i: &mut Type) {
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
                lt_token: Default::default(), ty: self.self_ty.clone(), position: 0, as_token: None, gt_token: Default::default(),
            });
        }

        syn::visit_mut::visit_type_mut(self, i);
    }
}

struct SelfTraitVisitor {
    depth: u32,
    method: Ident,
    trait_: Path,
    qself: QSelf,
}

impl SelfTraitVisitor {
    fn new(trait_: Path, self_ty: Box<Type>, method: Ident) -> SelfTraitVisitor {
        let qself = QSelf {
            lt_token: Default::default(),
            ty: self_ty,
            position: trait_.segments.len(),
            as_token: Some(Default::default()),
            gt_token: Default::default(),
        };

        SelfTraitVisitor {
            depth: 0,
            method,
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
            && matches!(rcv.kind, ReceiverKind::Value)
        {
            rcv.mutability = None;
        }
        syn::visit_mut::visit_fn_arg_mut(self, i);
    }
}

// -----------------------------------------------------------------------------
// Generators
// -----------------------------------------------------------------------------

/// Signature properties that cannot survive being rebuilt as a dispatch trampoline.
///
/// `#[dispatch]` re-emits each function as one `#[target_feature]` copy per backend
/// plus a const-folded ISA match. Two properties are incompatible with that:
///
/// - **`const`** - a `#[target_feature]` fn cannot be `const`, so the per-backend
///   copies could never be const even if the trampolines did forward it.
/// - **variadic `...`**. Only legal in an `extern` fn, the generated trampolines
///   would not be well-formed.
///
/// Neither was ever forwarded, so both used to be dropped _silently_: a
/// `#[dispatch] const fn` came out non-const with no diagnostic, and only failed
/// later at some unrelated const-context call site. Reject them here instead,
/// in the spirit of syn 3's `Modifiers::require_empty()`. Refuse syntax the macro
/// does not understand rather than quietly eating it.
///
/// Checked _after_ `skip_dispatch`, so a skipped function keeps both. Nothing is
/// rebuilt for it, and the item is emitted from the mutated AST unchanged.
fn reject_unsupported_signature(sig: &Signature) -> Option<TokenStream> {
    if let Some(constness) = &sig.constness {
        return Some(
            syn::Error::new_spanned(
                constness,
                "#[dispatch] cannot be applied to a `const fn`: the generated per-backend copies carry #[target_feature], which a `const fn` cannot. Remove `const`, or mark this function #[skip_dispatch].",
            )
            .into_compile_error(),
        );
    }

    if let Some(variadic) = &sig.variadic {
        return Some(
            syn::Error::new_spanned(
                variadic,
                "#[dispatch] cannot be applied to a variadic function: `...` is only legal in an `extern` fn, and the generated per-backend copies would not be well-formed. Mark this function #[skip_dispatch].",
            )
            .into_compile_error(),
        );
    }

    None
}

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

            if let Some(err) = reject_unsupported_signature(&f.sig) {
                f.block = syn::parse_quote! {{ #err }};
                continue;
            }

            let sig = &f.sig;
            let asyncness = &sig.asyncness;
            let abi = &sig.abi;
            let ident = &sig.ident;
            let fn_generics = &sig.generics;
            let inputs = &sig.inputs;
            let output = &sig.output;
            let defaultness = &f.modifiers.defaultness;

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
            if let Some((trait_, _)) = &item_impl.trait_ {
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

    if let Some(err) = reject_unsupported_signature(&f.sig) {
        *f.block = syn::parse_quote! {{ #err }};
        return;
    }

    let thermite = &attr.thermite;

    let sig = &f.sig;
    let asyncness = &sig.asyncness;
    let safety = &sig.safety;
    let abi = &sig.abi;
    let ident = &sig.ident;
    let generics = &sig.generics;
    let inputs = &sig.inputs;
    let output = &sig.output;

    let (impl_generics, _, where_clause) = generics.split_for_impl();

    let forward_args = forward_args(inputs.iter(), false);
    let forward_tys = forward_tys(generics.params.iter(), inputs.iter(), generics.where_clause.as_ref());
    let tf = quote! { ::<#(#forward_tys),*> };

    // When applied to a method inside an impl block, the function has a receiver (`&self`,
    // `&mut self`, or `self`).  Free-function inner copies can't have receivers, so we
    // replicate the helper-trait pattern from `gen_impl_block` here.  Unlike that path,
    // we only see the method - not the surrounding `impl` block - so the concrete Self
    // type is unknown.  The user must supply it via `#[dispatch(ConcreteType)]`; that
    // ident is used both as the dispatch match type and as the impl target.
    // `impl Trait for Self` is not valid inside a function body.
    let has_receiver = inputs.iter().any(|arg| matches!(arg, FnArg::Receiver(_)));
    if has_receiver {
        if attr.simd_is_default {
            *f.block = syn::parse_quote! {{
                compile_error!(
                    "#[dispatch] on a method with a receiver requires the concrete type name. \
                    Either annotate the whole impl block (`#[dispatch(Self)] impl Type { ... }`), \
                    or supply the type to this attribute: `#[dispatch(TypeName)]`."
                );
            }};
            return;
        }
        // If the supplied dispatch ident is one of this method's own generic type
        // parameters (e.g. `fn my_method<S: Simd>(&self)` with `#[dispatch(S)]`), the
        // user wants to dispatch on the generic `S` while `Self` is some other concrete
        // type the macro cannot see from the method alone. The helper trait must be
        // implemented for `Self`, so this can only be expressed on the impl block.
        if let Ok(simd_ident) = syn::parse2::<Ident>(attr.simd.clone())
            && sig.generics.type_params().any(|tp| tp.ident == simd_ident)
        {
            *f.block = syn::parse_quote! {{
                compile_error!(
                    "#[dispatch(S)] names a generic type parameter of this method, so the \
                     concrete `Self` type is unknown and the dispatch helper cannot be \
                     implemented for it. Annotate the whole impl block instead: \
                     `#[dispatch(S)] impl Type { ... }` - there the dispatch type and `Self` \
                     are tracked separately."
                );
            }};
            return;
        }

        let simd = attr.simd.clone();

        let helper_trait_name = quote::format_ident!("__DispatchHelper_{}", ident);

        let mut decl_sig = sig.clone();
        DemutSelfVisitor.visit_signature_mut(&mut decl_sig);

        let branch_defs = BACKENDS.iter().map(|b| {
            let dispatch_ident = format_backend(b.isa);
            let decl_inputs = &decl_sig.inputs;
            let decl_output = &decl_sig.output;
            let decl_wc = &decl_sig.generics.where_clause;
            quote! {
                #asyncness unsafe #abi fn #dispatch_ident #impl_generics(#decl_inputs) #decl_output #decl_wc;
            }
        });

        let dispatch_trait = quote! {
            #[allow(non_camel_case_types)] #[allow(clippy::missing_safety_doc)]
            unsafe trait #helper_trait_name {
                #decl_sig;
                #(#branch_defs)*
            }
        };

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
                #asyncness unsafe #abi fn #dispatch_ident #impl_generics(#inputs) #output #where_clause {
                    <Self as #helper_trait_name>::#ident #tf(#(#forward_args,)*)
                }
            }
        });

        let original_block = &f.block;

        let dispatch_impl = quote! {
            unsafe impl #helper_trait_name for #simd {
                #[inline(always)]
                #sig #original_block
                #(#branch_impls)*
            }
        };

        let branches = BACKENDS.iter().map(|b| {
            let dispatch_ident = format_backend(b.isa);
            let backend_ident = quote::format_ident!("{}", b.isa);
            quote! {
                #thermite::InstructionSet::#backend_ident => unsafe {
                    <Self as #helper_trait_name>::#dispatch_ident #tf(#(#forward_args,)*)
                }
            }
        });

        *f.block = syn::parse_quote! {{
            #dispatch_trait
            #dispatch_impl
            match const { <#simd as #thermite::HasIsa>::ISA } {
                #(#branches)*
                _ => unsafe { ::core::hint::unreachable_unchecked() }
            }
        }};

        return;
    }

    let simd = &attr.simd;

    let original_block = &f.block;

    let inner = quote! {
        #[inline(always)]
        #asyncness #safety #abi fn #ident #impl_generics(#inputs) #output #where_clause #original_block
    };

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
    forward_args_impl(inputs, false)
}

/// Like [`forward_args`], but for argument-typed `&T` / `&mut T` parameters, emits
/// `&ident` / `&mut ident` instead of a bare `ident`. Used at the outermost
/// `dispatch_dyn!` call sites so the caller can pass an owned value (e.g. `Vec<T>`
/// where the macro signature expects `&[T]`) and have Rust apply deref coercion.
/// Calling `foo(vec)` with `foo(&[T])` does not coerce; `foo(&vec)` does.
fn forward_args_reborrow<'a>(inputs: impl IntoIterator<Item = &'a FnArg>) -> Vec<TokenStream> {
    forward_args_impl(inputs, true)
}

fn forward_args_impl<'a>(inputs: impl IntoIterator<Item = &'a FnArg>, reborrow: bool) -> Vec<TokenStream> {
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
                    // Type-driven reborrow emission for reference parameters at the
                    // outermost macro call site.
                    //
                    //   - Slice / trait-object DST inner type (e.g. `&[T]`, `&mut [T]`,
                    //     `&dyn Trait`): emit `&*ident` / `&mut *ident`. Deref bridges
                    //     owners (`Vec<T>` -> `[T]`, `Box<[T]>` -> `[T]`, `String` ->
                    //     `str`) and reborrows references (`&[T]` -> `&[T]`,
                    //     `&mut [T]` -> `&mut [T]`).
                    //   - Sized inner type (e.g. `&Vec<T>`, `&mut Vec<T>`, `&T`): emit
                    //     plain `&ident` / `&mut ident`. We deliberately avoid `&*ident`
                    //     here because it would invoke `Deref{,Mut}` and overshoot to
                    //     the target type - e.g. `&mut Vec<T>` would dereference into
                    //     `&mut [T]` and fail to match the parameter type.
                    //
                    // Trade-off: for a `&mut T` (sized) parameter, the caller's binding
                    // must be `mut`. A function parameter `fn foo(buf: &mut Vec<f64>)`
                    // is *not* mut-bound by default - pass `mut buf: &mut Vec<f64>` in
                    // the signature, or reborrow at the call site before invoking the
                    // macro (`let buf = &mut *buf;`).
                    match (reborrow, &*arg.ty) {
                        (true, Type::Reference(r)) => {
                            let inner_is_dst = matches!(&*r.elem, Type::Slice(_) | Type::TraitObject(_))
                                || matches!(&*r.elem, Type::Path(p) if p.path.is_ident("str"));
                            match (r.mutability.is_some(), inner_is_dst) {
                                (true, true) => quote! { &mut *#ident },
                                (true, false) => quote! { &mut #ident },
                                (false, true) => quote! { &*#ident },
                                (false, false) => quote! { &#ident },
                            }
                        }
                        _ => quote! { #ident },
                    }
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

                // Do not recurse - the replacement is already fully expanded.
                return;
            }
        }

        syn::visit_mut::visit_type_mut(self, ty);
    }

    /// Handles expression positions: `f32x4::splat(1.0)`, `f32x4::ZERO`, etc.
    ///
    /// `f32x4::method` is an `Expr::Path` with two segments; the SIMD name is NOT in a
    /// type position so `visit_type_mut` never sees it.  We rewrite
    /// `f32x4::rest` -> `<#thermite::Vector<S::f32x4>>::rest`.
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

/// Input syntax for `dispatch_dyn!` - two forms sharing an optional prefix:
///
/// ```text
/// // Closure form: wraps an arbitrary body in per-backend #[target_feature] trampolines.
/// dispatch_dyn!(
///     [thermite = "path";]
///     [for<Ident [: Bound [+ Bound]*]>]
///     [<ExtraGenericParams>]
///     |arg: Type, ...|
///     [-> ReturnType]
///     [where ExtraWherePredicates]
///     { body }
/// )
///
/// // Call form: runtime-dispatches a call to a #[dispatch] function, which carries
/// // its own trampolines. Bare form injects the backend as the sole generic argument;
/// // the for<Ident> form substitutes `Ident` in the call expression.
/// dispatch_dyn!([thermite = "path";] func(args...))
/// dispatch_dyn!([thermite = "path";] for<Ident> expr)
/// ```
enum DispatchDynInput {
    Closure(DispatchDynClosure),
    Call(DispatchDynCall),
}

/// The closure form of `dispatch_dyn!`.
///
/// `body` may reference the `for<Ident>` binding as a generic type satisfying the stated
/// bound (default `Simd3`), as well as any extra generic parameters listed in
/// `<ExtraGenericParams>` (assumed to be in scope at the call site).
struct DispatchDynClosure {
    /// Path to the thermite crate root (defaults to `::thermite`).
    thermite: TokenStream,
    /// The identifier bound to the runtime-dispatched `Simd` type (from `for<S>`).
    /// Defaults to `S` if the `for<...>` clause is omitted.
    dispatch_ident: Ident,
    /// Explicit trait bounds on the dispatch type parameter (from `for<S: Bound + ...>`).
    /// If empty, the generated code defaults to `Simd3`.
    dispatch_bounds: Punctuated<TypeParamBound, Token![+]>,
    /// Zero or more extra generic parameters (`<T: Bound, const N: usize>`, etc.) that
    /// are threaded through the generated inner function and per-backend wrappers.
    /// The `where_clause` field on this `Generics` carries any `where` predicates.
    extra_generics: syn::Generics,
    inputs: Punctuated<FnArg, Token![,]>,
    output: ReturnType,
    body: syn::Block,
}

/// The call form of `dispatch_dyn!`.
///
/// Unlike the closure form, no `#[target_feature]` trampolines are generated: the
/// expansion is a plain `match InstructionSet::get()` whose arms instantiate the
/// expression at each backend's concrete `Simd` type. Correct per-backend codegen
/// therefore relies on the callee being a `#[dispatch]` function (or method), which
/// already carries its own trampolines internally.
///
/// The supported shape is a single dispatched call. The `for<Ident>` substitution is
/// token-level and thus technically works on any expression, but that is deliberately
/// undocumented: code inside the macro that isn't the `#[dispatch]` callee compiles
/// without target features, so anything beyond the call belongs outside the macro.
struct DispatchDynCall {
    /// Path to the thermite crate root (defaults to `::thermite`).
    thermite: TokenStream,
    /// `Some` for the `for<Ident> expr` form: every bare `Ident` token in `expr` is
    /// replaced with the backend type in each arm. `None` for the bare `func(args)`
    /// form, where the backend type is injected as the callee's sole generic argument.
    binder: Option<Ident>,
    expr: Expr,
}

impl Parse for DispatchDynInput {
    fn parse(stream: ParseStream) -> syn::Result<Self> {
        let mut thermite = quote! { ::thermite };

        // Optional `thermite = "some::path";` prefix.
        // The `Ident =` lookahead cannot be confused with a call-form expression:
        // `ident = ...` is an assignment, which is not a meaningful dispatch target.
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

        // Optional `for<S>` or `for<S: Bound + Bound2>` - the dispatch type binding.
        // Detected unambiguously: `for` keyword followed by `<`.
        let mut explicit_binder = None;
        if stream.peek(Token![for]) && stream.peek2(Token![<]) {
            stream.parse::<Token![for]>()?;
            stream.parse::<Token![<]>()?;
            let ty_param: syn::TypeParam = stream.parse()?;
            if ty_param.default.is_some() {
                return Err(syn::Error::new(
                    ty_param.ident.span(),
                    "default types (`= Type`) are not allowed in `for<...>` dispatch binding",
                ));
            }
            stream.parse::<Token![>]>()?;
            explicit_binder = Some((ty_param.ident, ty_param.bounds));
        }

        // Distinguish the closure form from the call form. The closure form continues
        // with `|args|` or `<ExtraGenerics> |args|`; anything else is an expression
        // (call form). A leading `<` is ambiguous between extra generics and a
        // qualified-path expression (`<Foo as Bar>::baz(..)`), so speculatively parse
        // generics and require the `|` that must follow them.
        let is_closure = stream.peek(Token![|])
            || (stream.peek(Token![<]) && {
                let fork = stream.fork();
                fork.parse::<syn::Generics>().is_ok() && fork.peek(Token![|])
            });

        if !is_closure {
            let binder = match explicit_binder {
                Some((ident, bounds)) => {
                    if !bounds.is_empty() {
                        return Err(syn::Error::new(
                            ident.span(),
                            "trait bounds on `for<...>` are not supported in the call form: the \
                             binder is substituted with concrete backend types, so the called \
                             function's own bounds apply. Remove the bounds, or use the closure \
                             form (`for<S: Bound> |args| { ... }`)",
                        ));
                    }
                    Some(ident)
                }
                None => None,
            };
            let expr: Expr = stream.parse()?;
            return Ok(DispatchDynInput::Call(DispatchDynCall { thermite, binder, expr }));
        }

        // Defaults to the identifier `S` with no explicit bounds (-> `Simd3` at codegen time).
        let (dispatch_ident, dispatch_bounds) =
            explicit_binder.unwrap_or_else(|| (Ident::new("S", proc_macro2::Span::call_site()), Punctuated::new()));

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

        // Optional `where ExtraWherePredicates` - attached to `extra_generics`.
        if stream.peek(Token![where]) {
            extra_generics.where_clause = Some(stream.parse()?);
        }

        // `{ body }`
        let body: syn::Block = stream.parse()?;

        Ok(DispatchDynInput::Closure(DispatchDynClosure {
            thermite,
            dispatch_ident,
            dispatch_bounds,
            extra_generics,
            inputs,
            output,
            body,
        }))
    }
}

/// Returns `#thermite::#path_str` as a `TokenStream`, where `path_str` is a
/// `"::"` separated path relative to the thermite crate root.
fn backend_type_path(thermite: &TokenStream, path_str: &str) -> TokenStream {
    let path: syn::Path = syn::parse_str(path_str).expect("invalid backend simd_type path");
    quote! { #thermite::#path }
}

pub fn dispatch_dyn_inner(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match syn::parse_macro_input!(input as DispatchDynInput) {
        DispatchDynInput::Closure(closure) => dispatch_dyn_closure(closure),
        DispatchDynInput::Call(call) => dispatch_dyn_call(call),
    }
}

/// Recursively replaces every occurrence of the bare ident `target` in `ts` with the
/// `replacement` tokens, descending into groups. Bumps `count` once per replacement.
fn substitute_ident(ts: TokenStream, target: &str, replacement: &TokenStream, count: &mut usize) -> TokenStream {
    let mut out = TokenStream::new();
    for tt in ts {
        match tt {
            TokenTree::Ident(ref i) if *i == target => {
                *count += 1;
                out.extend(replacement.clone());
            }
            TokenTree::Group(g) => {
                let mut inner = Group::new(g.delimiter(), substitute_ident(g.stream(), target, replacement, count));
                inner.set_span(g.span());
                out.extend([TokenTree::Group(inner)]);
            }
            other => out.extend([other]),
        }
    }
    out
}

/// Builds one match arm's expression for the call form: the input expression
/// instantiated at the given concrete backend type.
fn call_form_arm(binder: Option<&Ident>, expr: &Expr, simd_ty: &TokenStream) -> TokenStream {
    match binder {
        // `for<S> expr`: substitute every bare `S` token with the backend type.
        Some(ident) => {
            let mut count = 0;
            substitute_ident(expr.to_token_stream(), &ident.to_string(), simd_ty, &mut count)
        }
        // Bare `func(args)`: inject the backend type as the callee's sole generic
        // argument (validated in `dispatch_dyn_call`).
        None => {
            let mut call = expr.clone();
            if let Expr::Call(c) = &mut call
                && let Expr::Path(p) = &mut *c.func
                && let Some(last) = p.path.segments.last_mut()
            {
                last.arguments = PathArguments::AngleBracketed(syn::parse_quote! { ::<#simd_ty> });
            }
            call.into_token_stream()
        }
    }
}

/// Expands the call form of `dispatch_dyn!`.
///
/// Emits a `match InstructionSet::get()` whose arms instantiate the expression at each
/// backend's concrete `Simd` type. No `#[target_feature]` wrappers are generated here:
/// a `#[dispatch]` callee already contains its own per-backend trampolines, and the
/// runtime match discharges their feature preconditions. (Calling a non-`#[dispatch]`
/// generic function this way is still *correct*, but its body is compiled without
/// target features - use the closure form to wrap arbitrary code.)
fn dispatch_dyn_call(input: DispatchDynCall) -> proc_macro::TokenStream {
    let DispatchDynCall { thermite, binder, expr } = input;

    // Validate up front so errors surface once, with spans on the user's tokens.
    match &binder {
        Some(ident) => {
            let mut count = 0;
            substitute_ident(
                expr.to_token_stream(),
                &ident.to_string(),
                &TokenStream::new(),
                &mut count,
            );
            if count == 0 {
                return syn::Error::new(
                    ident.span(),
                    format!("the dispatch binder `{ident}` does not appear in the expression"),
                )
                .into_compile_error()
                .into();
            }
        }
        None => {
            let err = |tokens: &dyn ToTokens, msg: &str| -> proc_macro::TokenStream {
                syn::Error::new_spanned(tokens, msg).into_compile_error().into()
            };
            let Expr::Call(call) = &expr else {
                return err(
                    &expr,
                    "expected a plain `function(args)` call; for method calls, mark where the \
                     SIMD type goes with a `for<...>` binder: \
                     `dispatch_dyn!(for<S> receiver.method::<S>(args))`",
                );
            };
            let Expr::Path(path) = &*call.func else {
                return err(
                    &call.func,
                    "the called function must be a plain path; use a `for<...>` binder and \
                     write the SIMD type explicitly: `dispatch_dyn!(for<S> callee::<S>(args))`",
                );
            };
            // A `<T as Trait>::f(..)` callee has explicit generics on the qself, and
            // appending `::<Backend>` to the method segment would misplace them.
            if path.qself.is_some()
                || !matches!(
                    path.path.segments.last().map(|s| &s.arguments),
                    Some(PathArguments::None)
                )
            {
                return err(
                    &call.func,
                    "this callee already has explicit generic arguments; write the SIMD \
                     parameter explicitly with a `for<...>` binder: \
                     `dispatch_dyn!(for<S> func::<S, ...>(args))`",
                );
            }
        }
    }

    // One arm per backend with a concrete, runtime-dispatchable Simd type. The scalar
    // backend (empty target-feature string) becomes the `_ =>` fallback arm.
    let branches = BACKENDS.iter().filter_map(|b| {
        let simd_path_str = b.simd_type?;
        if b.target_feature.is_empty() {
            return None;
        }
        let isa_ident = quote::format_ident!("{}", b.isa);
        let simd_ty = backend_type_path(&thermite, simd_path_str);
        let arm = call_form_arm(binder.as_ref(), &expr, &simd_ty);
        Some(quote! { #thermite::isa::InstructionSet::#isa_ident => #arm })
    });

    let scalar_ty = BACKENDS
        .iter()
        .find(|b| b.target_feature.is_empty() && b.simd_type.is_some())
        .and_then(|b| b.simd_type)
        .unwrap_or("backend::scalar::Scalar");
    let fallback_ty = backend_type_path(&thermite, scalar_ty);
    let fallback = call_form_arm(binder.as_ref(), &expr, &fallback_ty);

    quote! {{
        match #thermite::isa::InstructionSet::get() {
            #(#branches,)*
            _ => #fallback
        }
    }}
    .into()
}

/// Expands the closure form of `dispatch_dyn!`.
fn dispatch_dyn_closure(input: DispatchDynClosure) -> proc_macro::TokenStream {
    let DispatchDynClosure {
        thermite,
        dispatch_ident,
        dispatch_bounds,
        extra_generics,
        inputs,
        output,
        mut body,
    } = input;

    // Rewrite bare SIMD type names (e.g. `f32x4`) in the body to their fully-qualified
    // `Vector<dispatch_ident::...>` form before any code generation happens.
    SimdTypeReplacer {
        thermite: &thermite,
        dispatch_ident: &dispatch_ident,
    }
    .visit_block_mut(&mut body);

    // Build the bound for the dispatch type parameter.
    // If the user wrote `for<S: Bound>` use that; otherwise default to `Simd3`.
    let dispatch_bound: TokenStream = if dispatch_bounds.is_empty() {
        quote! { #thermite::simd::Simd3 }
    } else {
        quote! { #dispatch_bounds }
    };

    let fwd_args = forward_args(inputs.iter(), false);
    // At the outermost call sites we receive identifiers from the macro caller's scope,
    // whose types may differ from the declared parameter types (e.g. caller passes
    // `Vec<f32>` while the parameter is `&[f32]`). Reborrow reference-typed parameters
    // so Rust's deref coercion can bridge the gap.
    let fwd_args_outer = forward_args_reborrow(inputs.iter());

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
        // Skip the scalar backend - it will be emitted as the `_ =>` fallback arm.
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
                unsafe { #dispatch_ident ::<#(#extra_fwd_tys),*> (#(#fwd_args_outer,)*) }
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

        // `#body` is a `syn::Block` and tokenizes as `{ stmts }`, so we use it as the
        // function body directly - wrapping it in another `{ #body }` would produce
        // `fn f() { { user_body } }` and trigger the `unused_braces` lint at the user's
        // call site.
        #[inline(always)]
        fn __dispatch_dyn_inner <#dispatch_ident: #dispatch_bound, #extra_params> (#inputs) #output #where_clause #body

        match #thermite::isa::InstructionSet::get() {
            #(#branches,)*
            _ => __dispatch_dyn_inner #fallback_tf (#(#fwd_args_outer,)*)
        }
    }}
    .into()
}
