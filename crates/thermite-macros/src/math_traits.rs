//! `math_traits!`: the unified `decl_math!` replacement.
//!
//! One function-like macro, consumed by both `thermite` and
//! `thermite-special`, that takes REAL Rust trait declarations and generates
//! the whole math-trait family around them:
//!
//! ```ignore
//! math_traits! {
//!     #![thermite = crate]          // path to the thermite crate (`crate` or `thermite`)
//!     #![scalar = ScalarMath]       // name of the scalar aggregate pair
//!
//!     #[element(FloatElement)]      // bound on E in the blanket impl (optional)
//!     pub trait CoreMath: FloatVector + PrimalProjection {
//!         fn poly(self, coeffs: &[Self::Element]) -> Self;
//!         fn nth_root_n<const N: usize>(self) -> Self;
//!         #[vector_only]            // no scalar_ form (signature names vector-only types)
//!         fn poly_primal(self, coeffs: &[Self::Primal]) -> Self;
//!         #[kind]                   // request-struct method: body is `kind.eval::<P>()`
//!         fn carlson<K: CarlsonKind<Output = Self>>(kind: K) -> Self;
//!         #[scalar_form((self, coeffs: &[Self]) -> Self)] // different scalar spelling
//!         fn legendre_series(self, coeffs: &[Self::Element]) -> Self;
//!     }
//!
//!     scalar_extras {
//!         // verbatim items: `_p`-suffixed go into `{scalar}WithPolicy`, the
//!         // rest into `{scalar}` (used for the exact `scalar_sqrt` pair).
//!     }
//! }
//! ```
//!
//! Per family this emits `{Name}WithPolicy` (required `_p` methods, leading
//! `P: Policy`), `{Name}` (default-policy forwarders), the blanket
//! `impl {Name} for M: {Name}WithPolicy`, and the dispatched blanket
//! `impl {Name}WithPolicy for V where V: Specialized{Name}<E>` whose
//! forwarder bodies carry the trace region markers. After all families it
//! emits the single monolithic scalar aggregate (`{scalar}WithPolicy` /
//! `{scalar}`) over every non-`#[vector_only]` method, implemented on bare
//! `f32`/`f64` through the width-1 `Vector<E>` and the `Unwrap` machinery.
//!
//! The region markers (`_enter`/`_region_arg`/... threading) are emitted
//! only when this crate's `trace` feature is on (`thermite/trace` and
//! `thermite-special/trace` forward to it), so a non-tracing build gets
//! forwarder bodies with no marker machinery at all.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{
    Attribute, FnArg, GenericParam, Generics, Ident, ItemTrait, Pat, ReturnType, Token, TraitItem, TraitItemFn, Type,
    TypeParamBound, Visibility,
};

use crate::region;

/// How a declared method maps onto the generated family.
enum FnKind {
    /// Ordinary: forwarded to the specialized trait, scalar form generated.
    Plain,
    /// No scalar form (the signature names types a bare scalar cannot spell).
    VectorOnly,
    /// Request-struct method: the forwarder body is `kind.eval::<P>()`, and
    /// the scalar form goes through `WrapTo`. The path is the kind trait,
    /// extracted from the `K` generic's bound with its arguments stripped.
    Kind(syn::Path),
    /// Scalar form exists but with a different spelling (`Self::Primal`
    /// tables collapse to plain `Self` at width 1).
    ScalarForm(ScalarSig),
}

/// The alternate scalar signature of a `#[scalar_form((args) -> ret)]` fn.
struct ScalarSig {
    inputs: Punctuated<FnArg, Token![,]>,
    output: ReturnType,
}

impl Parse for ScalarSig {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let args;
        syn::parenthesized!(args in input);
        let inputs = args.parse_terminated(FnArg::parse, Token![,])?;
        let output = input.parse()?;
        Ok(ScalarSig { inputs, output })
    }
}

struct MathFn {
    attrs: Vec<Attribute>,
    kind: FnKind,
    /// `#[compose]`: coarse tracing records this call as a composition
    /// (descends into the default body) rather than as a single node. No
    /// effect on emission. Forwarded through the surface exporter.
    compose: bool,
    sig: syn::Signature,
}

struct Family {
    attrs: Vec<Attribute>,
    vis: Visibility,
    /// The declared (default-policy) trait name, e.g. `CoreMath`.
    name: Ident,
    /// Bound on `E` in the blanket impl, from `#[element(...)]`.
    element: Option<syn::Path>,
    supertraits: Punctuated<TypeParamBound, Token![+]>,
    fns: Vec<MathFn>,
}

struct Input {
    thermite: syn::Path,
    /// Emit the surface exporter macro (`#![surface]`, or `#![surface(name)]`
    /// to override the default `__math_surface`, needed when one crate has
    /// several invocations). Opt-in, consumed by thermite-trace's
    /// `coarse_ops!` chain and, later, the VM's composite-replay generation.
    surface: Option<Ident>,
    /// Path prefix for the `Specialized*` traits (`#![specialized(self)]`).
    /// Defaults to `specialized`.
    specialized: syn::Path,
    /// Name of the scalar aggregate pair. `None` skips scalar emission
    /// entirely (thermite-complex: `scalar_` forms only make sense for bare
    /// `f32`/`f64`, not complex values).
    scalar: Option<Ident>,
    families: Vec<Family>,
    scalar_extras: Vec<TraitItemFn>,
}

fn take_attr(attrs: &mut Vec<Attribute>, name: &str) -> Option<Attribute> {
    let at = attrs.iter().position(|a| a.path().is_ident(name))?;
    Some(attrs.remove(at))
}

/// The kind trait from `fn f<K: KindTrait<Output = Self>>(..)`: the first
/// type-param bound, arguments stripped.
fn kind_trait(sig: &syn::Signature) -> syn::Result<syn::Path> {
    for param in &sig.generics.params {
        if let GenericParam::Type(t) = param
            && let Some(TypeParamBound::Trait(b)) = t.bounds.first()
        {
            let mut path = b.path.clone();
            if let Some(last) = path.segments.last_mut() {
                last.arguments = syn::PathArguments::None;
            }
            return Ok(path);
        }
    }
    Err(syn::Error::new(sig.span(), "#[kind] fn needs a `K: KindTrait<Output = Self>` generic"))
}

impl Parse for Input {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut thermite: Option<syn::Path> = None;
        let mut scalar: Option<Ident> = None;
        let mut surface: Option<Ident> = None;
        let mut specialized: syn::Path = syn::parse_quote! { specialized };

        for attr in input.call(Attribute::parse_inner)? {
            if attr.path().is_ident("thermite") {
                thermite = Some(attr.parse_args()?);
            } else if attr.path().is_ident("scalar") {
                scalar = Some(attr.parse_args()?);
            } else if attr.path().is_ident("surface") {
                surface = Some(match &attr.meta {
                    syn::Meta::Path(_) => format_ident!("__math_surface"),
                    _ => attr.parse_args()?,
                });
            } else if attr.path().is_ident("specialized") {
                specialized = attr.parse_args()?;
            } else {
                return Err(syn::Error::new(attr.span(), "unknown option; expected #![thermite = ...] or #![scalar = ...]"));
            }
        }
        // `#![thermite = crate]` parses as a name-value inner attribute only
        // with a literal. Accept the `#![thermite(crate)]` form via parse_args
        // above, and default sensibly.
        let thermite = thermite.ok_or_else(|| input.error("missing #![thermite(...)] option"))?;

        let mut families = Vec::new();
        let mut scalar_extras = Vec::new();

        while !input.is_empty() {
            if input.peek(Ident) && input.fork().parse::<Ident>()? == "scalar_extras" {
                input.parse::<Ident>()?;
                let body;
                syn::braced!(body in input);
                while !body.is_empty() {
                    scalar_extras.push(body.parse()?);
                }
                continue;
            }

            let tr: ItemTrait = input.parse()?;
            let mut attrs = tr.attrs;
            let element = match take_attr(&mut attrs, "element") {
                Some(a) => Some(a.parse_args()?),
                None => None,
            };

            let mut fns = Vec::new();
            for item in tr.items {
                let TraitItem::Fn(mut f) = item else {
                    return Err(syn::Error::new(item.span(), "math_traits! traits may only contain fns"));
                };
                let compose = take_attr(&mut f.attrs, "compose").is_some();
                let kind = if take_attr(&mut f.attrs, "vector_only").is_some() {
                    FnKind::VectorOnly
                } else if take_attr(&mut f.attrs, "kind").is_some() {
                    FnKind::Kind(kind_trait(&f.sig)?)
                } else if let Some(a) = take_attr(&mut f.attrs, "scalar_form") {
                    FnKind::ScalarForm(a.parse_args()?)
                } else {
                    FnKind::Plain
                };
                fns.push(MathFn { attrs: f.attrs, kind, compose, sig: f.sig });
            }

            families.push(Family {
                attrs,
                vis: tr.vis,
                name: tr.ident,
                element,
                supertraits: tr.supertraits,
                fns,
            });
        }

        Ok(Input {
            thermite,
            surface,
            specialized,
            scalar,
            families,
            scalar_extras,
        })
    }
}

/// Context shared by every emission site.
struct Cx {
    /// Path to the thermite crate (`crate` or `thermite`).
    path: syn::Path,
    /// True when `path` is literally `crate` (we are inside thermite).
    in_thermite: bool,
}

impl Cx {
    /// The dispatch attribute, spelled for the invoking crate.
    fn dispatch(&self) -> TokenStream2 {
        if self.in_thermite {
            quote! { #[thermite_macros::dispatch(Self, thermite = "crate")] }
        } else {
            let p = &self.path;
            quote! { #[#p::dispatch(Self)] }
        }
    }
}

/// `sig`'s generic parameter NAMES in declaration order (for turbofish).
fn generic_names(generics: &Generics) -> Vec<TokenStream2> {
    generics
        .params
        .iter()
        .map(|p| match p {
            GenericParam::Type(t) => {
                let i = &t.ident;
                quote! { #i }
            }
            GenericParam::Const(c) => {
                let i = &c.ident;
                quote! { #i }
            }
            GenericParam::Lifetime(l) => {
                let i = &l.lifetime;
                quote! { #i }
            }
        })
        .collect()
}

/// Argument names as passable expressions (`self` included).
fn arg_names(inputs: &Punctuated<FnArg, Token![,]>) -> Vec<TokenStream2> {
    inputs
        .iter()
        .map(|a| match a {
            FnArg::Receiver(_) => quote! { self },
            FnArg::Typed(t) => match &*t.pat {
                Pat::Ident(p) => {
                    let i = &p.ident;
                    quote! { #i }
                }
                other => quote! { #other },
            },
        })
        .collect()
}

/// `(name expr, type)` pairs for every argument.
fn arg_pairs(inputs: &Punctuated<FnArg, Token![,]>) -> Vec<(TokenStream2, Type)> {
    inputs
        .iter()
        .map(|a| match a {
            // syn 3: `Receiver` carries a `kind`, not a type. Math receivers
            // are always by-value `self`, whose type is `Self`.
            FnArg::Receiver(_) => (quote! { self }, syn::parse_quote! { Self }),
            FnArg::Typed(t) => {
                let name = match &*t.pat {
                    Pat::Ident(p) => {
                        let i = &p.ident;
                        quote! { #i }
                    }
                    other => quote! { #other },
                };
                (name, (*t.ty).clone())
            }
        })
        .collect()
}

/// The `_p` signature pieces: `<P: Policy, ...original generics>` and the
/// original inputs/output/where, name suffixed.
struct PSig {
    name: Ident,
    generics: TokenStream2,
    names: Vec<TokenStream2>,
    inputs: TokenStream2,
    output: TokenStream2,
    where_clause: TokenStream2,
}

fn p_sig(sig: &syn::Signature) -> PSig {
    let name = format_ident!("{}_p", sig.ident);
    let params = sig.generics.params.iter();
    let generics = quote! { <P: Policy #(, #params)*> };
    let names = generic_names(&sig.generics);
    let inputs = {
        let i = sig.inputs.iter();
        quote! { #(#i),* }
    };
    let output = {
        let o = &sig.output;
        quote! { #o }
    };
    let where_clause = match &sig.generics.where_clause {
        Some(w) => quote! { #w },
        None => quote! {},
    };
    PSig { name, generics, names, inputs, output, where_clause }
}

/// The region-marker prologue/epilogue for one forwarder, or plain
/// pass-through when the `trace` feature is off.
struct Markers {
    enter: TokenStream2,
    imm: TokenStream2,
    args: Vec<TokenStream2>,
    outs: TokenStream2,
    result: TokenStream2,
    exit: TokenStream2,
}

fn markers(cx: &Cx, sig: &syn::Signature, thread_args: bool) -> Markers {
    let names: Vec<TokenStream2> = arg_names(&sig.inputs);

    if !cfg!(feature = "trace") {
        return Markers {
            enter: quote! {},
            imm: quote! {},
            args: names,
            outs: quote! {},
            result: quote! { __result },
            exit: quote! {},
        };
    }

    let path = &cx.path;
    let fn_name = &sig.ident;
    let token = quote! { __region };
    let arg_m = quote! { #path::vector::GenericVector::_region_arg };
    let res_m = quote! { #path::vector::GenericVector::_region_result };

    let pairs = arg_pairs(&sig.inputs);

    let args: Vec<TokenStream2> = if thread_args {
        pairs
            .iter()
            .map(|(name, ty)| region::thread(name.clone(), ty, &token, &arg_m))
            .collect()
    } else {
        names
    };

    let imm = if thread_args {
        match region::imm_format(&sig.generics, &pairs) {
            Some((fmt, values)) => {
                quote! { <V as #path::vector::GenericVector>::_region_imm(__region, ::core::format_args!(#fmt, #(#values),*)); }
            }
            None => quote! {},
        }
    } else {
        quote! {}
    };

    let outs = if thread_args {
        let stmts = pairs.iter().filter_map(|(name, ty)| region::out_param(name.clone(), ty, &token, &res_m));
        quote! { #(#stmts)* }
    } else {
        quote! {}
    };

    let result_ty: Type = match &sig.output {
        ReturnType::Type(_, t) => (**t).clone(),
        ReturnType::Default => syn::parse_quote! { () },
    };
    let result = region::thread(quote! { __result }, &result_ty, &token, &res_m);

    Markers {
        enter: quote! { let __region = V::_enter(stringify!(#fn_name)); },
        imm,
        args,
        outs,
        result: quote! { #result },
        exit: quote! { V::_exit(__region); },
    }
}

/// One forwarder fn for the vector blanket impl, emitted as the two
/// `disable_dispatch` cfg arms.
fn forwarder(cx: &Cx, spec_path: &syn::Path, family: &Family, f: &MathFn) -> TokenStream2 {
    let attrs = &f.attrs;
    let base = family.name.to_string();
    let base = base.strip_suffix("Math").unwrap_or(&base);
    let spec = format_ident!("Specialized{base}Math");

    let is_kind = matches!(f.kind, FnKind::Kind(_));
    let m = markers(cx, &f.sig, !is_kind);
    let Markers { enter, imm, args, outs, result, exit } = m;

    let PSig { name, generics, names, inputs, output, where_clause } = p_sig(&f.sig);

    // The specialized trait's method keeps the bare name. Only the public
    // forwarder carries the `_p` suffix.
    let target = &f.sig.ident;
    let call = if is_kind {
        quote! { kind.eval::<P>() }
    } else {
        quote! { <V as #spec_path::#spec<E>>::#target::<P #(, #names)*>(#(#args),*) }
    };

    let body = quote! {{
        #enter
        #imm
        let __result = #call;
        #outs
        let __result = #result;
        #exit
        __result
    }};

    // A declaration already carrying #[skip_dispatch] must not get a second
    // one from the disable_dispatch arm.
    let skip = if f.attrs.iter().any(|a| a.path().is_ident("skip_dispatch")) {
        quote! {}
    } else {
        quote! { #[skip_dispatch] }
    };

    quote! {
        #[cfg(not(feature = "disable_dispatch"))]
        #(#attrs)* #[inline(always)]
        fn #name #generics(#inputs) #output #where_clause #body

        #[cfg(feature = "disable_dispatch")]
        #(#attrs)* #skip #[inline(always)]
        fn #name #generics(#inputs) #output #where_clause #body
    }
}

pub fn math_traits_inner(input: TokenStream) -> TokenStream {
    let Input {
        thermite,
        surface,
        specialized,
        scalar,
        families,
        scalar_extras,
    } = syn::parse_macro_input!(input as Input);
    let in_thermite = thermite.is_ident("crate");
    let cx = Cx { path: thermite.clone(), in_thermite };
    let path = &cx.path;
    let dispatch = cx.dispatch();

    let mut out = TokenStream2::new();

    for family in &families {
        let vis = &family.vis;
        let attrs = &family.attrs;
        let name = &family.name;
        let base = name.to_string();
        let base = base.strip_suffix("Math").unwrap_or(&base).to_string();
        let with_policy = format_ident!("{name}WithPolicy");
        let spec = format_ident!("Specialized{base}Math");
        let supers = &family.supertraits;
        let supers_opt = if supers.is_empty() { quote! {} } else { quote! { : #supers } };

        // --- the WithPolicy trait: required `_p` methods --------------------
        let p_decls = family.fns.iter().map(|f| {
            let fattrs = &f.attrs;
            let PSig { name, generics, inputs, output, where_clause, .. } = p_sig(&f.sig);
            quote! { #(#fattrs)* fn #name #generics(#inputs) #output #where_clause; }
        });

        let doc_wp = format!(
            "{base} math functions for floating-point vectors with customizable policies.\n\n\
             Each function carries a leading `P:\u{20}Policy` generic controlling the\n\
             precision/performance trade-off. For convenience, [`{name}`] provides the same\n\
             set of operations under [`DefaultPolicy`]. Every floating-point vector type\n\
             implementing [`specialized::{spec}`](specialized::{spec}) implements both\n\
             automatically.",
            spec = spec,
        );

        out.extend(quote! {
            #[doc = #doc_wp]
            #(#attrs)*
            #dispatch
            #vis trait #with_policy #supers_opt {
                #(#p_decls)*
            }
        });

        // --- the default-policy trait ---------------------------------------
        let defaults = family.fns.iter().map(|f| {
            let fattrs = &f.attrs;
            let sig = &f.sig;
            let PSig { name: p_name, names, .. } = p_sig(sig);
            let args = arg_names(&sig.inputs);
            quote! {
                #(#fattrs)* #[inline(always)] #sig
                { <Self as #with_policy>::#p_name::<DefaultPolicy #(, #names)*>(#(#args),*) }
            }
        });

        let doc_d = format!(
            "{base} math functions for floating-point vectors using the default policy.\n\n\
             The same operations as [`{with_policy}`], with every method's leading policy\n\
             fixed to [`DefaultPolicy`]. Implementors of [`{with_policy}`] automatically\n\
             implement this trait; each method here has a `_p`-suffixed counterpart there.",
        );

        out.extend(quote! {
            #[doc = #doc_d]
            #(#attrs)*
            #dispatch
            #vis trait #name: #with_policy {
                #(#defaults)*
            }

            impl<M> #name for M where M: #with_policy {}
        });

        // --- the dispatched vector blanket impl -----------------------------
        let fwd = family.fns.iter().map(|f| forwarder(&cx, &specialized, family, f));
        let e_bound = match &family.element {
            Some(b) => quote! { E: #b },
            None => quote! { E },
        };
        let supers_plus = if supers.is_empty() { quote! {} } else { quote! { + #supers } };

        out.extend(quote! {
            #dispatch
            impl<#e_bound, V: FloatVector<Element = E> #supers_plus> #with_policy for V
            where
                V: #specialized::#spec<E>,
            {
                #(#fwd)*
            }
        });
    }

    // --- the monolithic scalar aggregate ------------------------------------
    match &scalar {
        Some(scalar) => out.extend(scalar_aggregate(&cx, &specialized, scalar, &families, &scalar_extras, &dispatch)),
        None if !scalar_extras.is_empty() => {
            return syn::Error::new(scalar_extras[0].span(), "scalar_extras without #![scalar(...)]")
                .to_compile_error()
                .into();
        }
        None => {}
    }

    // --- the surface exporter (opt-in) ---------------------------------------
    if let Some(surface_name) = &surface {
    // A `#[doc(hidden)]` callback macro handing the whole declared surface
    // (signatures + trace-relevant tags, docs stripped) to a consumer.
    // thermite-trace's `coarse_ops!` reads the callable surface through it,
    // replacing source-scraping. Chain-composable: each exporter
    // appends its `@surface` block after any it received.
    let fam_blocks = families.iter().map(|family| {
        let name = &family.name;
        let fns = family.fns.iter().map(|f| {
            let sig = &f.sig;
            let compose = if f.compose { quote! { #[compose] } } else { quote! {} };
            let tag = match &f.kind {
                FnKind::Kind(_) => quote! { #[kind] },
                FnKind::ScalarForm(_) => quote! { #[scalar_form] },
                _ => quote! {},
            };
            quote! { #compose #tag #sig; }
        });
        quote! { family #name { #(#fns)* } }
    });
    let dollar = proc_macro2::Punct::new('$', proc_macro2::Spacing::Alone);
    out.extend(quote! {
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #surface_name {
            (#dollar(#dollar cb:ident)::+ ! { #dollar(#dollar inner:tt)* } #dollar(#dollar carry:tt)*) => {
                #dollar(#dollar cb)::+! {
                    #dollar(#dollar inner)*
                    #dollar(#dollar carry)*
                    @surface { #(#fam_blocks)* }
                }
            };
        }
    });
    }

    let _ = path;
    out.into()
}

/// Everything scalar: `{scalar}WithPolicy`, `{scalar}`, the blanket, and the
/// implementation on bare floats through the width-1 vector.
fn scalar_aggregate(
    cx: &Cx,
    spec_path: &syn::Path,
    scalar: &Ident,
    families: &[Family],
    extras: &[TraitItemFn],
    dispatch: &TokenStream2,
) -> TokenStream2 {
    let path = &cx.path;
    let scalar_wp = format_ident!("{scalar}WithPolicy");

    let mut wp_decls = Vec::new();
    let mut d_defaults = Vec::new();
    let mut impls = Vec::new();
    let mut any_scalar_form = false;

    for family in families {
        let base = family.name.to_string();
        let base = base.strip_suffix("Math").unwrap_or(&base).to_string();
        let spec = format_ident!("Specialized{base}Math");

        for f in &family.fns {
            let fattrs = &f.attrs;
            match &f.kind {
                FnKind::VectorOnly => {}
                FnKind::Kind(ktrait) => {
                    let sname = format_ident!("scalar_{}", f.sig.ident);
                    let sname_p = format_ident!("scalar_{}_p", f.sig.ident);
                    wp_decls.push(quote! {
                        #(#fattrs)* fn #sname_p<P: Policy, K: WrapTo>(kind: K) -> Self
                        where K::Wrapped: #ktrait, <K::Wrapped as #ktrait>::Output: Unwrap<Unwrapped = Self>;
                    });
                    d_defaults.push(quote! {
                        #(#fattrs)* #[inline(always)] fn #sname<K: WrapTo>(kind: K) -> Self
                        where K::Wrapped: #ktrait, <K::Wrapped as #ktrait>::Output: Unwrap<Unwrapped = Self>
                        { #scalar_wp::#sname_p::<DefaultPolicy, K>(kind) }
                    });
                    impls.push(quote! {
                        #(#fattrs)* #[skip_dispatch] #[inline(always)] fn #sname_p<P: Policy, K: WrapTo>(kind: K) -> Self
                        where K::Wrapped: #ktrait, <K::Wrapped as #ktrait>::Output: Unwrap<Unwrapped = Self>
                        { Unwrap::unwrap(<K::Wrapped as Unwrap>::wrap(kind).eval::<P>()) }
                    });
                }
                FnKind::Plain | FnKind::ScalarForm(_) => {
                    // The scalar spelling of the signature.
                    let (inputs, output) = match &f.kind {
                        FnKind::ScalarForm(s) => {
                            any_scalar_form = true;
                            (s.inputs.clone(), s.output.clone())
                        }
                        _ => (f.sig.inputs.clone(), f.sig.output.clone()),
                    };
                    let vname = &f.sig.ident;
                    let sname = format_ident!("scalar_{vname}");
                    let sname_p = format_ident!("scalar_{vname}_p");
                    let params = f.sig.generics.params.iter().collect::<Vec<_>>();
                    let generics_p = quote! { <P: Policy #(, #params)*> };
                    let generics_d = if params.is_empty() { quote! {} } else { quote! { <#(#params),*> } };
                    let names = generic_names(&f.sig.generics);
                    let wc = match &f.sig.generics.where_clause {
                        Some(w) => quote! { #w },
                        None => quote! {},
                    };
                    let inputs_ts = {
                        let i = inputs.iter();
                        quote! { #(#i),* }
                    };
                    let out_ts = quote! { #output };
                    let args = arg_names(&inputs);
                    // `self` renamed `this` inside the wrap/unwrap dance.
                    let bind: Vec<TokenStream2> = inputs
                        .iter()
                        .map(|a| match a {
                            FnArg::Receiver(_) => quote! { __this },
                            FnArg::Typed(t) => match &*t.pat {
                                Pat::Ident(p) => {
                                    let i = &p.ident;
                                    quote! { #i }
                                }
                                other => quote! { #other },
                            },
                        })
                        .collect();

                    wp_decls.push(quote! {
                        #(#fattrs)* fn #sname_p #generics_p(#inputs_ts) #out_ts #wc;
                    });
                    d_defaults.push(quote! {
                        #(#fattrs)* #[inline(always)] fn #sname #generics_d(#inputs_ts) #out_ts #wc
                        { <Self as #scalar_wp>::#sname_p::<DefaultPolicy #(, #names)*>(#(#args),*) }
                    });
                    impls.push(quote! {
                        #(#fattrs)* #[skip_dispatch] #[inline(always)] fn #sname_p #generics_p(#inputs_ts) #out_ts #wc
                        {
                            let (#(#bind,)*) = Unwrap::wrap((#(#args,)*));
                            let res = <#path::Vector<E> as #spec_path::#spec<E>>::#vname::<P #(, #names)*>(#(#bind),*);
                            Unwrap::unwrap(res)
                        }
                    });
                }
            }
        }
    }

    let (extra_wp, extra_d): (Vec<_>, Vec<_>) = extras
        .iter()
        .partition(|f| f.sig.ident.to_string().ends_with("_p"));

    let family_links_wp: Vec<String> = families.iter().map(|f| format!("- [`{}WithPolicy`]", f.name)).collect();
    let family_links_d: Vec<String> = families.iter().map(|f| format!("- [`{}`]", f.name)).collect();

    let doc_wp = format!(
        "Aggregate of all scalar math traits with customizable policies.\n\n\
         This trait collects every method from the following trait families into a single\n\
         trait implemented directly on `f32` and `f64`:\n\n{}\n\n\
         All methods are prefixed with `scalar_` to avoid conflicts with the inherent methods\n\
         already defined on `f32`/`f64` (e.g., `f32::sin`, `f32::exp`). The policy-aware\n\
         versions additionally carry a `_p` suffix, following the same convention as the\n\
         vector math traits.\n\n\
         # Limitations\n\n\
         This trait is **only** implemented for bare scalar types. Code that is generic over\n\
         a `FloatVector` bound will not accept a bare `f32` or `f64` - the scalar must be\n\
         wrapped in `Vector` first (e.g., `Vector::<f32>(x)`) to satisfy that bound.\n\
         [`{scalar}`] provides the default-policy spelling.",
        family_links_wp.join("\n"),
    );
    let doc_d = format!(
        "Aggregate of all scalar math traits using the default policy.\n\n\
         This trait collects every method from the following trait families into a single\n\
         trait implemented directly on `f32` and `f64`, using `DefaultPolicy` for all\n\
         operations:\n\n{}\n\n\
         All methods are prefixed with `scalar_`. See [`{scalar_wp}`] for the policy-aware\n\
         variant, which additionally carries a `_p` suffix on each method.\n\n\
         # Limitations\n\n\
         Implemented **only** for bare scalar types; see [`{scalar_wp}`]. All types\n\
         implementing [`{scalar_wp}`] automatically implement this trait.",
        family_links_d.join("\n"),
    );

    let on_unimpl_wp = format!(
        "`{{Self}}` is not a bare floating-point scalar; `{scalar_wp}` is implemented only for `f32` and `f64`. \
         For SIMD vectors, bound on `FloatVector` plus the vector math traits instead."
    );
    let on_unimpl_d = format!(
        "`{{Self}}` is not a bare floating-point scalar; `{scalar}` is implemented only for `f32` and `f64`. \
         For SIMD vectors, bound on `FloatVector` plus the vector math traits instead."
    );

    let spec_bounds = families.iter().map(|f| {
        let base = f.name.to_string();
        let base = base.strip_suffix("Math").unwrap_or(&base).to_string();
        let spec = format_ident!("Specialized{base}Math");
        quote! { + #spec_path::#spec<E> }
    });

    // The scalar_form fns exist because a `Self::Primal` table collapses to
    // `Self` at width 1. The pin makes that normalization hold.
    let primal_pin = if any_scalar_form {
        quote! { + PrimalProjection<Primal = #path::Vector<E>> }
    } else {
        quote! {}
    };

    quote! {
        #[doc = #doc_wp]
        #dispatch
        #[diagnostic::on_unimplemented(message = "`{Self}` is not a bare floating-point scalar", note = #on_unimpl_wp)]
        pub trait #scalar_wp: ElementExt<Element = Self> + FloatElementWithBits {
            #(#wp_decls)*
            #(#extra_wp)*
        }

        #[doc = #doc_d]
        #dispatch
        #[diagnostic::on_unimplemented(message = "`{Self}` is not a bare floating-point scalar", note = #on_unimpl_d)]
        pub trait #scalar: #scalar_wp {
            #(#d_defaults)*
            #(#extra_d)*
        }

        impl<M> #scalar for M where M: #scalar_wp {}

        #dispatch
        impl<E: ElementExt<Element = Self> + FloatElementWithBits> #scalar_wp for E
        where
            #path::Vector<E>: Unwrap<Unwrapped = E> +
                FloatVectorWithBits<Element = E,
                    Signed: Unwrap<Unwrapped = <E as Element>::Signed>,
                    Unsigned: Unwrap<Unwrapped = <E as Element>::Unsigned>,
                    SignedBits: Unwrap<Unwrapped = <E as FloatElementWithBits>::SignedBits>,
                    Bits: Unwrap<Unwrapped = <E as FloatElementWithBits>::Bits>
                >
                #primal_pin
                #(#spec_bounds)*,
            E: #path::register::FloatRegister<Storage = E>,
        {
            #(#impls)*
        }
    }
}
