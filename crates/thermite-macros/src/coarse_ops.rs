//! `coarse_ops!`: generates thermite-trace's `Op` enum and coarse impls
//! from the math surface exported by `math_traits!`.
//!
//! Invoked in thermite-trace as the tail of the exporter chain:
//!
//! ```ignore
//! thermite::__math_surface! {
//!     thermite_special::__math_surface! {
//!         thermite_macros::coarse_ops! {
//!             vector { add sub mul ... }
//!             blocked { FloatMath }
//!             statics { inverse_smoothstep ... }
//!             extra_items { SpecialMath { type ExpIntDetails = Self; } }
//!         }
//!     }
//! }
//! ```
//!
//! Each `__math_surface!` link appends an `@surface { family XMath { fns } }`
//! block, so this macro receives the complete callable surface (signatures
//! plus the trace-relevant tags), with no source scraping and nothing to
//! regenerate: a function added to core appears here at the next compile.
//!
//! Per public math fn (minus `#[compose]`/`#[kind]`/`#[scalar_form]`, which
//! coarse mode records as compositions, and minus `blocked` families, whose
//! trait bounds `Symbolic` cannot satisfy) this emits one `Op` variant and
//! one recording method in the family's `Specialized*Math` impl for
//! `Trace<Symbolic<T>, W>`. The `vector` block contributes name-only
//! variants for the hand-written vector-primitive impls (compiler-enforced,
//! since every entry exists because a call site names `Op::Variant`).
//!
//! `statics` lists fns whose public form takes `self` but whose
//! `Specialized*` declaration is an associated fn with a named first
//! argument. The impl must match the specialized spelling. Wrong entries
//! fail to compile in either direction (E0185/E0186).

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{FnArg, Ident, Pat, ReturnType, Token, TraitItemFn, Type};

use crate::region;

fn camel(name: &str) -> Ident {
    let mut out = String::new();
    for part in name.split('_') {
        let mut ch = part.chars();
        if let Some(c) = ch.next() {
            out.push(c.to_ascii_uppercase());
            out.extend(ch);
        }
    }
    format_ident!("{out}")
}

struct SurfaceFn {
    skip: bool,
    sig: syn::Signature,
}

struct SurfaceFamily {
    name: Ident,
    fns: Vec<SurfaceFn>,
}

struct Input {
    vector: Vec<Ident>,
    blocked: Vec<Ident>,
    statics: Vec<Ident>,
    extra_items: Vec<(Ident, TokenStream2)>,
    /// Bounds for the generated `ReplayMath` blanket impl: the WithPolicy
    /// traits whose methods the VM arms call.
    vm_bounds: Option<TokenStream2>,
    families: Vec<SurfaceFamily>,
}

impl Parse for Input {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut out = Input {
            vector: Vec::new(),
            blocked: Vec::new(),
            statics: Vec::new(),
            extra_items: Vec::new(),
            vm_bounds: None,
            families: Vec::new(),
        };

        while !input.is_empty() {
            if input.peek(Token![@]) {
                input.parse::<Token![@]>()?;
                let kw: Ident = input.parse()?;
                if kw != "surface" {
                    return Err(syn::Error::new(kw.span(), "expected @surface"));
                }
                let block;
                syn::braced!(block in input);
                while !block.is_empty() {
                    let fam_kw: Ident = block.parse()?;
                    if fam_kw != "family" {
                        return Err(syn::Error::new(fam_kw.span(), "expected `family`"));
                    }
                    let name: Ident = block.parse()?;
                    let body;
                    syn::braced!(body in block);
                    let mut fns = Vec::new();
                    while !body.is_empty() {
                        let mut f: TraitItemFn = body.parse()?;
                        let mut skip = false;
                        f.attrs.retain(|a| {
                            let tag = a.path().is_ident("compose")
                                || a.path().is_ident("kind")
                                || a.path().is_ident("scalar_form");
                            if tag {
                                skip = true;
                            }
                            !tag
                        });
                        fns.push(SurfaceFn { skip, sig: f.sig });
                    }
                    out.families.push(SurfaceFamily { name, fns });
                }
                continue;
            }

            let kw: Ident = input.parse()?;
            let block;
            syn::braced!(block in input);
            match kw.to_string().as_str() {
                "vector" | "blocked" | "statics" => {
                    let list = match kw.to_string().as_str() {
                        "vector" => &mut out.vector,
                        "blocked" => &mut out.blocked,
                        _ => &mut out.statics,
                    };
                    while !block.is_empty() {
                        list.push(block.parse()?);
                    }
                }
                "vm_bounds" => {
                    out.vm_bounds = Some(block.parse()?);
                }
                "extra_items" => {
                    while !block.is_empty() {
                        let fam: Ident = block.parse()?;
                        let items;
                        syn::braced!(items in block);
                        out.extra_items.push((fam, items.parse()?));
                    }
                }
                other => {
                    return Err(syn::Error::new(
                        kw.span(),
                        format!("unknown block `{other}`; expected vector/blocked/statics/extra_items/vm_bounds/@surface"),
                    ));
                }
            }
        }

        Ok(out)
    }
}

/// The lowering class of one argument type: the port of the generator's
/// `classify`.
enum Arg {
    /// A trace value (`Self`, its integer siblings): a node operand.
    Plain,
    /// A host scalar (`u32`/`i32`): rendered into the immediate.
    Imm,
    /// `&[Self]` / `[Self; N]` / `&GenericArray<..>`: n-ary operands.
    Seq,
    /// `Option<(Self, Self)>`: two arity arms.
    Option,
}

fn classify(ty: &Type, fn_name: &Ident) -> syn::Result<Arg> {
    if region::is_vector(ty) {
        return Ok(Arg::Plain);
    }
    match ty {
        Type::Path(p) if p.path.is_ident("u32") || p.path.is_ident("i32") => Ok(Arg::Imm),
        Type::Array(a) if region::is_vector(&a.elem) => Ok(Arg::Seq),
        Type::Reference(r) if r.mutability.is_none() => match &*r.elem {
            Type::Slice(s) if region::is_vector(&s.elem) => Ok(Arg::Seq),
            Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "GenericArray") => Ok(Arg::Seq),
            _ => Err(syn::Error::new(
                fn_name.span(),
                format!("coarse_ops: no lowering rule for an argument type of `{fn_name}`"),
            )),
        },
        _ => {
            if let Type::Path(p) = ty
                && p.path.segments.last().is_some_and(|s| s.ident == "Option")
            {
                return Ok(Arg::Option);
            }
            Err(syn::Error::new(
                fn_name.span(),
                format!("coarse_ops: no lowering rule for an argument type of `{fn_name}`"),
            ))
        }
    }
}

/// `format_args!("{N}, {n}")` for the node's immediate, if any: const
/// generics first, then host-scalar arguments, declaration order.
fn imm_expr(sig: &syn::Signature, imm_args: &[Ident]) -> Option<TokenStream2> {
    let mut parts = Vec::new();
    for p in &sig.generics.params {
        if let syn::GenericParam::Const(c) = p {
            parts.push(c.ident.to_string());
        }
    }
    parts.extend(imm_args.iter().map(|i| i.to_string()));
    if parts.is_empty() {
        return None;
    }
    let fmt = parts.iter().map(|p| format!("{{{p}}}")).collect::<Vec<_>>().join(", ");
    Some(quote! { format_args!(#fmt) })
}

/// `crate::value::opK[_imm](Op::V, [imm,] operands...)`.
fn fixed_call(variant: &Ident, imm: &Option<TokenStream2>, operands: &[TokenStream2]) -> TokenStream2 {
    let helper = match operands.len() {
        1 => Some("op1"),
        2 => Some("op2"),
        3 => Some("op3"),
        4 => Some("op4"),
        _ => None,
    };
    let imm_arg = imm.as_ref().map(|i| quote! { #i, });
    let suffix = if imm.is_some() { "_imm" } else { "" };
    match helper {
        Some(h) => {
            let h = format_ident!("{h}{suffix}");
            quote! { crate::value::#h(Op::#variant, #imm_arg #(#operands),*) }
        }
        None => {
            let h = format_ident!("opn{suffix}");
            quote! { crate::value::#h(Op::#variant, #imm_arg &[#(#operands),*]) }
        }
    }
}

fn emit_method(f: &SurfaceFn, statics: &[Ident]) -> syn::Result<TokenStream2> {
    let sig = &f.sig;
    let name = &sig.ident;
    let variant = camel(&name.to_string());

    // Rebuild the argument list, renaming the receiver `x: Self` when the
    // Specialized* declaration is an associated fn (the `statics` list).
    let force_static = statics.iter().any(|s| s == name);
    let mut inputs = Vec::new();
    let mut names: Vec<TokenStream2> = Vec::new();
    let mut kinds = Vec::new();
    for a in &sig.inputs {
        match a {
            FnArg::Receiver(_) if force_static => {
                inputs.push(quote! { x: Self });
                names.push(quote! { x });
                kinds.push(Arg::Plain);
            }
            FnArg::Receiver(_) => {
                inputs.push(quote! { self });
                names.push(quote! { self });
                kinds.push(Arg::Plain);
            }
            FnArg::Typed(t) => {
                let ty = &t.ty;
                let n = match &*t.pat {
                    Pat::Ident(p) => p.ident.clone(),
                    other => return Err(syn::Error::new(name.span(), format!("unsupported pattern {other:?}"))),
                };
                kinds.push(classify(ty, name)?);
                inputs.push(quote! { #n: #ty });
                names.push(quote! { #n });
            }
        }
    }

    let imm_names: Vec<Ident> = names
        .iter()
        .zip(&kinds)
        .filter(|(_, k)| matches!(k, Arg::Imm))
        .map(|(n, _)| syn::parse2(n.clone()).expect("imm arg is an ident"))
        .collect();
    let imm = imm_expr(sig, &imm_names);

    let params = sig.generics.params.iter();
    let generics = quote! { <P: Policy #(, #params)*> };
    let output = &sig.output;

    // Fused: tuple return, one node with several results.
    let body = if let ReturnType::Type(_, t) = output
        && let Type::Tuple(tup) = &**t
    {
        let n = tup.elems.len();
        let outs: Vec<Ident> = (0..n).map(|i| format_ident!("__r{i}")).collect();
        let syms = names.iter().zip(&kinds).filter(|(_, k)| matches!(k, Arg::Plain)).map(|(v, _)| quote! { #v.sym() });
        let rec = if let Some(imm) = &imm {
            quote! { crate::value::record_imm::<_, #n>(Op::#variant, #imm, &[#(#syms),*]) }
        } else {
            quote! { crate::value::record::<_, #n>(Op::#variant, &[#(#syms),*]) }
        };
        quote! {
            let [#(#outs),*] = #rec;
            (#(#outs),*)
        }
    } else if kinds.iter().any(|k| matches!(k, Arg::Seq)) {
        // Sequence arguments: collect all operand syms into one pool.
        let pushes = names.iter().zip(&kinds).filter_map(|(v, k)| match k {
            Arg::Plain => Some(quote! { a.push(#v.sym()); }),
            Arg::Seq => Some(quote! { a.extend(#v.iter().map(|__v| __v.sym())); }),
            _ => None,
        });
        let call = if let Some(imm) = &imm {
            quote! { crate::value::op_syms_imm(Op::#variant, #imm, &a) }
        } else {
            quote! { crate::value::op_syms(Op::#variant, &a) }
        };
        quote! {
            let mut a: Vec<Sym> = Vec::new();
            #(#pushes)*
            #call
        }
    } else if let Some(at) = kinds.iter().position(|k| matches!(k, Arg::Option)) {
        // Option<(Self, Self)>: two arms, arity distinguishes.
        let opt = &names[at];
        let mut with_edges = Vec::new();
        let mut without = Vec::new();
        for (v, k) in names.iter().zip(&kinds) {
            match k {
                Arg::Option => {
                    with_edges.push(quote! { __e0 });
                    with_edges.push(quote! { __e1 });
                }
                Arg::Plain => {
                    with_edges.push(v.clone());
                    without.push(v.clone());
                }
                _ => {}
            }
        }
        let some = fixed_call(&variant, &imm, &with_edges);
        let none = fixed_call(&variant, &imm, &without);
        quote! {
            match #opt {
                Some((__e0, __e1)) => #some,
                None => #none,
            }
        }
    } else {
        let plain: Vec<TokenStream2> =
            names.iter().zip(&kinds).filter(|(_, k)| matches!(k, Arg::Plain)).map(|(v, _)| v.clone()).collect();
        fixed_call(&variant, &imm, &plain)
    };

    Ok(quote! {
        #[inline]
        #[track_caller]
        fn #name #generics(#(#inputs),*) #output {
            #body
        }
    })
}

pub fn coarse_ops_inner(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as Input);

    let mut variants = Vec::new(); // (variant ident, snake name)
    let mut impls = TokenStream2::new();
    let mut vm_arms = Vec::new();

    for family in &input.families {
        if input.blocked.iter().any(|b| b == &family.name) {
            continue;
        }
        let live: Vec<&SurfaceFn> = family.fns.iter().filter(|f| !f.skip).collect();
        for f in &live {
            variants.push((camel(&f.sig.ident.to_string()), f.sig.ident.to_string()));
            vm_arms.push(replay_arm(f).unwrap_or_else(|| {
                let variant = camel(&f.sig.ident.to_string());
                quote! {
                    Op::#variant => {
                        return Err(crate::vm::VmError::Unsupported {
                            op,
                            detail: "const-generic or sequence immediate (VM-3)",
                        });
                    }
                }
            }));
        }
        if live.is_empty() {
            continue;
        }

        let base = family.name.to_string();
        let base = base.strip_suffix("Math").unwrap_or(&base).to_string();
        let spec = format_ident!("Specialized{base}Math");
        let extra = input
            .extra_items
            .iter()
            .filter(|(fam, _)| fam == &family.name)
            .map(|(_, items)| items.clone());
        let methods = live
            .iter()
            .map(|f| emit_method(f, &input.statics))
            .collect::<syn::Result<Vec<_>>>();
        let methods = match methods {
            Ok(m) => m,
            Err(e) => return e.to_compile_error().into(),
        };

        impls.extend(quote! {
            impl<T, W: TraceWidth> #spec<Symbolic<T>> for Trace<Symbolic<T>, W>
            where
                T: FloatElement + TraceElementName,
                T::Signed: TraceElementName,
                T::Unsigned: TraceElementName,
            {
                #(#extra)*
                #(#methods)*
            }
        });
    }

    for v in &input.vector {
        let name = v.to_string();
        let variant = camel(&name);
        if variants.iter().any(|(c, _)| *c == variant) {
            return syn::Error::new(v.span(), format!("vector op `{name}` collides with a math op"))
                .to_compile_error()
                .into();
        }
        variants.push((variant, name));
    }

    // --- the generated VM arms (VM-2) ------------------------------------
    let replay = match &input.vm_bounds {
        Some(bounds) => quote! {
            /// Generated math replay arms: one per coarse `Op`, each calling
            /// the real `_p` method under the caller's policy. The VM
            /// emulates nothing. `Ok(false)` = not a math op (the executor
            /// falls through to its vector arms first, so this is the
            /// terminal stop).
            pub trait ReplayMath<P: thermite::math::policy::Policy>:
                thermite::prelude::FloatVectorWithBits
            {
                fn replay_math(
                    op: Op,
                    imm: Option<&str>,
                    dst: &[crate::vm::Reg],
                    args: &[crate::vm::Reg],
                    regs: &mut crate::vm::Regs<Self>,
                ) -> Result<bool, crate::vm::VmError>;
            }

            impl<V, P: thermite::math::policy::Policy> ReplayMath<P> for V
            where
                V: thermite::prelude::FloatVectorWithBits + #bounds,
                // `powiv`/`hermitev` are declared over `Self::Signed` /
                // `Self::Unsigned`. The VM's slots hold the bits family. On
                // every real float vector these are the same types. State it
                // so the projections unify.
                V: thermite::prelude::GenericVector<
                    Signed = <V as thermite::prelude::FloatVectorWithBits>::SignedBits,
                    Unsigned = <V as thermite::prelude::FloatVectorWithBits>::Bits,
                >,
            {
                #[inline(always)]
                #[allow(unused_variables)]
                fn replay_math(
                    op: Op,
                    imm: Option<&str>,
                    dst: &[crate::vm::Reg],
                    args: &[crate::vm::Reg],
                    regs: &mut crate::vm::Regs<Self>,
                ) -> Result<bool, crate::vm::VmError> {
                    match op {
                        #(#vm_arms)*
                        _ => return Ok(false),
                    }
                    Ok(true)
                }
            }
        },
        None => quote! {},
    };

    let decls = variants.iter().map(|(v, s)| {
        let doc = format!("`{s}`");
        quote! { #[doc = #doc] #v, }
    });
    let names = variants.iter().map(|(v, s)| quote! { Op::#v => #s, });
    let all = variants.iter().map(|(v, _)| quote! { Op::#v, });
    let count = variants.len();

    quote! {
        /// Every operation a trace can record: one variant per public math
        /// function (coarse nodes) plus the vector-primitive surface the
        /// hand-written impls bind. Generated by `coarse_ops!` from the
        /// surface `math_traits!` exports (nothing to regenerate), and the
        /// whole surface is always present regardless of how the trace was
        /// produced.
        ///
        /// Discriminants are NOT stable across compiler-visible surface
        /// changes. Serialization must key on [`name`](Op::name).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Op {
            #(#decls)*
        }

        impl Op {
            /// The operation's name as recorded in listings.
            pub const fn name(self) -> &'static str {
                match self {
                    #(#names)*
                }
            }

            /// Every variant, for vocabulary walks.
            pub const ALL: [Op; #count] = [#(#all)*];
        }

        #impls

        #replay
    }
    .into()
}

/// The generated VM arm for one math fn: fetch typed operands, call the
/// real `_p` method under the caller's policy, store the result. `None`
/// when the signature needs machinery the VM does not have yet (VM-3
/// immediate lowering, sequence/option arities): those get an explicit
/// `Unsupported` arm naming the reason.
fn replay_arm(f: &SurfaceFn) -> Option<TokenStream2> {
    let sig = &f.sig;
    let name = &sig.ident;
    let p_name = format_ident!("{name}_p");
    let variant = camel(&name.to_string());

    // Any generic parameter means a const-generic immediate or a type
    // parameter: VM-3 territory.
    if !sig.generics.params.is_empty() {
        return None;
    }

    #[derive(PartialEq)]
    enum A {
        F,
        U,
        I,
        Imm,
    }
    let mut shape = Vec::new();
    for arg in &sig.inputs {
        match arg {
            FnArg::Receiver(_) => shape.push(A::F),
            FnArg::Typed(t) => match &*t.ty {
                Type::Path(p) if p.path.is_ident("u32") || p.path.is_ident("i32") => shape.push(A::Imm),
                ty if region::is_vector(ty) => {
                    let s = quote!(#ty).to_string();
                    shape.push(if s.contains("Signed") {
                        A::I
                    } else if s.contains("Unsigned") || s.contains("Bits") {
                        A::U
                    } else {
                        A::F
                    });
                }
                _ => return None,
            },
        }
    }

    // Result: Self, or a tuple of Self (fused).
    let n_out = match &sig.output {
        ReturnType::Type(_, t) => match &**t {
            Type::Tuple(tup) => {
                if !tup.elems.iter().all(region::is_vector) {
                    return None;
                }
                tup.elems.len()
            }
            ty if region::is_vector(ty) => 1,
            _ => return None,
        },
        ReturnType::Default => return None,
    };

    let n_imm = shape.iter().filter(|k| **k == A::Imm).count();
    let mut vec_at = 0usize;
    let fetches = shape.iter().enumerate().map(|(k, a)| {
        let var = format_ident!("__a{k}");
        match a {
            A::Imm => quote! {
                let #var = __imm_parts
                    .next()
                    .and_then(|p| p.trim().parse().ok())
                    .ok_or_else(|| crate::vm::VmError::BadImm {
                        op,
                        imm: imm.unwrap_or("").to_string(),
                    })?;
            },
            _ => {
                let fetch = match a {
                    A::F => quote! { regs.f(args[#vec_at], op)? },
                    A::U => quote! { regs.u(args[#vec_at], op)? },
                    A::I => quote! { regs.i(args[#vec_at], op)? },
                    A::Imm => unreachable!(),
                };
                vec_at += 1;
                quote! { let #var = #fetch; }
            }
        }
    });
    let fetches: Vec<_> = fetches.collect();

    let imm_setup = if n_imm > 0 {
        quote! {
            let mut __imm_parts = imm
                .ok_or(crate::vm::VmError::Unsupported { op, detail: "missing immediate" })?
                .split(',');
        }
    } else {
        quote! {}
    };

    // The call: receiver method or associated fn, args in declared order.
    let arg_vars: Vec<Ident> = (0..shape.len()).map(|k| format_ident!("__a{k}")).collect();
    let call = if matches!(sig.inputs.first(), Some(FnArg::Receiver(_))) {
        let rest = &arg_vars[1..];
        let recv = &arg_vars[0];
        quote! { #recv.#p_name::<P>(#(#rest),*) }
    } else {
        quote! { Self::#p_name::<P>(#(#arg_vars),*) }
    };

    let store = if n_out == 1 {
        quote! { regs.set(dst[0], crate::vm::Slot::F(__r)); }
    } else {
        let outs: Vec<Ident> = (0..n_out).map(|i| format_ident!("__r{i}")).collect();
        let sets = outs.iter().enumerate().map(|(i, o)| quote! { regs.set(dst[#i], crate::vm::Slot::F(#o)); });
        quote! {
            let (#(#outs),*) = __r;
            #(#sets)*
        }
    };

    Some(quote! {
        Op::#variant => {
            #imm_setup
            #(#fetches)*
            let __r = #call;
            #store
        }
    })
}
