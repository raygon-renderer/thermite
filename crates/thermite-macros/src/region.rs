//! Helper macros for the `decl_math!` region markers.
//!
//! The `decl_math!` forwarders bracket every public math call between
//! `_enter`/`_exit` hints so a symbolic trace can collapse the call to a
//! single node. For that node to be _reconstructible_ (replayed by calling
//! the real function), the markers must also carry the call's operands,
//! results and immediates. Which treatment each piece gets depends on its
//! type. `macro_rules!` cannot inspect a captured `:ty`, so these three
//! function-like macros do the classification, and `decl_math!` just
//! invokes them per argument / per return.
//!
//! Classification, by syntactic type:
//!
//! - `Self`, `Self::Signed`, `Self::Unsigned`, `Self::Bits`,
//!   `Self::SignedBits`: a traced vector threaded through `_region_arg` /
//!   `_region_result` (identity for real vectors, records for a trace).
//! - `[Self; N]`, `&[Self]`: element-wise threaded.
//! - `Option<(Self, Self)>`: threaded when `Some`.
//! - `&mut [Self; N]` is an out-parameter: its slots are threaded as
//!   _results_ after the call (`region_out!`).
//! - Host scalars (`i32`, `u32`, ...) and element slices: formatted into
//!   the region's immediate (`region_imm!`), alongside const generics.
//! - Everything else (`Self::Primal` coefficient tables, `ShTable`
//!   references, policies): passed through untouched and absent from the
//!   marker. `Self::Primal` is deliberately unthreaded: the math traits do
//!   not bound it as a vector, so the marker methods are not nameable on it.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Type;

/// Is this type a threaded vector: `Self` or one of its integer siblings?
pub(crate) fn is_vector(ty: &Type) -> bool {
    let Type::Path(p) = ty else { return false };
    if p.qself.is_some() {
        return false;
    }
    let segs: Vec<String> = p.path.segments.iter().map(|s| s.ident.to_string()).collect();
    match segs.as_slice() {
        [s] => s == "Self",
        [s, assoc] => s == "Self" && matches!(assoc.as_str(), "Signed" | "Unsigned" | "Bits" | "SignedBits" | "Real"),
        _ => false,
    }
}

/// Is this `Option<(Self, Self)>` (the `edges` arguments)?
fn is_option_of_vector_pair(ty: &Type) -> Option<&Type> {
    let Type::Path(p) = ty else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    match args.args.first()? {
        syn::GenericArgument::Type(inner) => Some(inner),
        _ => None,
    }
}

/// Thread a tuple type member-wise. Non-vector members pass through.
fn thread_tuple(value: TokenStream2, tup: &syn::TypeTuple, token: &TokenStream2, m: &TokenStream2) -> TokenStream2 {
    let members = tup.elems.iter().enumerate().map(|(i, elem)| {
        let idx = syn::Index::from(i);
        if is_vector(elem) {
            quote! { #m(#value.#idx, #token) }
        } else {
            quote! { #value.#idx }
        }
    });
    quote! { (#(#members),*) }
}

/// One expression threading `value: ty` through the marker `m` (`_region_arg`
/// or `_region_result`), or passing it through untouched.
pub(crate) fn thread(value: TokenStream2, ty: &Type, token: &TokenStream2, m: &TokenStream2) -> TokenStream2 {
    if is_vector(ty) {
        return quote! { #m(#value, #token) };
    }

    match ty {
        // [Self; N] by value
        Type::Array(arr) if is_vector(&arr.elem) => quote! {{
            let mut __arr = #value;
            for __v in __arr.iter_mut() {
                *__v = #m(*__v, #token);
            }
            __arr
        }},
        // &[Self]: record only. The reference passes through unchanged
        Type::Reference(r) if r.mutability.is_none() => match &*r.elem {
            Type::Slice(s) if is_vector(&s.elem) => quote! {{
                for __v in #value.iter() {
                    let _ = #m(*__v, #token);
                }
                #value
            }},
            _ => value,
        },
        Type::Tuple(tup) => thread_tuple(value, tup, token, m),
        _ => match is_option_of_vector_pair(ty) {
            Some(Type::Tuple(tup)) => {
                let threaded = thread_tuple(quote! { __pair }, tup, token, m);
                quote! {
                    match #value {
                        Some(__pair) => Some(#threaded),
                        None => None,
                    }
                }
            }
            _ => value,
        },
    }
}

/// Is this a host scalar worth recording in the immediate?
fn is_host_scalar(ty: &Type) -> bool {
    let Type::Path(p) = ty else { return false };
    p.qself.is_none()
        && p.path.get_ident().is_some_and(|i| {
            matches!(
                i.to_string().as_str(),
                "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "usize" | "isize" | "bool"
            )
        })
}

/// Is this an element slice/array (`&[Self::Element]`, `&[Self::Element; N]`)?
fn is_element_slice(ty: &Type) -> bool {
    let elem_path = |t: &Type| {
        let Type::Path(p) = t else { return false };
        let segs: Vec<String> = p.path.segments.iter().map(|s| s.ident.to_string()).collect();
        matches!(segs.as_slice(), [s, e] if s == "Self" && e == "Element")
    };
    match ty {
        Type::Reference(r) if r.mutability.is_none() => match &*r.elem {
            Type::Slice(s) => elem_path(&s.elem),
            Type::Array(a) => elem_path(&a.elem),
            _ => false,
        },
        _ => false,
    }
}

/// `math_traits!` helper: the out-parameter threading statement for one
/// argument, if its type is `&mut [Self; N]`.
pub(crate) fn out_param(
    name: TokenStream2,
    ty: &Type,
    token: &TokenStream2,
    method: &TokenStream2,
) -> Option<TokenStream2> {
    if let Type::Reference(r) = ty
        && r.mutability.is_some()
        && let Type::Array(arr) = &*r.elem
        && is_vector(&arr.elem)
    {
        return Some(quote! {
            for __v in #name.iter_mut() {
                *__v = #method(*__v, #token);
            }
        });
    }
    None
}

/// `math_traits!` helper: the immediate's format string and value expressions
/// for a signature (const generics then host-valued arguments, declaration
/// order), or `None` when the call carries none.
pub(crate) fn imm_format(
    generics: &syn::Generics,
    args: &[(TokenStream2, Type)],
) -> Option<(String, Vec<TokenStream2>)> {
    let mut fmt = Vec::new();
    let mut values = Vec::new();

    for param in &generics.params {
        if let syn::GenericParam::Const(c) = param {
            let name = &c.ident;
            fmt.push("{}");
            values.push(quote! { #name });
        }
    }

    for (name, ty) in args {
        if is_host_scalar(ty) {
            fmt.push("{}");
            values.push(quote! { #name });
        } else if is_element_slice(ty) {
            fmt.push("{:?}");
            values.push(quote! { #name });
        }
    }

    if values.is_empty() { None } else { Some((fmt.join(", "), values)) }
}
