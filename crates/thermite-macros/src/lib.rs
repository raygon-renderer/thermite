extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::{
    Attribute, FnArg, GenericArgument, Ident, ImplItem, ItemImpl, ItemTrait, Pat, PathArguments, TraitItem, Type,
    parse_macro_input, parse_quote, parse_quote_spanned, punctuated::Punctuated, spanned::Spanned, token::Comma,
};

const SKIP_MASKED: &str = "skip_masked";
const WITH_CONDITIONAL: &str = "conditional";
const SKIP_CONDITIONAL: &str = "skip_conditional";

// --- Core Utilities ---

/// Helper to check for and remove specific internal attributes.
fn take_attribute(attrs: &mut Vec<Attribute>, name: &str) -> bool {
    let len = attrs.len();
    attrs.retain(|attr| !attr.path().is_ident(name));
    attrs.len() < len
}

/// Extracts documentation attributes from a list of attributes.
fn get_doc_attrs(attrs: &[Attribute]) -> Vec<&Attribute> {
    attrs.iter().filter(|attr| attr.path().is_ident("doc")).collect()
}

/// Extracts simple identifier names from function arguments for forwarding.
fn extract_arg_names(inputs: &Punctuated<FnArg, Comma>) -> impl Iterator<Item = &Ident> {
    inputs.iter().map(|arg| match arg {
        FnArg::Typed(pat_type) => match &*pat_type.pat {
            Pat::Ident(pat_ident) => &pat_ident.ident,
            _ => panic!("Macro only supports simple identifier arguments."),
        },
        FnArg::Receiver(_) => panic!("self receiver not supported."),
    })
}

#[proc_macro_attribute]
pub fn register_trait(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut trait_def = parse_macro_input!(item as ItemTrait);
    let mut new_items: Vec<TraitItem> = Vec::new();
    let skip_all = take_attribute(&mut trait_def.attrs, SKIP_MASKED);
    let all_conditional = take_attribute(&mut trait_def.attrs, WITH_CONDITIONAL);

    for item in &mut trait_def.items {
        if let TraitItem::Fn(method) = item {
            if method.default.is_some() {
                method.attrs.push(parse_quote!(#[inline(always)]));
            }

            if skip_all || take_attribute(&mut method.attrs, SKIP_MASKED) {
                continue;
            }

            let with_conditional = (take_attribute(&mut method.attrs, WITH_CONDITIONAL) || all_conditional)
                && !take_attribute(&mut method.attrs, SKIP_CONDITIONAL);

            let name = &method.sig.ident;
            let arg_names: Vec<_> = extract_arg_names(&method.sig.inputs).collect();
            let (_, ty_gen, _) = method.sig.generics.split_for_impl();
            let turbo = ty_gen.as_turbofish();
            let doc = get_doc_attrs(&method.attrs);
            let unsafety = method.sig.unsafety.as_ref();

            // shared among all variants
            // if the method is unsafe, wrap the call in an unsafe block.
            // while not strictly necessary, clippy will complain about calling
            // unsafe functions outside of an unsafe block, even if the function itself
            // is marked unsafe.
            let call = quote_spanned! { name.span() =>
                #unsafety { Self::#name #turbo(#(#arg_names),*) }
            };

            if with_conditional {
                // --- Conditional (_c) variant ---
                let mut sig_c = method.sig.clone();
                sig_c.ident = format_ident!("{}_c", name);

                let Some(this) = get_first_arg_name(&sig_c.inputs).cloned() else {
                    panic!("Expected at least one argument for conditional method.");
                };

                sig_c.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));

                let m_doc =
                    format!("Computes [`{name}`](Self::{name}) when `mask` is true, returns `{this}` where false.");
                new_items.push(TraitItem::Fn(parse_quote_spanned! { sig_c.span() =>
                    #(#doc)* #[doc = #m_doc] #[inline(always)] #sig_c {
                        Self::blendv(mask, #this, #call)
                    }
                }));
            }

            // --- Masked (_m) variant ---
            let mut sig_m = method.sig.clone();
            sig_m.ident = format_ident!("{}_m", name);
            sig_m.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));
            sig_m.inputs.insert(0, parse_quote!(src: Storage<Self>));

            let m_doc = format!("Merges [`{name}`](Self::{name}) with `src` using `mask`.");
            new_items.push(TraitItem::Fn(parse_quote_spanned! { sig_m.span() =>
                #(#doc)* #[doc = #m_doc] #[inline(always)] #sig_m {
                    Self::blendv(mask, src, #call)
                }
            }));

            // --- Zeroed (_z) variant ---
            let mut sig_z = method.sig.clone();
            sig_z.ident = format_ident!("{}_z", name);
            sig_z.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));

            let z_doc = format!("Computes [`{name}`](Self::{name}) masked (zeroed where mask is false).");
            new_items.push(TraitItem::Fn(parse_quote_spanned! { sig_z.span() =>
                #(#doc)* #[doc = #z_doc] #[inline(always)] #sig_z {
                    Self::blendv(mask, Self::EMPTY, #call)
                }
            }));
        }
    }

    trait_def.items.extend(new_items);
    trait_def.into_token_stream().into()
}

#[proc_macro_attribute]
pub fn double_pump_impl(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut impl_block = parse_macro_input!(item as ItemImpl);
    let inner_type = match extract_inner_generic(&impl_block.self_ty) {
        Some(ty) => ty,
        None => {
            return syn::Error::new_spanned(&impl_block.self_ty, "Expected DoublePumpRegister<R>")
                .to_compile_error()
                .into();
        }
    };

    let skip_all = take_attribute(&mut impl_block.attrs, SKIP_MASKED);
    let all_conditional = take_attribute(&mut impl_block.attrs, WITH_CONDITIONAL);

    let mut new_items = Vec::new();

    for item in &mut impl_block.items {
        if let ImplItem::Fn(method) = item {
            let skip = skip_all || take_attribute(&mut method.attrs, SKIP_MASKED);
            let with_conditional = (take_attribute(&mut method.attrs, WITH_CONDITIONAL) || all_conditional)
                && !take_attribute(&mut method.attrs, SKIP_CONDITIONAL);

            let name = &method.sig.ident;
            let unsafety = method.sig.unsafety.as_ref();
            let (_, ty_gen, _) = method.sig.generics.split_for_impl();
            let turbo = ty_gen.as_turbofish();

            // always mark as #[inline(always)], even if there is a custom body
            method.attrs.push(parse_quote!(#[inline(always)]));

            // 1. Generate base body if empty
            if method.block.stmts.is_empty() {
                let (args_0, args_1) = split_args_for_call(&method.sig.inputs);

                method.block = parse_quote!({
                    #unsafety { DoublePumpRegister(
                        #inner_type::#name #turbo(#args_0),
                        #inner_type::#name #turbo(#args_1)
                    ) }
                });
            }

            if !skip {
                let doc = get_doc_attrs(&method.attrs);

                if with_conditional {
                    // --- Generate _c ---
                    let mut sig_c = method.sig.clone();
                    sig_c.ident = format_ident!("{}_c", name);
                    sig_c.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));

                    let (c0, c1) = split_args_for_call(&sig_c.inputs);
                    let c_name = &sig_c.ident;

                    new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_c.span() =>
                        #(#doc)*
                        #[inline(always)]
                        #sig_c {
                            #unsafety { DoublePumpRegister(
                                #inner_type::#c_name #turbo(#c0),
                                #inner_type::#c_name #turbo(#c1)
                            ) }
                        }
                    }));
                }

                // --- Generate _m ---
                let mut sig_m = method.sig.clone();
                sig_m.ident = format_ident!("{}_m", name);
                sig_m.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));
                sig_m.inputs.insert(0, parse_quote!(src: Storage<Self>));

                let (m0, m1) = split_args_for_call(&sig_m.inputs);
                let m_name = &sig_m.ident;

                new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_m.span() =>
                    #(#doc)*
                    #[inline(always)]
                    #sig_m {
                        #unsafety { DoublePumpRegister(
                            #inner_type::#m_name #turbo(#m0),
                            #inner_type::#m_name #turbo(#m1)
                        ) }
                    }
                }));

                // --- Generate _z ---
                let mut sig_z = method.sig.clone();
                sig_z.ident = format_ident!("{}_z", name);
                sig_z.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));

                let (z0, z1) = split_args_for_call(&sig_z.inputs);
                let z_name = &sig_z.ident;

                new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_z.span() =>
                    #(#doc)*
                    #[inline(always)]
                    #sig_z {
                        #unsafety { DoublePumpRegister(
                            #inner_type::#z_name #turbo(#z0),
                            #inner_type::#z_name #turbo(#z1)
                        ) }
                    }
                }));
            }
        }
    }

    impl_block.items.extend(new_items);
    impl_block.into_token_stream().into()
}

#[proc_macro_attribute]
pub fn bitand_z(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut impl_block = parse_macro_input!(item as ItemImpl);
    let mut new_items = Vec::new();

    let skip_all = take_attribute(&mut impl_block.attrs, SKIP_MASKED);

    for item in &mut impl_block.items {
        if let ImplItem::Fn(method) = item {
            if skip_all || take_attribute(&mut method.attrs, SKIP_MASKED) {
                continue;
            }

            let name = &method.sig.ident;
            let unsafety = method.sig.unsafety.as_ref();
            let arg_names: Vec<_> = extract_arg_names(&method.sig.inputs).collect();
            let (_, ty_gen, _) = method.sig.generics.split_for_impl();
            let turbo = ty_gen.as_turbofish();
            let doc = method.attrs.iter().filter(|attr| attr.path().is_ident("doc"));

            let mut sig_z = method.sig.clone();
            sig_z.ident = format_ident!("{}_z", name);
            sig_z.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));

            let z_doc = format!("Computes [`{name}`](Self::{name}) zero-masked using bitwise AND.");
            new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_z.span() =>
                #(#doc)* #[doc = #z_doc]
                #[inline(always)]
                #sig_z { Self::bitand(mask, #unsafety { Self::#name #turbo(#(#arg_names),*) }) }
            }));
        }
    }

    impl_block.items.extend(new_items);
    impl_block.into_token_stream().into()
}

#[proc_macro_attribute]
pub fn vector_trait(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut trait_def = parse_macro_input!(item as ItemTrait);
    let mut new_items: Vec<TraitItem> = Vec::new();

    let skip_all = take_attribute(&mut trait_def.attrs, SKIP_MASKED);
    let all_conditional = take_attribute(&mut trait_def.attrs, WITH_CONDITIONAL);

    for item in &mut trait_def.items {
        if let TraitItem::Fn(method) = item {
            if method.default.is_some() {
                method.attrs.push(parse_quote!(#[inline(always)]));
            }

            if skip_all || take_attribute(&mut method.attrs, SKIP_MASKED) {
                continue;
            }

            let conditional = (take_attribute(&mut method.attrs, WITH_CONDITIONAL) || all_conditional)
                && !take_attribute(&mut method.attrs, SKIP_CONDITIONAL);

            let name = &method.sig.ident;

            let doc = get_doc_attrs(&method.attrs);

            if conditional {
                // --- Conditional (_c) variant ---
                let mut sig_c = method.sig.clone();
                sig_c.ident = format_ident!("{}_c", name);
                sig_c.inputs.insert(1, parse_quote!(mask: Self::Mask));

                let m_doc =
                    format!("Computes [`{name}`](Self::{name}) when `mask` is true, returns `self` where false.");
                new_items.push(TraitItem::Fn(
                    parse_quote_spanned! { sig_c.span() => #(#doc)* #[doc = #m_doc] #sig_c; },
                ));
            }

            // --- Masked (_m) variant ---
            // signature: fn method_m(self, src: Storage<Self>, mask: Storage<Self::Mask>, ...)
            let mut sig_m = method.sig.clone();
            sig_m.ident = format_ident!("{}_m", name);
            sig_m.inputs.insert(1, parse_quote!(mask: Self::Mask));
            sig_m.inputs.insert(1, parse_quote!(src: Self));

            let m_doc = format!("Merges [`{name}`](Self::{name}) with `src` using `mask`.");
            new_items.push(TraitItem::Fn(
                parse_quote_spanned! { sig_m.span() => #(#doc)* #[doc = #m_doc] #sig_m; },
            ));

            // --- Zeroed (_z) variant ---
            // signature: fn method_z(self, mask: Storage<Self::Mask>, ...)
            let mut sig_z = method.sig.clone();
            sig_z.ident = format_ident!("{}_z", name);
            sig_z.inputs.insert(1, parse_quote!(mask: Self::Mask));

            let z_doc = format!("Computes [`{name}`](Self::{name}) masked (zeroed where mask is false).");
            new_items.push(TraitItem::Fn(
                parse_quote_spanned! { sig_z.span() => #(#doc)* #[doc = #z_doc] #sig_z; },
            ));
        }
    }

    trait_def.items.extend(new_items);
    trait_def.into_token_stream().into()
}

// --- Private Helpers ---

fn extract_inner_generic(ty: &Type) -> Option<Type> {
    if let Type::Path(tp) = ty
        && let Some(segment) = tp.path.segments.last()
        && let PathArguments::AngleBracketed(args) = &segment.arguments
        && let Some(GenericArgument::Type(inner)) = args.args.first()
    {
        return Some(inner.clone());
    }

    None
}

#[rustfmt::skip]
fn is_splittable(ty: &Type) -> bool {
    let Type::Path(tp) = ty else { return false };

    tp.path.is_ident("Self") || tp.path.segments.last()
        .is_some_and(|s| s.ident == "Storage" || s.ident == "DoublePumpRegister")
}

fn split_args_for_call(inputs: &Punctuated<FnArg, Comma>) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    let mut a0 = Vec::new();
    let mut a1 = Vec::new();

    for input in inputs {
        if let FnArg::Typed(pt) = input {
            let Pat::Ident(pi) = &*pt.pat else { continue };

            let name = &pi.ident;

            if is_splittable(&pt.ty) {
                a0.push(quote!(#name.0));
                a1.push(quote!(#name.1));
            } else {
                a0.push(quote!(#name));
                a1.push(quote!(#name));
            }
        }
    }

    (quote!(#(#a0),*), quote!(#(#a1),*))
}

fn get_first_arg_name(inputs: &Punctuated<FnArg, Comma>) -> Option<&Ident> {
    for input in inputs {
        if let FnArg::Typed(pt) = input {
            let Pat::Ident(pi) = &*pt.pat else { continue };

            return Some(&pi.ident);
        }
    }

    None
}
