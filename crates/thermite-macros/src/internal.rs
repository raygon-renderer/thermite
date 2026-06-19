use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::{
    Attribute, FnArg, GenericArgument, Ident, ImplItem, ItemImpl, ItemTrait, Pat, PathArguments, ReturnType, TraitItem,
    Type, parse_macro_input, parse_quote, parse_quote_spanned, punctuated::Punctuated, spanned::Spanned, token::Comma,
};

const MASKED: &str = "masked";
const CONDITIONAL: &str = "conditional";

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
fn extract_trait_arg_names(inputs: &Punctuated<FnArg, Comma>) -> impl Iterator<Item = &Ident> {
    inputs.iter().map(|arg| match arg {
        FnArg::Typed(pat_type) => match &*pat_type.pat {
            Pat::Ident(pat_ident) => &pat_ident.ident,
            _ => panic!("Macro only supports simple identifier arguments."),
        },
        FnArg::Receiver(_) => panic!("self receiver not supported."),
    })
}

#[rustfmt::skip]
fn skip_or_conditional_impl(method: &mut syn::ImplItemFn) -> (bool, bool) {
    let conditional = take_attribute(&mut method.attrs, CONDITIONAL);
    let skip = !(conditional || take_attribute(&mut method.attrs, MASKED)) || is_ineligible_return_type(&method.sig.output);
    (skip, conditional)
}

#[rustfmt::skip]
fn skip_or_conditional_trait(method: &mut syn::TraitItemFn) -> (bool, bool) {
    let conditional = take_attribute(&mut method.attrs, CONDITIONAL);
    let skip = !(conditional || take_attribute(&mut method.attrs, MASKED)) || is_ineligible_return_type(&method.sig.output);
    (skip, conditional)
}

pub fn derive_has_isa_inner(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);

    let mut krate: Option<syn::Path> = None;
    let mut explicit_param: Option<Ident> = None;

    for attr in &input.attrs {
        let path = attr.path();

        if path.is_ident("thermite") {
            if let Ok(lit) = attr.parse_args::<syn::LitStr>() {
                krate = lit.parse().ok();
            }
        } else if path.is_ident("isa") {
            explicit_param = attr.parse_args::<Ident>().ok();
        }
    }

    let krate: syn::Path = krate.unwrap_or_else(|| syn::parse_quote!(::thermite));

    // Resolve which type parameter to forward from.
    let isa_param: Ident = match explicit_param {
        Some(ident) => ident,
        None => {
            let first = input.generics.type_params().next();
            match first {
                Some(tp) => tp.ident.clone(),
                None => {
                    return syn::Error::new_spanned(
                        &input.ident,
                        "#[derive(HasIsa)] requires at least one type parameter, \
                         or an explicit `#[isa = S]` attribute",
                    )
                    .to_compile_error()
                    .into();
                }
            }
        }
    };

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    quote! {
        impl #impl_generics #krate::simd::HasIsa for #name #ty_generics #where_clause {
            const ISA: #krate::isa::InstructionSet = <#isa_param as #krate::simd::HasIsa>::ISA;
        }
    }
    .into()
}

pub fn register_trait_inner(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut trait_def = parse_macro_input!(item as ItemTrait);
    let mut new_items: Vec<TraitItem> = Vec::new();

    for item in &mut trait_def.items {
        let TraitItem::Fn(method) = item else { continue };

        if method.default.is_some() {
            method.attrs.push(parse_quote!(#[inline(always)]));
        }

        let (skip, with_conditional) = skip_or_conditional_trait(method);

        if skip {
            continue;
        }

        let name = &method.sig.ident;
        let arg_names: Vec<_> = extract_trait_arg_names(&method.sig.inputs).collect();
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

            let m_doc = format!("Computes [`{name}`](Self::{name}) when `mask` is true, returns `{this}` where false.");
            new_items.push(TraitItem::Fn(parse_quote_spanned! { sig_c.span() =>
                #(#doc)* #[doc = #m_doc] #[inline(always)] #[allow(unused)] #sig_c {
                    Self::blendv(mask, #this, #call)
                }
            }));
        }

        // --- Masked (_m) variant ---
        let mut sig_m = method.sig.clone();
        let m_name = format_ident!("{}_m", name);

        sig_m.ident = m_name.clone();
        sig_m.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));
        sig_m.inputs.insert(0, parse_quote!(src: Storage<Self>));

        let m_doc = format!("Merges [`{name}`](Self::{name}) with `src` using `mask`.");
        new_items.push(TraitItem::Fn(parse_quote_spanned! { sig_m.span() =>
            #(#doc)* #[doc = #m_doc] #[inline(always)] #[allow(unused)] #sig_m {
                Self::blendv(mask, src, #call)
            }
        }));

        // --- Zeroed (_z) variant ---
        // For this, the default behavior should actually be to call the _m variant with EMPTY,
        // since the _m variant may have better defaults on older platforms.
        let mut sig_z = method.sig.clone();
        sig_z.ident = format_ident!("{}_z", name);
        sig_z.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));

        let z_doc = format!("Computes [`{name}`](Self::{name}) masked (zeroed where mask is false).");
        new_items.push(TraitItem::Fn(parse_quote_spanned! { sig_z.span() =>
            #(#doc)* #[doc = #z_doc] #[inline(always)] #[allow(unused)] #sig_z {
                if const { <Self as CoreRegister>::HAS_EQUAL_SIZE_MASK } {
                    Self::bitand(<Self as CoreRegister>::from_mask(mask), #unsafety { #call })
                } else {
                    #unsafety { Self::#m_name #turbo (Self::EMPTY, mask, #(#arg_names),*) }
                }
            }
        }));
    }

    trait_def.items.extend(new_items);
    trait_def.into_token_stream().into()
}

pub fn double_pump_impl_inner(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut impl_block = parse_macro_input!(item as ItemImpl);
    let reg_ty = match extract_inner_generic(&impl_block.self_ty) {
        Some(ty) => ty,
        None => {
            return syn::Error::new_spanned(&impl_block.self_ty, "Expected DoublePumpRegister<R>")
                .to_compile_error()
                .into();
        }
    };

    let mut new_items = Vec::new();

    for item in &mut impl_block.items {
        // we only care about functions
        let ImplItem::Fn(method) = item else { continue };

        let (skip, with_conditional) = skip_or_conditional_impl(method);

        let name = &method.sig.ident;
        let unsafety = method.sig.unsafety.as_ref();
        let (_, ty_gen, _) = method.sig.generics.split_for_impl();
        let turbo = ty_gen.as_turbofish();

        // always mark as #[inline(always)], even if there is a custom body
        method.attrs.push(parse_quote!(#[inline(always)]));

        // 1. Generate base body if empty
        if method.block.stmts.is_empty() {
            let (args_0, args_1) = split_args_for_dp_call(&method.sig.inputs);

            method.block = parse_quote!({
                #unsafety { DoublePumpRegister(
                    #reg_ty::#name #turbo(#args_0),
                    #reg_ty::#name #turbo(#args_1)
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

                let (c0, c1) = split_args_for_dp_call(&sig_c.inputs);
                let c_name = &sig_c.ident;

                new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_c.span() =>
                    #(#doc)* #[inline(always)] #[allow(unused)] #sig_c {
                        #unsafety { DoublePumpRegister(
                            #reg_ty::#c_name #turbo(#c0),
                            #reg_ty::#c_name #turbo(#c1)
                        ) }
                    }
                }));
            }

            // --- Generate _m ---
            let mut sig_m = method.sig.clone();
            sig_m.ident = format_ident!("{}_m", name);
            sig_m.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));
            sig_m.inputs.insert(0, parse_quote!(src: Storage<Self>));

            let (m0, m1) = split_args_for_dp_call(&sig_m.inputs);
            let m_name = &sig_m.ident;

            new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_m.span() =>
                #(#doc)* #[inline(always)] #[allow(unused)] #sig_m {
                    #unsafety { DoublePumpRegister(
                        #reg_ty::#m_name #turbo(#m0),
                        #reg_ty::#m_name #turbo(#m1)
                    ) }
                }
            }));

            // --- Generate _z ---
            let mut sig_z = method.sig.clone();
            sig_z.ident = format_ident!("{}_z", name);
            sig_z.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));

            let (z0, z1) = split_args_for_dp_call(&sig_z.inputs);
            let z_name = &sig_z.ident;

            new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_z.span() =>
                #(#doc)* #[inline(always)] #[allow(unused)] #sig_z {
                    #unsafety { DoublePumpRegister(
                        #reg_ty::#z_name #turbo(#z0),
                        #reg_ty::#z_name #turbo(#z1)
                    ) }
                }
            }));
        }
    }

    impl_block.items.extend(new_items);
    impl_block.into_token_stream().into()
}

pub fn array_impl_inner(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut impl_block = parse_macro_input!(item as ItemImpl);
    let reg_ty = match extract_inner_generic(&impl_block.self_ty) {
        Some(ty) => ty,
        None => {
            return syn::Error::new_spanned(&impl_block.self_ty, "Expected ArrayRegister<R, N>")
                .to_compile_error()
                .into();
        }
    };

    let mut new_items = Vec::new();

    for item in &mut impl_block.items {
        let ImplItem::Fn(method) = item else { continue };

        let (skip, with_conditional) = skip_or_conditional_impl(method);

        let name = &method.sig.ident;
        let unsafety = method.sig.unsafety.as_ref();
        let (_, ty_gen, _) = method.sig.generics.split_for_impl();
        let turbo = ty_gen.as_turbofish();

        // always mark as #[inline(always)], even if there is a custom body
        method.attrs.push(parse_quote!(#[inline(always)]));

        let make_body = |sig: &syn::Signature, target_name: &Ident| -> proc_macro2::TokenStream {
            let num_inputs = sig.inputs.len();
            let mut arrays = Vec::with_capacity(num_inputs);
            let mut closure_params = Vec::with_capacity(num_inputs);
            let mut call_args = Vec::with_capacity(num_inputs);

            for input in &sig.inputs {
                if let FnArg::Typed(pt) = input {
                    let Pat::Ident(pi) = &*pt.pat else { continue };
                    let arg_name = &pi.ident;

                    if is_splittable(&pt.ty) {
                        arrays.push(quote!(#arg_name.0));
                        let reg_name = format_ident!("{}_reg", arg_name);
                        closure_params.push(reg_name.clone());
                        call_args.push(reg_name.to_token_stream());
                    } else {
                        call_args.push(arg_name.to_token_stream());
                    }
                }
            }

            let call = quote_spanned! { target_name.span() =>
                #unsafety { #reg_ty::#target_name #turbo(#(#call_args),*) }
            };

            match arrays.len() {
                0 => quote!({ ArrayRegister(#call) }),

                1 => {
                    let a0 = &arrays[0];
                    let p0 = &closure_params[0];
                    quote!({ ArrayRegister(#a0.map(#[inline(always)] |#p0| #call)) })
                }

                n => {
                    let array_zip = format_ident!("array_zip{n}");

                    quote!({
                       ArrayRegister(#array_zip(#(#arrays),*, #[inline(always)] |#(#closure_params),*| #call))
                    })
                }
            }
        };

        // 1. Generate base body if empty
        if method.block.stmts.is_empty() {
            let body = make_body(&method.sig, name);
            method.block = parse_quote!( #body );
        }

        if !skip {
            let doc = get_doc_attrs(&method.attrs);

            if with_conditional {
                // --- Generate _c ---
                let mut sig_c = method.sig.clone();
                sig_c.ident = format_ident!("{}_c", name);
                sig_c.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));

                let c_name = &sig_c.ident;
                let body = make_body(&sig_c, c_name);

                new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_c.span() =>
                    #(#doc)* #[inline(always)] #[allow(unused)] #sig_c #body
                }));
            }

            // --- Generate _m ---
            let mut sig_m = method.sig.clone();
            sig_m.ident = format_ident!("{}_m", name);
            sig_m.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));
            sig_m.inputs.insert(0, parse_quote!(src: Storage<Self>));

            let m_name = &sig_m.ident;
            let body = make_body(&sig_m, m_name);

            new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_m.span() =>
                #(#doc)* #[inline(always)] #[allow(unused)] #sig_m #body
            }));

            // --- Generate _z ---
            let mut sig_z = method.sig.clone();
            sig_z.ident = format_ident!("{}_z", name);
            sig_z.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));

            let z_name = &sig_z.ident;
            let body = make_body(&sig_z, z_name);

            new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_z.span() =>
                #(#doc)* #[inline(always)] #[allow(unused)] #sig_z #body
            }));
        }
    }

    impl_block.items.extend(new_items);
    impl_block.into_token_stream().into()
}

pub fn reduced_impl_inner(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut impl_block = parse_macro_input!(item as ItemImpl);
    let reg_ty = match extract_inner_generic(&impl_block.self_ty) {
        Some(ty) => ty,
        None => {
            return syn::Error::new_spanned(&impl_block.self_ty, "Expected ReducedRegister<R>")
                .to_compile_error()
                .into();
        }
    };

    let mut new_items = Vec::new();

    for item in &mut impl_block.items {
        // we only care about functions
        let ImplItem::Fn(method) = item else { continue };

        let (skip, with_conditional) = skip_or_conditional_impl(method);

        let name = &method.sig.ident;
        let unsafety = method.sig.unsafety.as_ref();
        let (_, ty_gen, _) = method.sig.generics.split_for_impl();
        let turbo = ty_gen.as_turbofish();

        // always mark as #[inline(always)], even if there is a custom body
        method.attrs.push(parse_quote!(#[inline(always)]));

        // 1. Generate base body if empty
        if method.block.stmts.is_empty() {
            let args = args_for_reduced_call(&method.sig.inputs);

            method.block = parse_quote!({
                #unsafety { ReducedRegister( #reg_ty::#name #turbo(#args), PhantomData ) }
            });
        }

        if !skip {
            let doc = get_doc_attrs(&method.attrs);

            if with_conditional {
                // --- Generate _c ---
                let mut sig_c = method.sig.clone();
                sig_c.ident = format_ident!("{}_c", name);
                sig_c.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));

                let args = args_for_reduced_call(&sig_c.inputs);
                let c_name = &sig_c.ident;

                new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_c.span() =>
                    #(#doc)* #[inline(always)] #[allow(unused)] #sig_c {
                        #unsafety { ReducedRegister( #reg_ty::#c_name #turbo(#args), PhantomData ) }
                    }
                }));
            }

            // --- Generate _m ---
            let mut sig_m = method.sig.clone();
            sig_m.ident = format_ident!("{}_m", name);
            sig_m.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));
            sig_m.inputs.insert(0, parse_quote!(src: Storage<Self>));

            let args = args_for_reduced_call(&sig_m.inputs);
            let m_name = &sig_m.ident;

            new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_m.span() =>
                #(#doc)* #[inline(always)] #[allow(unused)] #sig_m {
                    #unsafety { ReducedRegister( #reg_ty::#m_name #turbo(#args), PhantomData ) }
                }
            }));

            // --- Generate _z ---
            let mut sig_z = method.sig.clone();
            sig_z.ident = format_ident!("{}_z", name);
            sig_z.inputs.insert(0, parse_quote!(mask: Storage<Self::Mask>));

            let args = args_for_reduced_call(&sig_z.inputs);
            let z_name = &sig_z.ident;

            new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_z.span() =>
                #(#doc)* #[inline(always)] #[allow(unused)] #sig_z {
                    #unsafety { ReducedRegister( #reg_ty::#z_name #turbo(#args), PhantomData ) }
                }
            }));
        }
    }

    impl_block.items.extend(new_items);
    impl_block.into_token_stream().into()
}

pub fn inline_always_inner(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut impl_block = parse_macro_input!(item as ItemImpl);

    let inline_always: syn::Attribute = parse_quote!(#[inline(always)]);

    for item in &mut impl_block.items {
        let ImplItem::Fn(method) = item else { continue };
        method.attrs.push(inline_always.clone());
    }

    impl_block.into_token_stream().into()
}

pub fn vector_trait_inner(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut trait_def = parse_macro_input!(item as ItemTrait);
    let mut new_items: Vec<TraitItem> = Vec::new();

    for item in &mut trait_def.items {
        // we only care about functions
        let TraitItem::Fn(method) = item else { continue };

        if method.default.is_some() {
            method.attrs.push(parse_quote!(#[inline(always)]));
        }

        let (skip, conditional) = skip_or_conditional_trait(method);

        if skip {
            continue;
        }

        let name = &method.sig.ident;

        let doc = get_doc_attrs(&method.attrs);

        // 1 if method has a self receiver, 0 otherwise.
        // We want to insert new arguments after the self receiver if it exists.
        let insert_idx = method
            .sig
            .inputs
            .first()
            .map(|arg| matches!(arg, FnArg::Receiver(_)))
            .unwrap_or(false) as usize;

        if conditional {
            // --- Conditional (_c) variant ---
            let mut sig_c = method.sig.clone();
            sig_c.ident = format_ident!("{}_c", name);
            sig_c.inputs.insert(insert_idx, parse_quote!(mask: Self::Mask));

            let m_doc = format!("Computes [`{name}`](Self::{name}) when `mask` is true, returns `self` where false.");
            new_items.push(TraitItem::Fn(
                parse_quote_spanned! { sig_c.span() => #(#doc)* #[doc = #m_doc] #sig_c; },
            ));
        }

        // --- Masked (_m) variant ---
        // signature: fn method_m(self, src: Storage<Self>, mask: Storage<Self::Mask>, ...)
        let mut sig_m = method.sig.clone();
        sig_m.ident = format_ident!("{}_m", name);
        sig_m.inputs.insert(insert_idx, parse_quote!(mask: Self::Mask));
        sig_m.inputs.insert(insert_idx, parse_quote!(src: Self));

        let m_doc = format!("Merges [`{name}`](Self::{name}) with `src` using `mask`.");
        new_items.push(TraitItem::Fn(
            parse_quote_spanned! { sig_m.span() => #(#doc)* #[doc = #m_doc] #sig_m; },
        ));

        // --- Zeroed (_z) variant ---
        // signature: fn method_z(self, mask: Storage<Self::Mask>, ...)
        let mut sig_z = method.sig.clone();
        sig_z.ident = format_ident!("{}_z", name);
        sig_z.inputs.insert(insert_idx, parse_quote!(mask: Self::Mask));

        let z_doc = format!("Computes [`{name}`](Self::{name}) masked (zeroed where mask is false).");
        new_items.push(TraitItem::Fn(
            parse_quote_spanned! { sig_z.span() => #(#doc)* #[doc = #z_doc] #sig_z; },
        ));
    }

    trait_def.items.extend(new_items);
    trait_def.into_token_stream().into()
}

pub fn vector_impl_inner(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut impl_block = parse_macro_input!(item as ItemImpl);
    let reg_ty = match extract_inner_generic(&impl_block.self_ty) {
        Some(ty) => ty,
        None => {
            return syn::Error::new_spanned(&impl_block.self_ty, "Expected Vector<R>")
                .to_compile_error()
                .into();
        }
    };

    let mut new_items = Vec::new();

    for item in &mut impl_block.items {
        let ImplItem::Fn(method) = item else { continue };

        let (skip, with_conditional) = skip_or_conditional_impl(method);

        let name = &method.sig.ident;
        let unsafety = method.sig.unsafety.as_ref();
        let (_, ty_gen, _) = method.sig.generics.split_for_impl();
        let turbo = ty_gen.as_turbofish();

        // always mark as #[inline(always)], even if there is a custom body
        method.attrs.push(parse_quote!(#[inline(always)]));

        if method.block.stmts.is_empty() {
            let args = args_for_vector_call(&method.sig.inputs);

            method.block = parse_quote_spanned!(method.span() => {
                Vector(#unsafety { #reg_ty::#name #turbo(#args) })
            });
        }

        if skip {
            continue;
        }

        let doc = get_doc_attrs(&method.attrs);

        let insert_idx = method
            .sig
            .inputs
            .first()
            .map(|arg| matches!(arg, FnArg::Receiver(_)))
            .unwrap_or(false) as usize;

        if with_conditional {
            // --- Generate _c ---
            let mut sig_c = method.sig.clone();
            sig_c.ident = format_ident!("{}_c", name);

            let args = args_for_vector_call(&sig_c.inputs);

            sig_c.inputs.insert(insert_idx, parse_quote!(mask: Mask<#reg_ty>));

            let c_name = &sig_c.ident;

            new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_c.span() =>
                #(#doc)* #[inline(always)] #[allow(unused)] #sig_c {
                    Vector(#unsafety { #reg_ty::#c_name #turbo(mask.0, #args) })
                }
            }));
        }

        // --- Generate _m ---
        let mut sig_m = method.sig.clone();
        sig_m.ident = format_ident!("{}_m", name);

        let args = args_for_vector_call(&sig_m.inputs);

        sig_m.inputs.insert(insert_idx, parse_quote!(mask: Mask<#reg_ty>));
        sig_m.inputs.insert(insert_idx, parse_quote!(src: Self));

        let m_name = &sig_m.ident;

        new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_m.span() =>
            #(#doc)* #[inline(always)] #[allow(unused)] #sig_m {
                Vector(#unsafety { #reg_ty::#m_name #turbo(src.0, mask.0, #args) })
            }
        }));

        // --- Generate _z ---
        let mut sig_z = method.sig.clone();
        sig_z.ident = format_ident!("{}_z", name);

        let args = args_for_vector_call(&sig_z.inputs);

        sig_z.inputs.insert(insert_idx, parse_quote!(mask: Mask<#reg_ty>));

        let z_name = &sig_z.ident;

        new_items.push(ImplItem::Fn(parse_quote_spanned! { sig_z.span() =>
            #(#doc)* #[inline(always)] #[allow(unused)] #sig_z {
                Vector(#unsafety { #reg_ty::#z_name #turbo(mask.0, #args) })
            }
        }));
    }

    impl_block.items.extend(new_items);
    impl_block.into_token_stream().into()
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
    let tp = match ty {
        Type::Path(tp) => tp,
        Type::Reference(r) => return is_splittable(&r.elem),
        _ => return false,
    };

    tp.path.is_ident("Self") || tp.path.segments.last()
        .is_some_and(|s| s.ident == "Storage" || s.ident == "DoublePumpRegister" || s.ident == "ReducedRegister" || s.ident == "Self")
}

#[rustfmt::skip]
fn is_vectorlike_type(ty: &Type) -> bool {
    let tp = match ty {
        Type::Path(tp) => tp,
        Type::Reference(r) => return is_vectorlike_type(&r.elem),
        _ => return false,
    };

    // Allow `Self::*`
    if tp.path.segments.first().is_some_and(|s| s.ident == "Self") {
        // Unless it's `Self::Element`
        if let Some(second) = tp.path.segments.get(1) && second.ident == "Element" {
            return false;
        }

        return true;
    }

    // Allow `Self`, `Vector`, `Mask`
    tp.path.is_ident("Self") || tp.path.segments.last().is_some_and(|s| s.ident == "Vector" || s.ident == "Mask")
}

fn is_ineligible_return_type(ty: &ReturnType) -> bool {
    match ty {
        ReturnType::Type(_, ty) => is_ineligible_type(ty),
        ReturnType::Default => true,
    }
}

fn is_ineligible_type(ty: &Type) -> bool {
    let tp = match ty {
        Type::Path(tp) => tp,
        Type::Reference(r) => return is_ineligible_type(&r.elem),
        _ => return false,
    };

    let Some(last) = tp.path.segments.last() else {
        return false;
    };

    if last.ident == "Element" {
        return true;
    }

    // `Storage<Self::Something>`
    if last.ident == "Storage"
        && let PathArguments::AngleBracketed(args) = &last.arguments
        && let Some(GenericArgument::Type(inner)) = args.args.first()
        && let Type::Path(inner_tp) = inner
        && let Some(inner_first) = inner_tp.path.segments.first()
        && inner_first.ident == "Self"
        && inner_tp.path.segments.len() > 1
    {
        return true;
    }

    false
}

fn split_args_for_dp_call(inputs: &Punctuated<FnArg, Comma>) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
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
                let name = name.to_token_stream();
                a0.push(name.clone());
                a1.push(name);
            }
        }
    }

    (quote!(#(#a0),*), quote!(#(#a1),*))
}

fn args_for_reduced_call(inputs: &Punctuated<FnArg, Comma>) -> proc_macro2::TokenStream {
    let args = inputs.iter().filter_map(|input| match input {
        FnArg::Receiver(r) => Some(quote_spanned!(r.span() => self.0)),
        FnArg::Typed(pt) => {
            let Pat::Ident(pi) = &*pt.pat else {
                return None;
            };

            let name = &pi.ident;

            Some(if is_splittable(&pt.ty) {
                quote!(#name.0)
            } else {
                name.to_token_stream()
            })
        }
    });

    quote!(#(#args),*)
}

fn args_for_vector_call(inputs: &Punctuated<FnArg, Comma>) -> proc_macro2::TokenStream {
    let args = inputs.iter().filter_map(|input| match input {
        FnArg::Receiver(r) => Some(quote_spanned!(r.span() => self.0)),
        FnArg::Typed(pt) => {
            let Pat::Ident(pi) = &*pt.pat else {
                return None;
            };

            let name = &pi.ident;

            Some(if is_vectorlike_type(&pt.ty) {
                quote!(#name.0)
            } else {
                name.to_token_stream()
            })
        }
    });

    quote!(#(#args),*)
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
