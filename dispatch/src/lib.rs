#![allow(unused)]

extern crate proc_macro;

use proc_macro2::TokenStream;
use quote::quote;

use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    visit_mut::VisitMut,
    Attribute, AttributeArgs, ConstParam, FnArg, GenericArgument, GenericParam, Ident, ImplItem, ImplItemMethod, Item,
    ItemFn, ItemImpl, Lifetime, Lit, Meta, NestedMeta, Pat, PatType, PathArguments, Receiver, Signature, Token, Type,
    TypeParam,
};

// TODO: Add more
static BACKENDS: &[(&str, &str)] = &[("AVX", "avx,fma"), ("AVX2", "avx2,fma")];

type PunctuatedAttributes = Punctuated<NestedMeta, Token![,]>;

struct DispatchAttributes {
    pub attributes: PunctuatedAttributes,
}

impl Parse for DispatchAttributes {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(DispatchAttributes {
            attributes: Punctuated::parse_terminated(input)?,
        })
    }
}

/// Generates monomorphized backend `target_feature` function calls to the annotated function or `impl` block.
#[proc_macro_attribute]
pub fn dispatch(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let attr = syn::parse_macro_input!(attr as DispatchAttributes);
    let item = syn::parse_macro_input!(item as Item);

    proc_macro::TokenStream::from(match item {
        Item::Fn(fn_item) => gen_function(attr.attributes, fn_item),
        Item::Impl(impl_block) => gen_impl_block(attr.attributes, impl_block),
        _ => unimplemented!(),
    })
}

fn parse_attr(attr: PunctuatedAttributes) -> (TokenStream, TokenStream) {
    let default_simd = quote::format_ident!("S");
    let mut simd = quote! { #default_simd };
    let mut thermite = quote! { ::thermite };

    for attr in attr.iter() {
        match attr {
            NestedMeta::Meta(Meta::Path(path)) => {
                simd = quote! { #path };
            }
            NestedMeta::Meta(Meta::NameValue(nv)) if nv.path.is_ident("thermite") => {
                let lit = match nv.lit {
                    Lit::Str(ref s) => s.value(),
                    _ => panic!("Invalid thermite path: {:?}", nv.lit.span()),
                };
                let path = quote::format_ident!("{}", lit);
                thermite = quote! { #path };
            }
            _ => {}
        }
    }

    (simd, thermite)
}

fn gen_impl_block(attr: PunctuatedAttributes, item: ItemImpl) -> TokenStream {
    let (simd, thermite) = parse_attr(attr);

    let ItemImpl {
        attrs,
        defaultness,
        unsafety,
        impl_token,
        generics,
        trait_,
        self_ty,
        items,
        ..
    } = &item;

    let mut impl_lifetimes = Vec::new();
    let mut impl_generics_not_lifetimes = Vec::new();

    for g in generics.params.iter() {
        match g {
            GenericParam::Lifetime(lf) => impl_lifetimes.push(&lf.lifetime),
            _ => impl_generics_not_lifetimes.push(g),
        }
    }

    let (impl_generics, _, where_clause) = generics.split_for_impl();

    let items = items.iter().map(|item| {
        match item {
            ImplItem::Method(method) => {
                let ImplItemMethod {
                    attrs: fn_attrs,
                    vis,
                    defaultness,
                    sig,
                    block,
                } = method;

                let Signature {
                    asyncness,
                    unsafety,
                    abi,
                    ident,
                    generics: fn_generics,
                    inputs,
                    output,
                    ..
                } = sig;

                let mut lifetimes = impl_lifetimes.clone();
                let mut impl_generics_not_lifetimes = impl_generics_not_lifetimes.clone();

                for g in fn_generics.params.iter() {
                    match g {
                        GenericParam::Lifetime(lf) => lifetimes.push(&lf.lifetime),
                        _ => impl_generics_not_lifetimes.push(g),
                    }
                }

                let ref where_clause = generics
                    .where_clause.as_ref()
                    .into_iter()
                    .chain(fn_generics.where_clause.as_ref().into_iter())
                    .map(|wc| wc.predicates.clone().into_iter())
                    .flatten().collect::<Vec<_>>();

                let forward_args_inner = forward_args(inputs.iter(), true);
                let forward_args_outer = forward_args(inputs.iter(), false);

                let tf = {
                    let lifetimes = lifetimes.iter().filter_map(|lf| {
                        for arg in inputs.iter() {
                            match arg {
                                FnArg::Typed(ty) if is_late_bound(lf, &ty.ty) => return None,
                                _ => {}
                            }
                        }

                        Some(quote! { #lf })
                    });

                    let generics = impl_generics_not_lifetimes.iter().map(|g| {
                        match g {
                            GenericParam::Type(TypeParam { ident, ..}) | GenericParam::Const(ConstParam { ident, ..}) => quote! { #ident },
                            _ => unimplemented!()
                        }
                    });

                    quote! {
                        ::< #(#lifetimes,)* #(#generics,)* >
                    }
                };

                let ref renamed_inputs: Vec<_> = inputs.iter().map(|input| {
                    match input {
                        FnArg::Receiver(rcv) => {
                            let Receiver { attrs, reference, mutability, .. } = rcv;

                            let reference = match reference {
                                Some((_, lf)) => quote! { & #lf #mutability },
                                None => quote! {}
                            };

                            quote! { #(#attrs)* __self: #reference #self_ty }
                        }
                        _ => quote! { #input }
                    }
                }).collect();

                let mut block = block.clone();

                // Replace all occurrences of `self` with `__self`
                SelfVisitor.visit_block_mut(&mut block);

                let inner = quote! {
                    #[inline(always)]
                    #unsafety fn #ident< #(#lifetimes,)* #(#impl_generics_not_lifetimes,)* >(#(#renamed_inputs,)*) #output where #(#where_clause,)* #block
                };

                let branches = BACKENDS.iter().map(|(backend, instrset)| {
                    let dispatch_ident = quote::format_ident!("__dispatch_{}", backend.to_lowercase());
                    let backend = quote::format_ident!("{}", backend);

                    quote! {
                        #thermite::SimdInstructionSet::#backend => {
                            #[inline]
                            #[target_feature(enable = #instrset)]
                            #asyncness unsafe fn #dispatch_ident < #(#lifetimes,)* #(#impl_generics_not_lifetimes,)* >
                            (#(#renamed_inputs,)*) #output where #(#where_clause,)* {
                                // call named inner function
                                #ident #tf (#(#forward_args_inner,)*)
                            }

                            unsafe { #dispatch_ident #tf (#(#forward_args_outer,)*) }
                        }
                    }
                });

                let (fn_impl_generics, _, fn_where_clause) = fn_generics.split_for_impl();

                quote! {
                    #(#fn_attrs)* #vis #unsafety fn #ident #fn_impl_generics(#inputs) #output #fn_where_clause {
                        #inner
                        match <#simd as #thermite::Simd>::INSTRSET {
                            #(#branches)*
                            _ => unsafe { #thermite::unreachable_unchecked() }
                        }
                    }
                }
            }
            _ => quote! { #item }, // verbatim
        }
    });

    let res = match trait_ {
        Some((bang, name, for_)) => quote! {
            #defaultness #unsafety #impl_token #impl_generics #bang #name #for_ #self_ty #where_clause {
                #(#items)*
            }
        },
        None => quote! {
            #defaultness #unsafety #impl_token #impl_generics #self_ty #where_clause {
                #(#items)*
            }
        },
    };

    res
}

fn forward_args<'a>(inputs: impl IntoIterator<Item = &'a FnArg>, inner: bool) -> Vec<TokenStream> {
    let mut forward_args = Vec::new();

    for input in inputs {
        forward_args.push(match input {
            FnArg::Typed(arg) => {
                if let Pat::Ident(ref param) = *arg.pat {
                    let ref ident = param.ident;
                    quote! { #ident }
                } else {
                    unimplemented!()
                }
            }
            FnArg::Receiver(rcv) => {
                if inner {
                    quote! { __self }
                } else {
                    quote! { self }
                }
            }
        })
    }

    forward_args
}

fn gen_function(attr: PunctuatedAttributes, item: ItemFn) -> TokenStream {
    let (simd, thermite) = parse_attr(attr);

    let ItemFn {
        sig,
        vis,
        block,
        attrs: fn_attrs,
    } = item;

    let Signature {
        asyncness,
        unsafety,
        abi,
        ident,
        generics,
        inputs,
        output,
        ..
    } = &sig;

    // let this handle half of the work
    let (impl_generics, _, where_clause) = generics.split_for_impl();

    let ref mut forward_args = forward_args(inputs.iter(), false);

    let ref mut forward_tys = Vec::new();

    // Accumulate generic names for forwarding, with the exception of late-bound lifetimes
    'forwarding_tys: for generic in generics.params.iter() {
        forward_tys.push(match generic {
            GenericParam::Type(ty) => {
                let ref ident = ty.ident;
                quote! { #ident }
            }
            GenericParam::Lifetime(lf) => {
                for arg in inputs.iter() {
                    match arg {
                        FnArg::Typed(ty) if is_late_bound(&lf.lifetime, &*ty.ty) => continue 'forwarding_tys,
                        _ => {}
                    }
                }
                quote! { #lf }
            }
            GenericParam::Const(c) => {
                let ref ident = c.ident;
                quote! { #ident }
            }
        });
    }

    // TODO: Test if inner function must always be redeclared
    let inner = quote! {
        #[inline(always)]
        #asyncness #unsafety #abi fn #ident #impl_generics(#inputs) #output #where_clause #block
    };

    let tf = quote! { ::<#(#forward_tys),*> };

    let mut branches = Vec::new();

    for (backend, instrset) in BACKENDS {
        let dispatch_ident = quote::format_ident!("__dispatch_{}", backend.to_lowercase());
        let backend = quote::format_ident!("{}", backend);

        branches.push(quote! {
            #thermite::SimdInstructionSet::#backend => {
                #[inline]
                #[target_feature(enable = #instrset)]
                #asyncness unsafe fn #dispatch_ident #impl_generics(#inputs) #output #where_clause {
                    // call named inner function
                    #ident #tf (#(#forward_args,)*)
                }

                unsafe { #dispatch_ident #tf (#(#forward_args,)*) }
            }
        });
    }

    quote! {
        #(#fn_attrs)* #vis #asyncness #unsafety #abi fn #ident #impl_generics(#inputs) #output #where_clause {
            #inner

            match <#simd as #thermite::Simd>::INSTRSET {
                #(#branches,)*
                _ => unsafe { #thermite::unreachable_unchecked() }
            }
        }
    }
}

struct SelfVisitor;
impl VisitMut for SelfVisitor {
    fn visit_ident_mut(&mut self, i: &mut Ident) {
        if i == "self" {
            let span = i.span();
            *i = quote::format_ident!("__self");
            i.set_span(span);
        }
    }
}

// Checks if a lifetime is late-bound anywhere in a type
fn is_late_bound(lf: &Lifetime, ty: &Type) -> bool {
    match ty {
        Type::Reference(r) => Some(lf) == r.lifetime.as_ref() || is_late_bound(lf, &*r.elem),
        Type::Array(a) => is_late_bound(lf, &*a.elem),
        Type::Group(g) => is_late_bound(lf, &*g.elem),
        Type::ImplTrait(i) => i.bounds.iter().any(|bound| match bound {
            syn::TypeParamBound::Lifetime(l) => l == lf,
            _ => false,
        }),
        Type::Paren(p) => is_late_bound(lf, &*p.elem),
        Type::Ptr(p) => is_late_bound(lf, &*p.elem),
        Type::Slice(s) => is_late_bound(lf, &*s.elem),
        Type::TraitObject(t) => t.bounds.iter().any(|bound| match bound {
            syn::TypeParamBound::Lifetime(l) => l == lf,
            _ => false,
        }),
        Type::Tuple(t) => t.elems.iter().any(|elem| is_late_bound(lf, elem)),
        Type::Path(p) => p.path.segments.iter().any(|seg| match &seg.arguments {
            PathArguments::Parenthesized(p) => p.inputs.iter().any(|ty| is_late_bound(lf, ty)),
            PathArguments::AngleBracketed(a) => a.args.iter().any(|arg| match arg {
                GenericArgument::Lifetime(l) => l == lf,
                GenericArgument::Type(ty) => is_late_bound(lf, ty),
                GenericArgument::Binding(b) => is_late_bound(lf, &b.ty),
                GenericArgument::Constraint(c) => c.bounds.iter().any(|bound| match bound {
                    syn::TypeParamBound::Lifetime(l) => l == lf,
                    syn::TypeParamBound::Trait(t) => unimplemented!(),
                }),
                _ => unimplemented!(),
            }),
            _ => false,
        }),
        Type::BareFn(f) => unimplemented!(),
        _ => false,
    }
}
