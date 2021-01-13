#![allow(unused)]

extern crate proc_macro;

use proc_macro2::{Span, TokenStream};
use quote::quote;

use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    visit_mut::VisitMut,
    Attribute, AttributeArgs, ConstParam, Expr, ExprCall, ExprPath, FnArg, GenericArgument, GenericMethodArgument,
    GenericParam, Ident, ImplItem, ImplItemMethod, Item, ItemFn, ItemImpl, ItemMod, ItemTrait, Lifetime, Lit, Meta,
    NestedMeta, Pat, PatType, Path, PathArguments, PathSegment, QSelf, Receiver, ReturnType, Signature, Token, Type,
    TypeParam, WherePredicate,
};

#[cfg(not(any(feature = "neon", feature = "wasm")))]
static BACKENDS: &[(&str, &str)] = &[
    ("SSE2", "sse2"),
    ("SSE42", "sse4.2"),
    ("AVX", "avx"),
    ("AVX2", "avx2,fma"),
];

#[cfg(feature = "neon")]
static BACKENDS: &[(&str, &str)] = &[("NEON", "neon")];

#[cfg(feature = "wasm32")]
static BACKENDS: &[(&str, &str)] = &[("WASM32", "simd128")];

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
        Item::Fn(fn_item) => gen_function(&attr.attributes, &fn_item),
        Item::Impl(impl_block) => gen_impl_block(&attr.attributes, impl_block),
        Item::Trait(trait_item) => gen_trait_def(&attr.attributes, trait_item),
        Item::Mod(module) => gen_mod_def(&attr.attributes, &module),
        _ => unimplemented!("#[dispatch] is only supported on naked functions, impl blocks or trait defintions!"),
    })
}

fn parse_attr(attr: &PunctuatedAttributes) -> (TokenStream, TokenStream) {
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

struct TypeVisitor {
    self_ty: Box<Type>,
}

impl VisitMut for TypeVisitor {
    fn visit_type_mut(&mut self, i: &mut Type) {
        match i {
            Type::Path(p) if p.qself.is_none() => {
                if p.path.is_ident("Self") {
                    *i = (*self.self_ty).clone();
                } else if p.path.segments.len() > 1 {
                    let first = p.path.segments.first_mut();
                    if first.map(|s| s.ident == "Self") == Some(true) {
                        let mut path = Punctuated::new();
                        let old_path = std::mem::replace(&mut p.path.segments, Punctuated::new());
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
                }
            }
            _ => {}
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
    self_arg: Expr,
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
            self_arg: Expr::Path(ExprPath {
                attrs: Vec::new(),
                qself: None,
                path: Path {
                    leading_colon: None,
                    segments: {
                        let mut segments = Punctuated::new();
                        segments.push(PathSegment {
                            ident: quote::format_ident!("self"),
                            arguments: PathArguments::None,
                        });
                        segments
                    },
                },
            }),
        }
    }
}

impl VisitMut for SelfTraitVisitor {
    // TODO: Avoid cloning as much
    fn visit_expr_mut(&mut self, i: &mut Expr) {
        if self.depth == 0 {
            match i {
                Expr::MethodCall(m) if m.method == self.method => match &mut *m.receiver {
                    Expr::Path(p) if p.path.is_ident("self") => {
                        let mut path = self.trait_.clone();
                        path.segments.push(PathSegment {
                            ident: m.method.clone(),
                            arguments: match &mut m.turbofish {
                                None => PathArguments::None,
                                Some(tf) => {
                                    let mut generic_args = Punctuated::new();
                                    let method_args = std::mem::replace(&mut tf.args, Punctuated::new());

                                    for arg in method_args.into_iter() {
                                        generic_args.push(match arg {
                                            GenericMethodArgument::Const(c) => GenericArgument::Const(c),
                                            GenericMethodArgument::Type(ty) => GenericArgument::Type(ty),
                                        });
                                    }

                                    PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
                                        colon2_token: Some(Default::default()),
                                        lt_token: Default::default(),
                                        args: generic_args,
                                        gt_token: Default::default(),
                                    })
                                }
                            },
                        });

                        let call_attrs = std::mem::replace(&mut m.attrs, Vec::new());
                        let path_attrs = std::mem::replace(&mut p.attrs, Vec::new());

                        *i = Expr::Call(ExprCall {
                            attrs: call_attrs,
                            func: Box::new(Expr::Path(ExprPath {
                                attrs: path_attrs,
                                qself: Some(self.qself.clone()),
                                path,
                            })),
                            paren_token: m.paren_token.clone(),
                            args: {
                                let mut args = Punctuated::new();
                                args.push(self.self_arg.clone()); // insert self first
                                for arg in m.args.iter() {
                                    args.push(arg.clone());
                                }
                                args
                            },
                        });
                    }
                    _ => {}
                },
                Expr::Call(c) => match &mut *c.func {
                    Expr::Path(p) if p.path.segments.len() == 2 => {
                        if p.path.segments.first().unwrap().ident == "Self"
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
                },
                _ => {}
            }
        }

        syn::visit_mut::visit_expr_mut(self, i);
    }

    // Ensure that we don't rewrite non-associated scopes by tracking when we enter and leave them.

    fn visit_item_fn_mut(&mut self, i: &mut syn::ItemFn) {
        self.depth += 1;
        syn::visit_mut::visit_item_fn_mut(self, i);
        self.depth -= 1;
    }

    fn visit_impl_item_method_mut(&mut self, i: &mut syn::ImplItemMethod) {
        self.depth += 1;
        syn::visit_mut::visit_impl_item_method_mut(self, i);
        self.depth -= 1;
    }

    fn visit_expr_closure_mut(&mut self, i: &mut syn::ExprClosure) {
        self.depth += 1;
        syn::visit_mut::visit_expr_closure_mut(self, i);
        self.depth -= 1;
    }

    fn visit_trait_item_method_mut(&mut self, i: &mut syn::TraitItemMethod) {
        self.depth += 1;
        syn::visit_mut::visit_trait_item_method_mut(self, i);
        self.depth -= 1;
    }
}

struct DemutSelfVisitor;

impl VisitMut for DemutSelfVisitor {
    fn visit_fn_arg_mut(&mut self, i: &mut FnArg) {
        match i {
            FnArg::Receiver(rcv) if rcv.reference.is_none() => {
                rcv.mutability = None;
            }
            _ => {}
        }

        syn::visit_mut::visit_fn_arg_mut(self, i);
    }
}

fn gen_mod_def(attr: &PunctuatedAttributes, mod_item: &ItemMod) -> TokenStream {
    let ItemMod {
        attrs,
        vis,
        mod_token,
        ident,
        content,
        semi,
    } = mod_item;

    let content = content.as_ref().map(|(_, items)| {
        let items = items.iter().map(|item| match item {
            Item::Fn(fn_item) => gen_function(attr, fn_item),
            Item::Impl(impl_block) => gen_impl_block(attr, impl_block.clone()),
            Item::Trait(trait_item) => gen_trait_def(attr, trait_item.clone()),
            Item::Mod(module) => gen_mod_def(attr, module),
            _ => quote! { #item },
        });

        quote! { { #(#items)* } }
    });

    quote! {
        #(#attrs)* #vis #mod_token #ident #content #semi
    }
}

fn gen_impl_block(attr: &PunctuatedAttributes, mut item_impl: ItemImpl) -> TokenStream {
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
    } = &mut item_impl;

    let mut tyv = TypeVisitor {
        self_ty: self_ty.clone(),
    };

    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();

    let extra_bounds = where_clause.map(|wc| {
        let mut extra_bounds = Vec::new();

        for pred in wc.predicates.iter() {
            match pred {
                WherePredicate::Type(ty) if ty.bounded_ty == **self_ty => extra_bounds.push(&ty.bounds),
                _ => {}
            }
        }

        quote! {
            : #(#extra_bounds +)*
        }
    });

    let items = items.iter_mut().map(|mut item| match item {
        ImplItem::Method(ImplItemMethod {
            attrs: fn_attrs,
            vis,
            defaultness,
            sig,
            block,
        }) => {
            let Signature {
                asyncness,
                unsafety,
                abi,
                ident,
                generics: fn_generics,
                inputs,
                output,
                ..
            } = &*sig;

            let helper_trait_name = quote::format_ident!("__DispatchHelper_{}", ident);

            let mut decl_sig = sig.clone();
            DemutSelfVisitor.visit_signature_mut(&mut decl_sig);

            tyv.visit_return_type_mut(&mut decl_sig.output);
            for arg in decl_sig.inputs.iter_mut() {
                tyv.visit_fn_arg_mut(arg);
            }
            if let Some(where_clause) = &mut decl_sig.generics.where_clause {
                tyv.visit_where_clause_mut(where_clause);
            }

            // If a trait implementation, disambiguate all calls to self. or Self:: to refer to the parent trait
            if let Some((_, trait_, _)) = &trait_ {
                SelfTraitVisitor::new(trait_.clone(), self_ty.clone(), sig.ident.clone()).visit_block_mut(block);
            }

            let (fn_impl_generics, _, fn_where_clause) = fn_generics.split_for_impl();

            // define the instrset branch functions for the dispatch trait
            let branch_defs = BACKENDS.iter().map(|(backend, instrset)| {
                let dispatch_ident = format_backend(backend);
                let inputs = &decl_sig.inputs;
                let output = &decl_sig.output;
                let where_clause = &decl_sig.generics.where_clause;

                quote! {
                    #asyncness unsafe #abi fn #dispatch_ident #fn_impl_generics(#inputs) #output #where_clause;
                }
            });

            // define the dispatch trait
            let dispatch_trait = quote! {
                #[allow(non_camel_case_types)]
                unsafe trait #helper_trait_name #impl_generics #extra_bounds #where_clause {
                    #defaultness #decl_sig;
                    #(#branch_defs)*
                }
            };

            // format args for forwarding
            let ref mut forward_args = forward_args(inputs.iter(), false);
            let ref mut forward_tys = forward_tys(fn_generics.params.iter(), inputs.iter());

            let tf = quote! { ::<#(#forward_tys),*> };

            // impl instrset methods
            let branch_impls = BACKENDS.iter().map(|(backend, instrset)| {
                let dispatch_ident = format_backend(backend);

                quote! {
                    #[inline]
                    #[target_feature(enable = #instrset)]
                    #asyncness unsafe #abi fn #dispatch_ident #fn_impl_generics(#inputs) #output #fn_where_clause {
                        <Self as #helper_trait_name #type_generics>::#ident #tf(#(#forward_args,)*)
                    }
                }
            });

            // impl Dispatch trait. We can copy the entire function definition here, which makes it really easy
            let dispatch_impl = quote! {
                unsafe impl #impl_generics #helper_trait_name #type_generics for #self_ty #where_clause {
                    #[inline(always)]
                    #defaultness #sig #block
                    #(#branch_impls)*
                }
            };

            // Define the match branches and calls to the dispatch functions
            let branches = BACKENDS.iter().map(|(backend, instrset)| {
                let dispatch_ident = format_backend(backend);
                let backend = quote::format_ident!("{}", backend);

                quote! {
                    #thermite::SimdInstructionSet::#backend => unsafe {
                        <Self as #helper_trait_name #type_generics>::#dispatch_ident #tf(#(#forward_args,)*)
                    }
                }
            });

            // define final function defintion, block, and match statement
            let res = quote! {
                #[allow(unused_mut)]
                #(#fn_attrs)* #vis #defaultness #sig {
                    #dispatch_trait
                    #dispatch_impl

                    match <#simd as #thermite::Simd>::INSTRSET {
                        #(#branches)*
                        _ => unsafe { #thermite::unreachable_unchecked() }
                    }
                }
            };

            res
        }
        _ => quote! { #item }, // unchanged
    });

    match &trait_ {
        Some((bang, trait_, for_)) => quote! {
            #defaultness #unsafety #impl_token #impl_generics #bang #trait_ #for_ #self_ty #where_clause {
                #(#items)*
            }
        },
        None => quote! {
            #defaultness #unsafety #impl_token #impl_generics #self_ty #where_clause {
                #(#items)*
            }
        },
    }
}

fn gen_trait_def(attrs: &PunctuatedAttributes, trait_item: ItemTrait) -> TokenStream {
    quote! { #trait_item }
}

fn format_backend(backend: &str) -> Ident {
    quote::format_ident!("__dispatch_{}", backend.to_lowercase())
}

/// Accumulate generic names for forwarding, with the exception of late-bound lifetimes
fn forward_tys<'a>(
    fn_generics: impl IntoIterator<Item = &'a GenericParam>,
    inputs: impl IntoIterator<Item = &'a FnArg> + Clone,
) -> Vec<TokenStream> {
    let mut forward_tys = Vec::new();

    'forwarding_tys: for generic in fn_generics.into_iter() {
        forward_tys.push(match generic {
            GenericParam::Type(ty) => {
                let ref ident = ty.ident;
                quote! { #ident }
            }
            GenericParam::Lifetime(lf) => {
                for arg in inputs.clone().into_iter() {
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

    forward_tys
}

fn forward_args<'a>(inputs: impl IntoIterator<Item = &'a FnArg>, inner: bool) -> Vec<TokenStream> {
    let mut forward_args = Vec::new();

    for input in inputs {
        forward_args.push(match input {
            FnArg::Receiver(rcv) => quote! { self },
            FnArg::Typed(arg) => {
                if let Pat::Ident(ref param) = *arg.pat {
                    let ref ident = param.ident;
                    quote! { #ident }
                } else {
                    unimplemented!()
                }
            }
        })
    }

    forward_args
}

fn gen_function(attr: &PunctuatedAttributes, item: &ItemFn) -> TokenStream {
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

    let ref mut forward_tys = forward_tys(generics.params.iter(), inputs.iter());

    // TODO: Test if inner function must always be redeclared
    let inner = quote! {
        #[inline(always)]
        #asyncness #unsafety #abi fn #ident #impl_generics(#inputs) #output #where_clause #block
    };

    let tf = quote! { ::<#(#forward_tys),*> };

    let mut branches = Vec::new();

    for (backend, instrset) in BACKENDS {
        let dispatch_ident = format_backend(backend);
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
