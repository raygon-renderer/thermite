#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate proc_macro;

use proc_macro::TokenStream;

mod dispatch;
mod internal;
mod late_bound;

/// Compile-time ISA dispatch for functions, `impl` blocks, traits, and modules.
///
/// Rewrites the annotated item so that every method/function body is wrapped in an
/// `#[inline(always)]` inner copy and then called through a per-backend
/// `#[target_feature(enable = "...")]` trampoline, selected at compile time by matching
/// on `<S as HasIsa>::ISA` - a const that is resolved when `S` is monomorphized.
///
/// # Syntax
///
/// ```rust,ignore
/// // `S` is the default SIMD type parameter name; override with a positional ident:
/// #[dispatch]
/// fn my_fn<S: HasIsa>(...) { ... }
///
/// // Explicit SIMD parameter name:
/// #[dispatch(V)]
/// fn my_fn<V: HasIsa>(...) { ... }
///
/// // Override the thermite crate path (needed when calling from inside thermite itself):
/// #[dispatch(thermite = "crate")]
/// fn my_fn<S: HasIsa>(...) { ... }
///
/// // Both together:
/// #[dispatch(V, thermite = "crate")]
/// fn my_fn<V: HasIsa>(...) { ... }
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
/// For a function `fn foo<S: HasIsa>(args...)`:
///
/// 1. The original body is moved into an `#[inline(always)]` copy named `foo`.
/// 2. For each backend a `#[target_feature(enable = "...")] unsafe fn __dispatch_<backend>`
///    is generated that calls `foo` under the appropriate CPU feature flags.
/// 3. The outer body becomes a `match <S as HasIsa>::ISA { ... }` that selects the
///    right trampoline.  Because `ISA` is a const, LLVM folds the match away at
///    monomorphization time - there is no runtime branch.
///
/// `impl` blocks use a private helper trait to allow the trampolines to call back into
/// `Self` without recursion.
///
/// # Methods with receivers (`&self`, `&mut self`, `self`)
///
/// Applying `#[dispatch]` directly to a single method that has a receiver requires
/// supplying the **concrete implementing type** as the first attribute argument.  This
/// is necessary because the macro only sees the method, not the surrounding `impl`
/// block, so it cannot infer `Self`.  `impl Trait for Self` is not valid inside a
/// function body.
///
/// ```rust,ignore
/// impl MyType {
///     // OK - concrete type supplied explicitly:
///     #[dispatch(MyType)]
///     fn process(&self) { ... }
/// }
/// ```
///
/// The supplied ident is used both as the dispatch match type
/// (`<MyType as HasIsa>::ISA`) and as the impl target of the internal helper trait.
///
/// **Prefer annotating the whole `impl` block** when all (or most) methods need
/// dispatch - it is less repetitive and avoids repeating the type name per method:
///
/// ```rust,ignore
/// #[dispatch(Self)]          // `Self` is resolved correctly at the impl-block level
/// impl MyType {
///     fn process(&self) { ... }
///     #[skip_dispatch]       // opt individual methods out if needed
///     fn helper(&self) { ... }
/// }
/// ```
#[proc_macro_attribute]
pub fn dispatch(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    dispatch::dispatch_inner(attr, item)
}

/// Runtime ISA-dispatched expression.
///
/// Comes in two forms:
///
/// - **Closure form** - wraps an arbitrary body in an `#[inline(always)]` inner function
///   generic over `S: Simd` (plus any caller-supplied extra generics), creates
///   `#[target_feature]`-annotated wrappers for each backend that has a complete
///   `Simd` implementation, then dispatches at runtime via `InstructionSet::get()`.
///   The syntax is similar to closures, but captures are done via arguments, and must
///   be typed.
/// - **Call form** - runtime-dispatches a single call to a `#[dispatch]` function
///   without any closure-like syntax: `dispatch_dyn!(my_kernel(a, b))`. See
///   *"The call form"* below.
///
/// # The dispatch boundary hides the chosen backend
///
/// The whole point of `dispatch_dyn!` is to pick the best available ISA at runtime and
/// run the body under it, so the **outside world cannot know which backend was chosen**,
/// and therefore cannot mention its SIMD types. Concrete vector types like
/// `Vector<<S as Simd>::f32x4>`, `f32xN`, etc. depend on the generic `S`, which only
/// exists *inside* the body. The per-backend `#[target_feature]` trampolines and the
/// outer `match` arm aren't generic over `S`; if a vector type appeared in the
/// parameter list or return type, that type would have nowhere to come from and
/// nowhere to go.
///
/// In practice this means:
///
/// - **Parameters and return types must be ISA-agnostic.** Use scalar types
///   (`f32`, `i32`, `bool`), slices (`&[f32]`, `&mut [f32]`), owned containers
///   (`Vec<f32>`, `Box<[f32]>`), or any `Copy` / non-SIMD type. **Never** use
///   `f32x4`, `f32xN`, `Vector<S::f32x4>`, `Mask<S::f32x4>`, etc. in the macro's
///   signature.
/// - **All SIMD work happens inside the body.** Load from slices into vectors,
///   process, store back out. The body is where the SIMD rewriter and the generic
///   `S` are in scope; the macro signature is the I/O contract with the surrounding
///   scalar world.
///
/// This is by design: a function that *returns* SIMD vectors couldn't have its
/// return type spelled at the call site (the caller doesn't know which backend ran),
/// so it couldn't be assigned to a variable or used in any way. The dispatch boundary
/// is necessarily scalar-shaped.
///
/// # Syntax
///
/// ```rust,ignore
/// // Basic form - no explicit dispatch binding. Signature uses only ISA-agnostic types.
/// dispatch_dyn!(|data: &[f32]| -> f32 { /* SIMD work here */ });
///
/// // Explicit dispatch binding (recommended): `for<Ident>` names the backend type.
/// // Default bound is `Simd3`. `S` is in scope inside the body:
/// dispatch_dyn!(for<S> |data: &mut [f32]| {
///     let (head, mid, tail) = data.try_aligned_simd_iter_mut::<f32xN>();
///     for v in mid { *v = v.sin(); }
///     /* ... */
/// });
///
/// // Custom bound - restrict or widen the set of usable Simd traits:
/// dispatch_dyn!(for<S: Simd> |data: &[f32]| -> f32 { /* ... */ });
/// dispatch_dyn!(for<S: Simd3 + MyCustomTrait> |data: &[f32]| { /* ... */ });
///
/// // With extra caller-provided generics and a where clause:
/// dispatch_dyn!(for<S> <T: Clone, const N: usize> |arg: T| -> T where T: Debug { arg });
///
/// // Override the thermite crate path (needed when calling from inside thermite itself):
/// dispatch_dyn!(thermite = "crate"; for<S> |data: &[f32]| -> f32 { /* ... */ });
/// ```
///
/// When `for<Ident>` is present, `Ident` is in scope inside `body` as a generic type
/// satisfying the stated bound (or `Simd3` by default).  When omitted, no explicit
/// dispatch binding is in scope - rely on the automatic SIMD type rewriting below.
///
/// # The call form
///
/// When the work is already packaged as a `#[dispatch]` function, the closure form's
/// machinery is redundant: the callee carries its own per-backend `#[target_feature]`
/// trampolines internally, so all that's needed at the call site is the runtime
/// backend selection. The call form provides exactly that:
///
/// ```rust,ignore
/// #[dispatch(S)]
/// fn dot<S: Simd>(a: &[f32], b: &[f32]) -> f32 { /* ... */ }
///
/// // Bare form: the selected backend is injected as the callee's ONLY generic
/// // argument. Expands to (roughly):
/// //   match InstructionSet::get() {
/// //       InstructionSet::X86V3 => dot::<X86V3>(&a, &b),
/// //       /* ...one arm per backend... */
/// //       _ => dot::<Scalar>(&a, &b),
/// //   }
/// let r = dispatch_dyn!(dot(&a, &b));
///
/// // `for<Ident>` form: `Ident` marks where the backend type goes, so callees
/// // with extra generic parameters work too:
/// let r = dispatch_dyn!(for<S> dot::<S>(&a, &b));
/// let r = dispatch_dyn!(for<S> scale::<S, f32>(&a, factor));
///
/// // The `for<Ident>` form also dispatches METHOD calls on a receiver, when the
/// // method itself is generic over the backend (a `#[dispatch(S)] impl` block):
/// #[dispatch(S)]
/// impl Kernel {
///     fn run<S: Simd>(&self, data: &[f32]) -> f32 { /* ... */ }
/// }
/// let r = dispatch_dyn!(for<S> kernel.run::<S>(&data));
/// ```
///
/// Rules and caveats:
///
/// - **The bare form requires the SIMD parameter to be the callee's only generic
///   parameter** (partial turbofish is not allowed in Rust). For callees with extra
///   type/const parameters, and for method calls, use the `for<Ident>` form and
///   write the turbofish yourself.
/// - **Keep the call form to a single dispatched call** (free function or method).
///   Do all other work outside the macro (`dispatch_dyn!(dot(&a, &b)).sqrt()`, not
///   the other way around): any code inside the macro that isn't the `#[dispatch]`
///   callee is compiled without target features.
/// - **Arguments are ordinary expressions**, evaluated in the selected arm - no
///   capture-by-name or reborrow rules apply, unlike the closure form. `dot(&data[..n])`
///   works directly.
/// - **The callee should be a `#[dispatch]` function.** The call form emits no
///   `#[target_feature]` wrappers of its own; calling a plain generic function through
///   it is still correct, but the body will be compiled without target features
///   (i.e. with scalar-quality codegen). Wrap arbitrary code in the closure form
///   instead.
/// - Trait bounds on the binder (`for<S: Bound>`) are rejected: the binder is replaced
///   by concrete backend types, so the callee's own bounds are what's checked.
/// - No automatic SIMD type rewriting is performed in the call form; `f32xN` etc. are
///   only rewritten inside closure-form bodies.
/// - The `thermite = "path";` prefix works the same as in the closure form.
///
/// Any extra generic parameters from `<...>` are assumed to be in scope at the macro call
/// site; the macro passes them through as explicit turbofish arguments.
///
/// # Automatic SIMD type rewriting
///
/// Before code generation the macro rewrites every **bare, unqualified** reference to a
/// known `Simd` associated-type name into its fully-qualified `Vector<S::...>` form.
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
///
/// # Argument forwarding and reborrow rules
///
/// The macro captures call-site locals **by name** - each parameter must correspond
/// to an in-scope binding of the same identifier. Forwarding to the per-backend
/// trampoline is type-driven, with different rules for slice DST parameters and
/// sized reference parameters:
///
/// | Declared parameter type | Emitted forwarding | What the caller may hold |
/// |---|---|---|
/// | `&[T]` / `&mut [T]` (slice DST)        | `&*ident` / `&mut *ident` | `Vec<T>`, `Box<[T]>`, `[T; N]`, `&[T]`, `&mut [T]` - any binding |
/// | `&str` / `&dyn Trait` (other DSTs)     | `&*ident` / `&mut *ident` | `String`, `Box<str>`, `Box<dyn Trait>`, `&str`, `&dyn Trait` |
/// | `&T` (sized, e.g. `&Vec<U>`, `&f64`)    | `&ident`                | the value itself, any binding, or a reference to it (deref-coerces) |
/// | `&mut T` (sized, e.g. `&mut Vec<U>`)   | `&mut ident`            | a `mut`-bound owned value **or** a `mut`-bound reference |
/// | anything else (by-value)               | `ident`                 | the value itself (moved) or a `Copy` primitive |
///
/// The slice-DST row uses `&*ident` rather than `&ident` so that `Vec<T>` / `Box<[T]>`
/// can be passed where `&[T]` is expected - without that, the user would have to write
/// `&vec[..]` at the call site. For sized reference parameters the macro emits a plain
/// borrow because `&*ident` would invoke `Deref{,Mut}` and overshoot - e.g. `&mut *vec`
/// where `vec: Vec<f64>` produces `&mut [f64]`, which does **not** match a `&mut Vec<f64>`
/// parameter.
///
/// ## Recommendations
///
/// 1. **Prefer slice / `str` parameters over wrapper types in the macro signature.**
///    `|data: &[f32]|` is more flexible than `|data: &Vec<f32>|` - it accepts owners,
///    boxed slices, and slice references uniformly without any binding gymnastics. Only
///    use `&Vec<T>` / `&mut Vec<T>` when you genuinely need wrapper-specific methods
///    (e.g. `.push`, `.reserve`, `.clear`).
///
/// 2. **For a sized `&mut T` parameter, mark the caller's binding `mut`.** If the
///    caller is a function parameter, write `mut` in the signature:
///
///    ```rust,ignore
///    // Won't compile - `spectrum_buf` is not a `mut` binding, so the macro's
///    // `&mut spectrum_buf` is rejected.
///    fn render(spectrum_buf: &mut Vec<f64>) {
///        dispatch_dyn!(|spectrum_buf: &mut Vec<f64>| { spectrum_buf.push(1.0); });
///    }
///
///    // OK - one extra `mut` makes the binding itself mutable.
///    fn render(mut spectrum_buf: &mut Vec<f64>) {
///        dispatch_dyn!(|spectrum_buf: &mut Vec<f64>| { spectrum_buf.push(1.0); });
///    }
///    ```
///
///    Or, if you cannot change the signature, reborrow into a local first:
///
///    ```rust,ignore
///    fn render(spectrum_buf: &mut Vec<f64>) {
///        let spectrum_buf = &mut *spectrum_buf;
///        dispatch_dyn!(|spectrum_buf: &mut Vec<f64>| { spectrum_buf.push(1.0); });
///    }
///    ```
///
/// 3. **Owned by-value parameters move the caller's binding.** `|data: Vec<f32>|`
///    consumes the caller's `data`. Use a slice / reference parameter if you want
///    the caller to retain ownership.
///
/// 4. **SIMD types belong only in the body - never in the signature.** See the
///    *"The dispatch boundary hides the chosen backend"* section above. The macro
///    cannot reasonably accept or return SIMD vector / mask types, because their
///    identity depends on the runtime-selected backend (`S`) which the surrounding
///    scalar code has no way to name. Treat each `dispatch_dyn!` invocation as a
///    scalar-in / scalar-out island around a region of SIMD work:
///
///    ```rust,ignore
///    // Pattern: load from slice, process with SIMD, store back to slice.
///    dispatch_dyn!(for<S> |xs: &[f32], out: &mut [f32]| {
///        let (head, mid, tail) = xs.try_aligned_simd_iter::<f32xN>();
///        let (oh,   om,  ot)   = out.try_aligned_simd_iter_mut::<f32xN>();
///        // ... SIMD body operates on f32xN<S> values ...
///    });
///    ```
///
///    If you need a vector value to escape the dispatch boundary, reduce it first
///    (`v.sum_elements()`, `v.max_element()`, `v.into_array()`) and return the scalar.
#[proc_macro]
pub fn dispatch_dyn(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    dispatch::dispatch_dyn_inner(input)
}

/// Derives `thermite::simd::HasIsa` by forwarding the `ISA` constant and the `Native`
/// backend type from a generic parameter.
///
/// By default the first type parameter is used as the source. Use `#[isa = S]` to pick a
/// different one. When the `thermite` crate is renamed, use `#[thermite = "other"]` to
/// refer to the correct crate name instead of the default external `::thermite` path.
///
/// # Example
/// ```ignore
/// #[derive(HasIsa)]
/// struct MyType<S: Simd, T> { ... }           // forwards from S
///
/// #[derive(HasIsa)]
/// #[isa = T]
/// struct MyType<S, T: Simd> { ... }           // forwards from T
///
/// // When the `thermite` crate is renamed to something else:
/// #[derive(HasIsa)]
/// #[thermite = "other"]
/// struct Inner<S: Simd> { ... }
/// ```
#[proc_macro_derive(HasIsa, attributes(isa, thermite))]
pub fn derive_has_isa(input: TokenStream) -> TokenStream {
    internal::derive_has_isa_inner(input)
}

#[proc_macro_attribute]
pub fn register_trait(attr: TokenStream, item: TokenStream) -> TokenStream {
    internal::register_trait_inner(attr, item)
}

#[proc_macro_attribute]
pub fn double_pump_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    internal::double_pump_impl_inner(attr, item)
}

#[proc_macro_attribute]
pub fn array_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    internal::array_impl_inner(attr, item)
}

#[proc_macro_attribute]
pub fn reduced_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    internal::reduced_impl_inner(attr, item)
}

#[proc_macro_attribute]
pub fn inline_always(attr: TokenStream, item: TokenStream) -> TokenStream {
    internal::inline_always_inner(attr, item)
}

#[proc_macro_attribute]
pub fn vector_trait(attr: TokenStream, item: TokenStream) -> TokenStream {
    internal::vector_trait_inner(attr, item)
}

#[proc_macro_attribute]
pub fn vector_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    internal::vector_impl_inner(attr, item)
}
