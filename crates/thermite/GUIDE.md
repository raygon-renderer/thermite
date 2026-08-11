Thermite Guide
==============

## Rule 0

Never touch the `Register` layer or individual backend register types unless you
know exactly what you're doing.

## Introduction

Thermite's goal is to enable writing simple, high-performance code and have them be generic both across hardware backends and across vector widths. The same code can potentially be invoked with `f32x8` on AVX2 (x86-v3) or with `f64x2` on ARM NEON.

However, before we get into that, there are a few rules:

### Rules

0. Do not touch the `Register` layer unless you know exactly what you're doing. Stick with the `*Vector` traits.
1. `#[thermite::dispatch]` and `#[inline(always)]` are **MANDATORY**, or else your code will be abysmally slow.
2. Prefer trait bounds over concrete vector types. Commit to a fixed lane count when the data structure is genuinely defined by it, not out of convenience.
3. Thermite isn't magic. You must consider what you're doing before you expect it to be fast.
4. Look for any built-in methods before re-implementing things yourself. All of Thermite's standard library is highly optimized.
5. Avoid Scalar work whenever possible.
6. Doing nothing is better than doing something clever.

I'll explain more on these as we go.

### Example

```rust
use thermite::{prelude::*, math::TranscendentalMath};

#[thermite::dispatch(V)]
fn gaussian<V: TranscendentalMath>(x: V) -> V {
    (-x * x).exp()
}
```

Nothing in the `gaussian` signature names a backend, a vector width, or even an element type,
only the capability the body needs. `#[thermite::dispatch]` is what keeps it *fast*: it
emits one `#[target_feature]`-annotated copy of the body per backend, which is the only
way the intrinsics behind `exp` can be inlined at all. If `#[thermite::dispatch]` isn't included,
Rust will "forget" which backend this is compiling for, and de-optimize the entire thing. Seriously,
`#[thermite::dispatch]` is **NECESSARY** to the highest degree. Skipping it is worse than
not using SIMD at all.

Worth noting before you write your own: `thermite-special` already has a `gaussian`, in the
general form with amplitude and width parameters. It's used here because everyone recognizes
it, but it's also a small demonstration of Rule 4, since the example function is itself a
thing you wouldn't have needed to write.

To run the function, you have to cross from ordinary scalar code into SIMD code, and that
boundary is `dispatch_dyn!`. It detects the CPU once at runtime, then calls the copy of
the body compiled for the best ISA available. Inside the block, bare width names are
rewritten to the chosen backend's vector types: `f32x4` is that backend's 4-lane `f32`
vector, and `f32xN` is its *widest native* one (4 lanes on SSE2 or NEON, 8 on AVX2, 16
on AVX-512). Ask the type how wide it turned out to be rather than hard-coding it:

```rust
# use thermite::{prelude::*, math::TranscendentalMath};
# #[thermite::dispatch(V)]
# fn gaussian<V: TranscendentalMath>(x: V) -> V { (-x * x).exp() }
#
// 15 elements: odd, so there is a remainder left over at every native width.
let xs: Vec<f32> = (0..15).map(|i| i as f32 * 0.25 - 1.75).collect();
let mut ys = vec![0.0f32; xs.len()]; // output

// Do note, this is not a real closure. `dispatch_dyn!` creates a real function
// behind the scenes to pass values into under the correct backend and target features.
// See the `dispatch_dyn!` docs for full usage examples and limitations.
thermite::dispatch_dyn!(|xs: &[f32], ys: &mut [f32]| {
    let mut x_chunks = xs.chunks_exact(f32xN::lanes());
    let mut y_chunks = ys.chunks_exact_mut(f32xN::lanes());

    // SIMD-accelerated mapping
    for (x, y) in Iterator::zip(&mut x_chunks, &mut y_chunks) {
        gaussian(f32xN::from_slice(x)).copy_to_slice(y);
    }

    // The leftovers go through the *same* function, using a 1-lane scalar `Vector<f32>` instead.
    for (x, y) in Iterator::zip(x_chunks.remainder().iter(), y_chunks.into_remainder()) {
        *y = gaussian(x.as_vector()).extract::<0>(); // `as_vector` wraps the scalar in the `Vector` type
    }
});

for (&x, &y) in xs.iter().zip(&ys) {
    // we can use the scalar path as a "ground truth" to ensure the vectorized kernel is correct
    assert!((y - gaussian(x.as_vector()).extract::<0>()).abs() < 1e-6);
}
```

The closure's parameters are captured by name from the surrounding scope (`xs` and `ys`
here), and both they and the return type must be ISA-agnostic (plain scalars, slices,
`Vec`, and so on). A vector type can never appear in that signature, because the caller
has no way to know which backend was picked. All the SIMD work happens inside: load from
slices, compute, store back.

## Getting Started

Thermite isn't on crates.io yet, so depend on it by git:

```toml
[dependencies]
thermite = { git = "https://github.com/raygon-renderer/thermite" }
```

Add whichever companion crates you need from the same repository. The defaults are the
right defaults, so there's usually nothing to configure.

It builds on **stable Rust**, with an MSRV of **1.95** and edition 2024. The crate is
`no_std` by default, so a std application wants `features = ["std"]`, which mostly buys
formatted panic messages.

Of the feature flags, five are worth knowing about:

- **`std`** enables std. Off by default.
- **`wasm`** turns on the WASM SIMD128 backend. It stays opt-in because SIMD128 is a
  proposal an engine may not have enabled. There is deliberately no `neon` feature, since
  NEON is mandatory on AArch64 and the backend is always compiled there.
- **`strict_ieee754`** gives spec-exact denormals, NaNs, and min/max. It implies
  `preserve_denormals` and `disable_fast_fma`, and it is much slower. Turn it on knowing
  what you're buying.
- **`disable_dispatch`** replaces runtime dispatch with `#[inline(always)]`. It is an
  advanced option that bloats or slows most builds, and it is not the way to make dispatch
  cheaper.
- **`nightly`** unlocks the nightly-only paths, and then requires a nightly compiler. The
  AVX-512 tiers and the experimental SPIR-V backend live behind their own flags.

One more deserves a warning rather than a description. **`algebraic-scalar`** lets LLVM
reassociate float arithmetic on the scalar backend so its loops autovectorize, which is a
real win where it applies. It is also fundamentally incompatible with compensated
arithmetic, because reassociation is exactly what destroys the error terms double-double
precision is built on, and `thermite-compensated` refuses to compile alongside it. Cargo
unifies features across the whole dependency graph, so an unrelated crate switching this on
can break a build that never asked for it.

The two `avx2-` defaults assume that a CPU with AVX2 also has F16C and PCLMULQDQ, which is
true of every shipping AVX2 part, and they let Thermite skip a runtime check.

The smallest thing that works, with no macros and no dispatch, is three lines:

```rust
use thermite::prelude::*;

let x = Vector::<f32>::splat(3.0);
let y = x * x;

assert_eq!(y.extract::<0>(), 9.0);
```

That runs on the scalar backend, one lane wide, which is why it needs no dispatch at all.
It is also the same code path the tail of a real kernel takes, and the reference
implementation you check a vectorized version against. Everything after this is about
getting more lanes doing that work at once.

## Thermite Architecture

Thermite stacks three layers, `Element` to `Register` to `Vector`, and you only ever touch
the last one. The two that matter to this section are backend registers and `Vector` traits. The registers are low-level, backend-specific types that wrap the CPU's raw SIMD registers. The `Vector` traits are the high-level side: the same operations with the same semantics, no matter which backend sits underneath. When writing generic code, you should always use the `Vector` traits and avoid touching the backend registers directly. This ensures that your code remains portable and can take advantage of different SIMD architectures without modification.

However, that being said, knowing different parameters of the backend can be useful for performance tuning. For example, knowing the number of lanes in a vector can help you optimize your algorithms to minimize the number of iterations and maximize the use of SIMD operations.

Every type that implements `GenericVector` also requires `HasIsa`, which is an escape hatch of sorts to some of the native hardware's capabilities. Furthermore, any custom type can `#[derive(HasIsa)]` to forward that information. `#[thermite::dispatch(T)]` requires that the given type `T` implements `HasIsa`, that's how it can know which backend to dispatch to. `thermite::dispatch_dyn!` can inject that type into the pseudo-closure, so it can be passed down.

It's important to understand that the Vector traits don't necessarily have to be anything other than what those traits require. Thermite provides a considerable ecosystem of composite types such as `Complex`, `Dual`, and `Compensated` that implement the same traits, which notably _any generic function accepting some vector type will also accept those composite types_. This allows for a great deal of flexibility and code reuse, as you can write generic functions that work with a wide variety of vector types without having to write separate implementations for each one. `thermite-dual` even provides a simple `AutoDiff` trait that can be used to invoke a generic function with dual numbers to provide automatic forward-mode differentiation without lifting a finger. Entire codebases could be written in terms of generic vector types, and then later be run with `Dual` or `Compensated` to get automatic differentiation or higher precision without any changes to the original code.

Composite types can even interact with each other to a degree, such as `Complex<Dual<V, N>>`. There are some limitations, but the general idea is that you can compose these types to get the desired behavior. For example, you could use `Complex<Compensated<V>>` to get complex arithmetic in double-double precision, or `Dual<Compensated<V>, N>` to differentiate in double-double precision. This could be especially useful in scientific computing applications where both complex numbers and high precision are required, or places where double-precision isn't available, such as on GPUs, where `Compensated` can be used to reach near double-precision accuracy with only single-precision hardware. Absolutely wild, right? Even `thermite-sort` automatically works with `Dual` and `Compensated` types, so you can sort in double-double precision or sort complex numbers without any extra work (in their in-memory SoA forms, at least).

## Vector Widths

Every width is available on every backend. For the 32-bit and 64-bit element families
(`f32`, `i32`, `u32`, `f64`, `i64`, `u64`, and `usize`) that means `x2`, `x4`, `x8`, and
`x16`, and the same four widths exist for the small integer families (`i8`, `u8`, `i16`,
`u16`). There are also 3-lane forms for the cases where three components is the natural
shape. Each name is parameterized by the backend, so you write `f32x8<S>` and get a fully
typed vector for whichever `S` you're compiling against.

That uniformity is deliberate. Code that wants sixteen lanes of `f32` can ask for sixteen
lanes of `f32` and it will compile and run correctly on a machine whose registers hold
four. What it cannot do is make that free.

When you ask for a width the hardware doesn't have, Thermite builds it out of the widths
the hardware does have. A sixteen-lane `f32` vector on a machine with eight-lane registers
is represented as two of them, and every operation is applied to both halves. This is not a
scalar fallback, and it is not slow in the way a fallback would be. It's still real vector
code, and for a straight-line arithmetic kernel it costs about what you would expect, which
is roughly twice the work for twice the lanes.

It stops being free when an operation has to cross the boundary between those registers.
Anything that moves data between lanes, such as a shuffle, a reduction, a scan, or a
cross-lane compare, has to be stitched together across the pieces, and that costs more than
the same operation on a single native register. The same applies going the other way: a
width narrower than the register leaves part of the register idle, so you're paying full
price for a fraction of the throughput.

### That's what `xN` is for

The `xN` names solve this. `f32xN`, `i32xN`, `u64xN` and friends resolve to whatever the
current backend's native width actually is, so there's nothing to emulate and no lanes left
idle:

| Backend | `f32xN` etc. | `f64xN` etc. | `u16xN` etc. | `u8xN` etc. | Status |
|---|---|---|---|---|---|
| SSE2, SSE4.2 | 4 | 2 | 8 | 16 | ready |
| AVX2 | 8 | 4 | 16 | 32 | ready |
| NEON | 4 | 2 | 8 | 16 | ready |
| WASM SIMD128 | 4 | 2 | 8 | 16 | ready, opt-in via `wasm` |
| Scalar | 1 | 1 | 1 | 1 | ready |
| AVX-512 | 16 | 8 | 32 | 64 | **not implemented yet** |
| SPIR-V | 1 | 1 | | | **experimental, not ready to use** |

The last two are worth being clear about. Their designs are settled, and the widths above
are what they will be, but neither is finished. AVX-512 currently exists as the tier system
that describes which sub-extensions a given part actually has, since AVX-512 is a foundation
plus a dozen optional extensions rather than one ISA, and the register implementations
behind it are still to come. That one is a question of time and of having the hardware to
validate against. The SPIR-V backend is further along but still experimental, sits behind
its own feature flag, and requires nightly. Don't build on either yet.

The scalar backend being genuinely one lane is not a placeholder, it's the point. It's what
makes the same generic function usable as its own reference implementation, which the
tail-handling loop in the introduction relies on. SPIR-V reports one lane for a different
reason: on a GPU the parallelism comes from running many invocations rather than from wide
registers, so the width lives in the dispatch instead of in the type.

### Small elements get very wide

Look at the right-hand columns above. A register is a fixed number of *bits*, so the
smaller the element the more of them fit. The same AVX2 register that holds eight `f32`
lanes holds thirty-two `u8` lanes, and on AVX-512 that becomes sixty-four.

This has a consequence that catches people, because the fixed-width names stop at `x16` for
every element family. There is no `u8x32`, and there is no `u8x64`. Those registers are
real and Thermite uses them, but the only way to name one is `u8xN`. If you write `u8x16`
on AVX2 because sixteen sounded like plenty, you have asked for half a register and you
will get half the throughput, on hardware that was ready to give you all of it.

It matters for more than throughput. Some instructions only exist at the full width, so
reaching them at all means going through `u8xN`. The 256-bit sum-of-absolute-differences on
AVX2 is one of these: it lives on the 32-lane `u8` register, which has no fixed-width name,
so `u8xN` is the only door to it.

So the rule from the previous section applies with more force here, not less. For byte and
short work, which is most classification, text processing, and image work, reach for `xN`
and let it be as wide as the machine allows.

So the guidance is short. Inside a dispatched region, reach for `xN` by default and let the
backend decide how wide that is.

### Fewer lanes than the register

Going the other direction has its own story. A two-lane vector on a machine with four-lane
registers is not a smaller register, because there is no such thing. It's the full register
with the upper lanes carried along and ignored.

That's the right choice, and it beats falling back to scalars, but it isn't free either.
The unused lanes hold whatever they hold, and for most operations that's harmless, since
adding two lanes you'll never read costs nothing extra. For some it isn't harmless, and the
ignored lanes have to be cleaned up so their garbage doesn't leak into the answer. Anything
that reduces across lanes is the obvious case, and so is anything where a junk value could
raise a flag or produce a NaN that then spreads. So a narrow vector is usually the same
speed as the full-width one and occasionally a little slower, while doing a fraction of the
work either way.

There's one exception worth knowing. For the smallest element types, a two-lane vector is
literally two scalars rather than a masked-off register, because at that size there's
nothing to mask off usefully.

The practical read is the same as before. Two-lane vectors are for when the problem has two
components, not for when you want a smaller register, because you will not get one.

### Three lanes, and the two ways to spell it

Three-lane vectors are the case most people actually want, because 3D is full of them. They
work, and using them is fine. There are two flavors, and the difference matters.

The `x3A` types are three lanes stored in a four-lane register, with the fourth deliberately
unused. Alignment and performance characteristics are those of the four-lane register,
because that's what it is. This is the predictable one, and on CPU backends it is what you
want.

The `x3` types are true three-lane vectors, and they make no promise about their storage.
A backend may implement them as an honest three-component register where the hardware has
one, which GPUs do, or as anything else it likes. On most CPU backends they end up identical
to the `x3A` types, since CPU SIMD has no three-lane representation to offer. They are also
not automatically available, since each backend has to opt in and define them.

The reason to care is portability of intent. If you want the CPU-friendly padded layout,
say `x3A` and you'll get it everywhere. If you want a genuine three-component vector where
one exists, which on a shader backend is the native thing and the padded form is not, say
`x3` and let the backend decide. That's the only route to a true three-component vector on
SPIR-V.

### Size is not a contract, alignment is

The above has a consequence worth stating plainly, because it's the kind of thing that bites
once and confuses for an afternoon.

A vector type is an abstraction over a register, not a struct with a guaranteed layout.
`size_of::<V>()` is therefore not reliably `size_of::<V::Element>() * V::LANES`. A three-lane
vector padded into four lanes is larger than three elements. A width assembled from several
native registers may carry its own arrangement. None of this is a defect, it's what lets one
type name work across backends that disagree about what registers exist.

`align_of::<V>()` is a different matter, and it is exactly as trustworthy as it looks. It's
the real alignment of the underlying register, and it's the number that decides whether an
aligned access is legal. When something asks for an aligned pointer, this is the value it
means. Allocate to it, assert against it, and build your containers around it with
confidence.

The rest of the contract is everything the API hands you. `LANES` and `lanes()` give the
lane count. The element type is what you asked for. Loads, stores, `from_slice`,
`copy_to_slice`, `into_array`, and the slice iterators all round-trip exactly the lanes you
expect, in order. Those hold everywhere.

The byte footprint is the one thing that isn't in the contract. So don't compute offsets
into a buffer of vectors by hand, don't assume a `[V; N]` is bit-identical to an array of
elements, and don't transmute between the two. Use the conversions. The library takes the
same care internally: the slice alignment helpers explicitly check whether a vector's size
matches its elements times its lanes, and when it doesn't they hand the whole slice back as
unaligned rather than pretending otherwise.

### When a fixed width is the right answer

The default is not a prohibition, and it's worth being clear about that, because the
opposite mistake also exists.

Some structures are defined by their width rather than merely implemented at it. A wide
tree whose nodes fan out to exactly eight children is the clearest case. That eight is not
a SIMD detail that leaked into the design, it's the branching factor, and it determines the
shape of the tree, how deep traversal goes, how much of a node a single test rejects, and
how well the whole thing behaves statistically. Change it to four or sixteen and you have a
different data structure with different properties, not the same one running at a different
speed. The same goes for a four-component color, a fixed transform, or a table whose size is
part of the problem statement.

In those cases the fixed width is the design, and writing it as a fixed width is correct
even on a machine whose registers don't match. A structure whose properties you chose
deliberately is worth more than the throughput you would claw back by letting the width
float, and the emulation described above means the code stays correct and reasonably fast
everywhere regardless.

The distinction is whether the number came from the problem or from the hardware you
happened to be testing on. If a reader cannot tell which, say so in a comment. What remains
a genuine mistake is a *generic* function that inspects `V::LANES` and changes its algorithm
based on the answer. That is a different thing from committing to a width at the type level,
and it comes up again under generic kernels below.

## Essential Entry Points

Other than `dispatch_dyn!` and `#[dispatch]`, the other essential entry points are the `Vector` traits themselves. The `Vector` traits provide a consistent interface for vector operations across different backends, and they are the primary way to interact with Thermite's SIMD capabilities. The most important traits are:

- `GenericVector` is the base trait for all vector types. It assumes nothing other than a sequence of values, without any particular arithmetic.
- `NumericVector` adds arithmetic operations, but not necessarily floating-point ones.
- `FloatVector` adds floating-point operations, including transcendental functions.
- `IntegerVector`/`SignedIntegerVector`/`UnsignedIntegerVector` add integer operations, in signed and unsigned variants.

I highly suggest simply browsing the [`thermite::vector`] module docs to see what is available. The traits are designed to be as general as possible, so you can use them in a wide variety of contexts.

`FloatVectorWithBits` is a special trait that allows you to access the underlying bit representation of floating-point vectors, which can be very useful for certain operations, such as bit manipulation or implementing certain algorithms that require direct access to the bits of floating-point numbers.

## Writing Generic Kernels

This is the part that takes the most getting used to, so it's worth being explicit about.
Every kernel you write is generic over some `V`, and the only real question is what goes
after the colon. Ask for exactly the capability the body uses and nothing more, because
every bound you add is a type that can no longer call your function.

Start with the loosest thing that compiles. A two-component dot product over SoA lanes
needs multiplication and addition, so `NumericVector` covers it, and integer vectors keep
working:

```rust
use thermite::prelude::*;

#[inline(always)]
fn dot2<V: NumericVector>(ax: V, ay: V, bx: V, by: V) -> V {
    ax * bx + ay * by
}
```

That's a multiply followed by an add, which is exactly what fused multiply-add exists for.
Reaching for `mul_adde` costs one more bound, because `MulAddExt` is a supertrait of
`FloatVector` rather than `NumericVector`:

```rust
use thermite::prelude::*;

#[inline(always)]
fn dot2<V: FloatVector>(ax: V, ay: V, bx: V, by: V) -> V {
    ax.mul_adde(bx, ay * by)
}
```

`FloatVector` is a strictly smaller set of types (integers are gone now), but in exchange
the body becomes one instruction on any backend with FMA, and stays correct on the ones
without it. That trade is the whole game, so make it deliberately.

### Check the surface before you write anything

Rule 4 is worth repeating here, because the trait surface is far larger than it looks.
`mix` (linear interpolation), `scale` (multiply by a scalar), `clamp`, `step`, `smoothstep`,
`rescale`, `hypot`, `logaddexp`, `one_minus_sq` and several hundred others already exist,
most with a policy-aware `_p` twin and a masked variant.

Some of them exist precisely because the obvious spelling is wrong. `one_minus_sq` is not
a convenience wrapper around `1 - x * x`. That spelling loses all its significant digits
as `x` approaches `±1`, so the built-in picks a cancellation-free formulation based on
whether the backend has FMA. Skim [`thermite::vector`] and
[`thermite::math`] before writing a helper. The odds are good that it already exists, and
that it's faster and more accurate than what you were about to write.

### What to write after `V:`

| You need | Bound |
|---|---|
| `+ - * /`, comparisons, `min`/`max` | `NumericVector` |
| negation, `abs`, sign handling | `SignedVector` |
| `sqrt`, `floor`, `round`, FMA | `FloatVector` |
| `exp`, `ln`, `sin`, `powf` | `TranscendentalMath` |
| `hypot`, `atan2`, distance-flavored things | `SpatialMath` |
| bit tricks on the float representation | `FloatVectorWithBits` |
| `erf`, `gamma`, activation functions | `SpecialMath` |

`SpecialMath` is the one entry above that isn't in core thermite. It comes from the
`thermite-special` companion crate, covered later, and it's listed here because a reader
hunting for `erf` needs to know it exists somewhere.


The math traits already require what they need, so they usually replace a bound rather
than add to one. `TranscendentalMath` implies `CoreMath`, which implies `FloatVector`,
which is why the `gaussian` above only had to name one trait. Do note the gotcha: the
prelude imports the math traits *anonymously*, so their methods resolve but the names
don't. The moment you want to write one in a bound, you need the explicit import:

```rust
use thermite::prelude::*;
use thermite::math::TranscendentalMath; // required to NAME it below

#[inline(always)]
fn decay<V: TranscendentalMath>(t: V, tau: V) -> V {
    (-(t / tau)).exp()
}
```

### Associated types and constants

A generic kernel can't name `f32`, so when you need the scalar type, the lane count, or
the result of a comparison, you ask the vector for them.

- `V::Element` is the scalar behind the lanes, for parameters the caller supplies as
  plain numbers.
- `V::Mask` is what a comparison returns. More on masks later.
- `V::LANES` is a `const usize`, and `V::lanes()` is the runtime form. They agree today, but
  prefer `lanes()` in loop bounds and address arithmetic. A scalable-vector backend, SVE or
  RISC-V V, can only report its width at runtime, and code written against `lanes()` carries
  over unchanged while code written against the constant does not.

```rust
use thermite::prelude::*;

// The caller passes an ordinary scalar; we broadcast it once, inside.
#[inline(always)]
fn clamp_above<V: FloatVector>(v: V, limit: V::Element) -> V {
    v.min(V::splat(limit))
}
```

That splat folds into a broadcast, so it costs nothing. If the operation is a multiply,
though, reach for `scale` rather than splatting and multiplying by hand. It carries the
same meaning and lowers better on some targets, GPUs in particular.

For constants, prefer what the trait already hands you (`V::ZERO`, `V::ONE`, `V::TWO`,
`V::HALF`, and the rest of `FloatConsts`) over building them at runtime. `V::splat(x)`
covers everything else.

### What not to do

The failure mode is writing code that only makes sense at one width. It compiles, it
passes tests on your machine, and it quietly does the wrong thing (or the slow thing) on
every other backend.

```rust,ignore
// Don't. This is a scalar loop wearing a vector costume, and it will be
// slower than not using SIMD at all. `v.sqrt()` does the whole thing
// in one instruction.
fn bad<V: FloatVector>(v: V) -> V {
    let mut out = V::ZERO;
    for i in 0..V::LANES {
        out = out.insertv(i, v.extractv(i).scalar_sqrt());
    }
    out
}

// Also don't. The moment someone instantiates this at 4 or 16 lanes,
// or with `Dual`, the special case silently stops applying.
fn also_bad<V: FloatVector>(v: V) -> V {
    if V::LANES == 8 { fast_path(v) } else { slow_path(v) }
}
```

Reading and writing individual lanes is the thing to avoid. It's not that `extract` is
forbidden. It's the correct tool at the boundary, and the tail-handling loop in the
introduction uses it. Inside a kernel, though, it means the work isn't actually vectorized.

Branching on `V::LANES` is a subtler version of the same mistake. If two widths genuinely
need different algorithms, that's what the policy system and `if const` capability gating
are for, and those stay correct when a composite type shows up.

Note the difference between that and choosing a fixed width deliberately, covered under
vector widths above. A structure that *is* eight wide should say so in its types. A generic
function that asks how wide it happens to be, and quietly does something else when the
answer changes, is the problem.

## Masks and Branchless Code

A comparison can't hand back a `bool`, because there is no single answer for eight lanes
at once. It hands back a `Mask` instead, holding one boolean per lane, and everything in
this section is about what you do with one.

```rust
use thermite::prelude::*;

#[inline(always)]
fn safe_div<V: FloatVector>(a: V, b: V, fallback: V) -> V {
    let bad = b.cmp_eq(V::ZERO);

    // Takes the true value first, the false value second.
    bad.select(fallback, a / b)
}
```

The thing to internalize is that `select` is not an `if`. Both `fallback` and `a / b` are
computed for every lane, and then one of them is discarded. That's usually exactly what
you want, since a branch that some lanes take and others don't cannot exist inside a
single register. It does mean a piecewise function costs the sum of its pieces rather than
the larger one, so it's always worth a moment checking whether some single formulation
covers both cases.

Masks combine with the ordinary bitwise operators, so a range test is an intersection:

```rust
use thermite::prelude::*;

#[inline(always)]
fn in_unit_interval<V: FloatVector>(v: V) -> V::Mask {
    v.cmp_ge(V::ZERO) & v.cmp_le(V::ONE)
}
```

On unsigned integer vectors, reach for `in_range` instead. It gets the same answer from a
single unsigned compare by using wrapping subtraction, rather than the two compares and an
AND written above.

### Masked variants, and when to prefer them

Most operations carry `_c`, `_m`, and `_z` twins. `_c` (conditional) leaves `self`
untouched where the mask is false, `_m` (merge) takes that lane from a `src` operand
instead, and `_z` (zero) writes zero there. The mask comes first, except in `_m` where
`src` precedes it. The generated documentation spells out the exact signature on every
method, so there's nothing to memorize.

Prefer them over a `select` with a constant arm. Writing `x.sqrt_z(m)` states the intent and
lets each backend pick its own best encoding, which is an AND on SSE and AVX2, a `bsl` on
NEON, and a k-mask on AVX-512 where the masking is free. Writing `m.select(x.sqrt(),
V::ZERO)` instead pins the expression to blend semantics, which costs real instructions on
SSE and AVX2 and leaves the AVX-512 encoding on the table. `zz` and `nz` are the standalone
forms of the same idea, zeroing where a mask is false and where it is true respectively.

### When to actually branch

`all()`, `any()`, and `none()` collapse a mask down to a real `bool`. Use them only when
the answer lets you skip work, never to emulate a per-lane `if`.

```rust
use thermite::prelude::*;
use thermite::math::TranscendentalMath;

#[inline(always)]
fn refine<V: TranscendentalMath>(x: V, threshold: V) -> V {
    let needs_work = x.cmp_gt(threshold);

    // The expensive path is skipped entirely for vectors that don't need it.
    if needs_work.any() { needs_work.select(x.ln(), x) } else { x }
}
```

That branch pays for itself only if it skips something substantial (a transcendental, a
memory walk, a second pass) and if it's predictable. The branchless version has no best case
and no worst case, so a branch only wins when the skip is large and the prediction is right
most of the time. On data that makes the outcome a coin flip, the misprediction costs more
than the work it avoided.


Masks answer a few more questions worth knowing about. `first_set` and `last_set` give the
index of the lowest or highest true lane, which turns a comparison into a vectorized
`memchr`. `count_set` counts the true lanes, and `count_set_many` counts across several
masks at once for less than the sum of its parts, because a population count cannot
observe lane order and the backend is free to merge the masks before reducing.

## The Math Library and Policies

Every math function comes in two forms. The bare name runs under `DefaultPolicy`, and the
`_p` twin lets you say what you actually want:

```rust
use thermite::prelude::*;
use thermite::math::policy::policies::Precision;

type V = Vector<f64>;

let x = V::splat(0.5);

let quick = x.exp();                 // DefaultPolicy
let careful = x.exp_p::<Precision>();  // same function, different tradeoff

assert!((quick.extract::<0>() - careful.extract::<0>()).abs() < 1e-12);
```

Policies are types rather than values, and their parameters are associated constants, so
every branch inside the function folds away at compile time. Choosing one costs nothing at
runtime, which means the choice belongs at the call site. An inner loop can run
`UltraPerformance` while a validation path ten lines later runs `Reference`.

### The presets

| Policy | What you're asking for |
|---|---|
| `UltraPerformance` | Speed above all. No NaN or overflow handling, and results outside the domain are undefined. |
| `HighPerformance` | Fast, still skipping special-case checks, but a reasonable answer for ordinary inputs. |
| `Performance` | Fast without giving up precision. The default on CPU targets. |
| `Precision` | Precision first. Often costs little on backends with FMA. |
| `Size` | Smallest code, no unrolling or hard-coded expansions. Pairs with `opt-level = "z"`. |
| `Reference` | A reference answer, sometimes via infinite sums. Very expensive, meant for validation. |

`DefaultPolicy` is not always `Performance`. It's `Size` on WASM and `GpuDefault` on SPIR-V,
which is the right call on both, and worth knowing before it surprises you.


### Composing policies

The presets are starting points. Modifiers wrap a policy and move one axis:

```rust
use thermite::prelude::*;
use thermite::math::policy::policies::{ExtraPrecision, HighPerformance};

type V = Vector<f32>;

// Stay fast, but bump the precision tier up one step.
let y = V::splat(2.0).exp_p::<ExtraPrecision<HighPerformance>>();
```

`LessPrecision` goes the other way, and the rest set a single parameter directly:
`CheckOverflow`, `UnrollLoops`, `AvoidBranching`, `MaxIterations`, `UseCompensation`, plus
the tier setters from `WorstPrecision` through `ReferencePrecision`. Underneath, a policy
is a `PolicyParameters`: overflow checking, loop unrolling, precision tier, branch
avoidance, an iteration cap for the functions that converge iteratively, compensated
arithmetic, and denormal behavior.

That last parameter earns more attention than it looks like it deserves. Denormals can
cost over 100x on some processors, so most policies flush them to zero at a small constant
cost, while the performance-oriented policies use the cheaper crush trick (`a - (a - x)`)
that is nearly free unless the value really is denormal, and expensive when it is.

### FMA, and the one letter that changes it

There are two multiply-add families, told apart by a trailing `e`:

- `mul_add` is always accurate. Given hardware FMA it's the single instruction. Without,
  it falls back to a vectorized compensated emulation that still delivers single-rounding
  accuracy and still runs in SIMD.
- `mul_adde` is estimating. Given hardware FMA it's the same single instruction. Without,
  it degrades to a plain multiply followed by an add.

The `e` therefore decides only what happens on hardware that lacks FMA. Reach for
`mul_add` when you're accumulating and the rounding matters, and `mul_adde` when you just
want the instruction wherever it exists. If you're unsure, take `mul_adde`, then gate the
accuracy-critical path on `V::HAS_TRUE_FMA` behind an `if const` if measurement says it
matters.

The one configuration where `mul_add` becomes a real trap is `disable_fast_fma` or
`strict_ieee754`, which swap the emulated fallback for scalar `libm::fma`. That result is
exact and very slow.

### One-off scalar math

Bare `f32` and `f64` are not vectors, but they still get the whole math library through
`ScalarMath`, under `scalar_`-prefixed names with the same `_p` twins:

```rust
use thermite::prelude::*;
use thermite::math::policy::policies::Precision;

let a = 0.5f32.scalar_exp();
let b = 0.5f64.scalar_ln_p::<Precision>();
```

Use these for setup, constants, and test oracles. Inside a kernel, a scalar call means the
other lanes are idle, which is Rule 5.

## Slices and Data Layout

The introduction walked a slice with `chunks_exact` and handled the leftovers by hand.
That works, and it's worth seeing once, but `SimdSlice` does the same job without the
bookkeeping. It's implemented for `[T]`, so it's available on anything that derefs to a
slice:

```rust
use thermite::backend::scalar::prelude::*;

let mut data: Vec<f32> = (0..10).map(|i| i as f32).collect();

// `head` and `tail` are the scalar edges, `chunks` is everything in between.
let (head, chunks, tail) = data.try_aligned_simd_iter_mut::<f32x4>();

for v in chunks {
    *v = *v * *v;
}
for x in head.iter_mut().chain(tail.iter_mut()) {
    *x = *x * *x;
}

assert_eq!(data[9], 81.0);
```

There are four iteration modes, in shared and mutable pairs, and picking between them is
mostly a question of what you know about the allocation.

`try_aligned_simd_iter` is the one to reach for by default. It never panics, it hands back
the leading and trailing scalar remainders explicitly, and the middle yields `&V` or
`&mut V` straight out of the slice, so each step is an aligned load with no per-iteration
bookkeeping.

`aligned_simd_iter` is the same thing when you control the allocation and the slice is
already exactly aligned with no remainder on either end. It panics rather than coping,
which is the point: it turns a layout assumption into a loud failure instead of a slow
path you never notice.

`unaligned_simd_iter` handles arbitrary slices, and yields vectors by value rather than by
reference, since an unaligned slice cannot be reinterpreted as a slice of vectors. Each
step is an unaligned load. Reach for it when the data isn't yours to align. Note that it
returns a pair rather than a triple, just the iterator and a trailing remainder, because
with no alignment to reach there is no leading remainder to hand back.

`streaming_simd_iter` is the aligned iterator with non-temporal loads and stores, for large
sequential data you will not read again. It keeps the traffic out of cache, so it pays off
on a write-once buffer and hurts on anything you're about to touch a second time. Each item
is a wrapper, so you call `.load()` for the non-temporal path or `.load_cached()` when you
change your mind.

Alignment is worth caring about in hot loops over large buffers and not worth thinking
about anywhere else. `V::align_slice` is the underlying split if you want it directly. One
detail worth knowing: for a vector whose byte size isn't its element size times its lane
count, which happens with a non-hardware width like a 3-lane vector, `align_slice` gives
up and returns the whole slice as unaligned rather than pretending.

### Every way in and out

Getting data into and out of a register is where most of the friction lives, and there are
more options than there look to be. They sort by what you already have.

| What you've got | Reach for |
|---|---|
| A slice, alignment unknown | `from_slice` / `copy_to_slice` |
| A slice you're iterating | the `SimdSlice` iterators above |
| An aligned pointer | `load` / `store` |
| Any pointer | `load_unaligned` / `store_unaligned` |
| A large write-once buffer, aligned | `load_streaming` / `store_streaming` |
| Interleaved (AoS) data | `load_deinterleaved` / `store_interleaved` |
| Scattered indices into a big slice | `gather` / `scatter` |
| Indices into a small table | `lookup` |
| Only some lanes are valid | `store_masked`, `load_m` / `load_z` |
| No memory at all | `splat`, `single`, `new`, `indexed` |

The safe, boring pair is `from_slice` and `copy_to_slice`. Both assert the slice is long
enough and then issue an unaligned access, so they work on anything:

```rust
use thermite::backend::scalar::prelude::*;

let src = [1.0f32, 2.0, 3.0, 4.0, 5.0];

let v = f32x4::from_slice(&src);   // takes the first 4, ignores the rest

let mut out = [0.0f32; 4];
(v * v).copy_to_slice(&mut out);

assert_eq!(out, [1.0, 4.0, 9.0, 16.0]);
```

Those assertions are the catch. One length check is nothing, but inside a hot loop the
checks break LLVM's scheduling around the surrounding SIMD, which is discussed in the
performance section and is worth taking seriously. In a loop, prefer the slice iterators,
which do the checking once.

`load` and `store` are the raw pointer forms, and **`load` is an aligned load**. This
catches people constantly. A lookup table living in an ordinary `Box` or `Vec` is not
aligned to a vector, so `load` on it faults, and it faults based on where the allocator
happened to put the data, which means it can pass every test you run. Use `load_unaligned`
or hold the data in something aligned.

`load_streaming` and `store_streaming` are the non-temporal forms, which move data without
disturbing cache. They win on a large buffer you write once and won't revisit, and they lose
on anything you're about to read again. **These require alignment as well**, which is easy
to forget given that the reason you reached for them was the size of the buffer rather than
its layout. The streaming slice iterators enforce it for you by panicking on a misaligned or
ragged slice, which is the friendlier way to find out.

So three families need an aligned pointer: the plain `load` and `store`, the masked accesses,
and the streaming pair. Only the `_unaligned` forms and the safe `from_slice` and
`copy_to_slice` are exempt. `align_of::<V>()` is the value they all mean, and it is
dependable, which is discussed under vector widths above.

### When the data isn't contiguous

`gather` builds a vector by reading one element per lane at an index you supply, and
`scatter` writes back the same way. Indices are counted in elements rather than bytes. The
safe `gather` bounds-checks every lane and panics if any is out of range, `gather_or` and
`gather_or_zero` substitute a value instead of panicking, and `gather_if` takes a mask so
disabled lanes never touch memory at all. The `_ptr` variants skip the checks and are
`unsafe`.

Reach for these when the access pattern is genuinely irregular, such as walking an index
buffer or sampling a texture. Do not reach for them to paper over a layout you control. A
gather costs many times a contiguous load even on hardware that implements it natively,
because it is still one memory access per lane underneath.

`lookup` is the one worth knowing about, because it is easy to miss and it is not the same
thing as `gather`:

```rust
use thermite::backend::scalar::prelude::*;

let table = [10.0f32, 20.0, 30.0, 40.0];
let v = f32x4::lookup(&table, u32x4::new([3, 1, 0, 2]));

assert_eq!(v.into_array(), [40.0, 20.0, 10.0, 30.0].into());
```

It is semantically a gather, but specialized for tables roughly the size of the vector
itself, where the backend can do the whole thing with shuffle instructions and never touch
memory. Hand it a table too large for that and it falls back to `gather` on its own. Out
of range indices yield the table's first element rather than panicking. If you're indexing
a small constant table, which covers most polynomial coefficient selection and most
classification work, this is the one you want.

### Partial vectors, and the scalar boundary

`store_masked` writes only the lanes where a mask is true and suppresses the rest. `load_m`
and `load_z` are the masked loads, merging from a source or zeroing the disabled lanes
respectively. All three are `unsafe` and take raw pointers, and all three want alignment.

It is tempting to reach for `store_masked` as a branchless way to finish a ragged tail, and
that is worth resisting for now. Pointing it past the end of an allocation and trusting the
mask to suppress the write is not something Rust's memory model currently blesses, whatever
the hardware does, and no backend implements a hardware masked store today, so it would not
buy you speed either. Handle tails the way the introduction does, with a scalar remainder
loop.

For building a vector without touching memory: `splat` broadcasts one value to every lane,
`single` puts a value in lane zero and zeros the rest, `new` takes an array, and `indexed`
gives you `[0, 1, 2, ...]`, which is the usual starting point for an index-carrying loop
(advance it by adding `offset()`).

Going the other way, `extract::<I>` pulls out one lane at a compile-time index and lowers
to a single instruction. `extractv` takes a runtime index and costs more. `insert` and
`insertv` are the reverse. These are the scalar boundary, and they belong at the edges of
your code. A loop over lanes calling `extract` is the anti-pattern from the generic kernels
section.

### Filtering, scanning, and divergence

Three primitives come up constantly once you're past simple element-wise work, and none of
them are obvious if you're coming from scalar code.

**Compaction.** `compress` left-packs the lanes a mask selects, so `[a, b, c, d]` with
`[true, false, true, false]` becomes `[a, c, b, d]`, selected lanes first. Combine that
with a masked store of the leading population count and you have stream compaction, which
is how you filter, strip, or minify without branching. `compress_z` zeroes the tail instead
of keeping the unselected lanes, and `compress_m` takes the tail from a second vector, which
is the accumulator step of a buffered compactor, paired with `align`, which slides a window
across two registers so the packed lanes can be carried forward by the running count.
`expand` is the exact inverse of `compress`, scattering a packed vector back out to the
positions a mask names, and it has the same `_z` and `_m` variants. On AVX-512 these lower
to `vpcompress`, elsewhere to a permute table or a portable partition.

**Scans.** `prefix_sum` gives a running total per lane, so `[1, 2, 3, 4]` becomes
`[1, 3, 6, 10]`, which is exactly how you turn per-lane counts into write offsets. There are
`prefix_min` and `prefix_max`, plus `reverse_prefix_sum`, `reverse_prefix_min`, and
`reverse_prefix_max` running the other way. Cost is `O(log2 LANES)` where the backend has a
native cross-lane `align`, and a sequential walk where it doesn't, chosen at compile time.
To scan only some lanes, neutralize the rest first with `v.zz(mask).prefix_sum()`.

**Divergence.** This is the one worth knowing about even if you never use it. A packet whose
lanes want different work is normally a disaster for SIMD. `group_by_value` turns it into a
short sequence of uniform sub-packets, yielding each distinct value once along with the mask
of lanes holding it, in order of first occurrence:

```rust,ignore
let mut groups = geom_ids.group_by_value(active);
while let Some((geom_id, lanes)) = groups.next_group() {
    shade(geom_id, lanes);
}
```

`count_conflicts` is the related primitive, giving each lane the number of earlier lanes
holding the same value. It's what makes a scatter with duplicate indices safe, since a
plain scatter silently drops all but one write.

### Changing element type

`cast` converts between element types while preserving the lane count, in any direction:
widening, narrowing, signed to unsigned, int to float. It follows the same semantics scalar
`as` would, including saturating float-to-int. `fast_cast` gives that up for fewer
instructions when you've already ruled out the problematic inputs, and `saturating_cast`
covers narrowing between integers of the same signedness.

Masks have their own conversion. A mask from an `f32x8` comparison and one from an `i32x8`
comparison are the same eight booleans but different types, so `mask.cast()` reinterprets
one as the other. That's what lets a float comparison drive a select on an integer vector of
the same width.

### Two layouts, and neither one is "the fast one"

The layout question matters more than the iteration question, and there are two answers.

**Structure of arrays** keeps each component in its own buffer, so `x` for eight objects is
one load and a kernel over `FloatVector` processes eight objects at a time. This is the
throughput layout, it's what most of this guide assumes, and it's the default because most
kernels are batched.

**Array of structs** puts `x, y, z, x, y, z` in memory, one object at a time, and a register
holds a single object with its components in the lanes. `LinAlg3Vector` and `LinAlg4Vector`
are built for exactly this: `dot3` and `dot4` take two vectors and return a **scalar**,
because they are horizontal reductions across the lanes of one object. `cross3`, `refract`,
`quat4_product`, `quat_to_mat3`, and the `mat3_*` and `mat4_*` families work the same way.
These ride hardware primitives the SoA layout cannot reach at all.

So this is the latency layout, for when you have one object and want the answer now: a
camera transform, a scene-graph node, one ray against many boxes. Pick SoA when you have a
batch and want throughput, AoS when you have one thing and want it fast. Neither is the
right answer in general, and the mistake is assuming the layout you learned first applies
everywhere.

The trap is that the two spell things similarly. A lane-wise SoA dot product over separate
`ax, ay, bx, by` vectors is not `dot3`, and calling `dot3` because the name matched will
hand you a scalar where you wanted a vector. Know which layout you're in.

When data arrives interleaved and you want SoA, `interleave` and `deinterleave` convert in
registers, with radix forms for wider groupings. Better still, `load_deinterleaved` and
`store_interleaved` fold the transpose into the memory access itself instead of making it a
separate pass, and they have `_arrays` and `_grouped` forms for wider or ragged component
counts. Prefer those over loading and then transposing.

`thermite-geometry` provides both layouts ready made, split into `soa` and `aos` modules,
covering vectors, points, rays, bounds, matrices, quaternions, tangents, and transforms.

## The 3D and 4D Primitives

`LinAlg3Vector` and `LinAlg4Vector` are the AoS side of the library, and they are the most
specialized code in it. Every method assumes one object per register with its components in
the lanes, and every one of them is tuned for that assumption. If you are writing an AoS
library, a scene graph, a transform stack, or anything that walks single objects, these are
not a convenience layer over the general operations. They are the reason that code can be
fast at all.

They are also the easiest part of Thermite to use slightly wrong, in ways that cost
performance without ever being incorrect enough to notice. The rest of this section is the
list of ways.

### The `3` suffix is not cosmetic

`dot3`, `min_element3`, `max_element3`, `sum_elements3`, and `prod_elements3` operate on the
first three lanes of a register that may hold four. They exist because the general versions
would have to account for the fourth lane, and accounting for it means zeroing it first.
The suffixed versions skip that entirely.

So the choice is not stylistic. Calling `sum_elements` on a 4-lane register holding a 3D
vector either gives you the wrong answer, because whatever is in `w` joins the sum, or costs
you a zeroing step to make it right. Calling `sum_elements3` does neither.

When you genuinely need a clean fourth lane, which homogeneous coordinates require, `zero4`
and `one4` set it to zero or one efficiently. That's the sanitize-once pattern: fix `w`
when the data enters, then use the suffixed operations throughout.

### Const parameters that change the arithmetic

Several methods take a const generic that picks between real alternatives, and the default
choice is yours to make rather than something the library can infer.

`cross3` and `quat4_vec3_product` take `DOP`, selecting the difference-of-products
formulation. It's more accurate, and it needs hardware FMA to be efficient. Set it `false`
for speed, `true` for accuracy or when you know FMA is present. Getting this backwards on a
machine without FMA is a silent slowdown, not an error.

The `mat3_*` and `mat4_*` families take `COLUMN_MAJOR`, and this one bites harder than it
looks. It selects storage convention, which is fine, but for `mat3_product` and
`mat4_product` a `false` value also flips the multiplication order to `rhs * lhs`. That is
the correct behavior for row-major matrices and it is exactly the kind of thing that
produces a transformed scene that looks almost right.

For `quat_to_mat3` and `quat_to_mat4` the same parameter is genuinely free, since only the
compile-time sign masks differ between producing the rotation's columns and its rows.

### Let the specialized versions do their job

`quat4_vec3_product` changes algorithm based on the architecture, using the double-cross
method where shuffles are cheap and falling back to two dot products and a cross product
where they aren't, because cross products need several permutes to do well in SIMD. A
hand-rolled quaternion rotation gets one of those two strategies and keeps it everywhere.

`quat_to_mat3` is trig-free, built from pairwise products of the quaternion components with
no `sin`, `cos`, or `sqrt` anywhere. If you are converting quaternions to matrices by any
route that involves trigonometry, this is strictly better.

The matrix inverses are worth reading before use. `mat3_inverse_inplace` **returns the
determinant** rather than the matrix, avoids a copy, and lets you apply your own tolerance.
An exactly zero determinant leaves the matrix untouched, and a near-zero one gives you a
finite but unreliable result, so inspecting the returned value is the whole point.
`mat3_inverse` is the convenience wrapper that returns `Option` and copies.

### Two things to know before you start

The lane count has to be three or four. That's what the traits are for, and it's why the
`x3A` and `x3` types from the vector widths chapter exist.

Where a method's output has a fourth lane, such as `mat3_transpose` or `quat_to_mat3`, that
lane is **unspecified**. Not zero, not preserved, unspecified. Read it and you're reading
whatever the implementation found convenient. If you need it clean, `zero4` or `one4` is
one instruction.

## Performance

### Rule 1, and why it is not a suggestion

Rule 1 says `#[thermite::dispatch]` and `#[inline(always)]` are mandatory. It's worth
seeing what actually happens when they're missing, because nothing else in this guide
costs as much and nothing else fails as quietly.

Here is one generic kernel, compiled two ways:

```rust,ignore
#[inline(always)]
fn dot2<V: FloatVector>(ax: V, ay: V, bx: V, by: V) -> V {
    ax.mul_adde(bx, ay * by)
}

#[thermite::dispatch(V)]
pub fn dot2_dispatched<V: FloatVector>(ax: V, ay: V, bx: V, by: V) -> V {
    dot2(ax, ay, bx, by)
}
```

With the dispatch boundary, at `f32x8` on x86-v3, the whole body is this:

```asm
vmovaps     ymm0, ymmword ptr [r8]
vmulps      ymm0, ymm0, ymmword ptr [rax]
vmovaps     ymm1, ymmword ptr [rdx]
vfmadd231ps ymm0, ymm1, ymmword ptr [r9]
vmovaps     ymmword ptr [rcx], ymm0
vzeroupper
ret
```

One multiply, one fused multiply-add, on 256-bit registers. That's the whole function.

Calling `dot2` with no `#[dispatch]` anywhere above it produces this instead:

```asm
call _RNvNtNtNtCs9N2lWLRSIT9_4core9core_arch3x863avx13__mm256_mul_ps
call _RNvNtNtNtCs9N2lWLRSIT9_4core9core_arch3x863fma15__mm256_fmadd_ps
```

surrounded by two dozen `movaps` on 128-bit `xmm` registers, shuffling arguments to and
from the stack. The intrinsics did not inline, so each one survives in the binary as a
function you call to execute a single instruction. There is no scheduling across them, no
register allocation across them, and no constant folding across them.

That's the tell to look for. Streams of `call` into `__mm256_*` stubs, `xmm` where you
expected `ymm`, and no `vfmadd` anywhere. It compiles, it passes every test, and it is
slower than not using SIMD at all.

The reason is that `core::arch` intrinsics are `#[target_feature]` functions, and rustc
will not inline one of those into a caller that doesn't enable those features.
`#[dispatch]` is what creates the enabled context, by emitting a `#[target_feature]`
trampoline per backend. `#[inline(always)]` is what carries the context down into helpers,
because features propagate into a callee only when it's actually inlined. Plain `#[inline]`
is a hint the optimizer routinely declines, and it declines most often in exactly the large
bodies where this matters.

None of that costs code size the way it sounds like it should. The trampoline holds the
target features itself, so the dispatched function is free to stay out of line as one copy
per backend. Inline aggressively inside the kernel and let the compiler size the boundary.
The one place not to put `#[dispatch]` is a tiny leaf helper, where a dispatch boundary
only blocks the inlining you wanted.

### Gate hardware paths with `if const`

Capability constants resolve at compile time inside the dispatched context, so an `if const`
on one costs nothing and lets a single generic body pick the right algorithm per backend:

```rust,ignore
if const { V::HAS_TRUE_FMA } {
    an = an.mul_add(quarter, lambda * quarter);  // fusing is safe, the hardware has it
} else {
    an = (an + lambda) * quarter;                // the extra multiply would be wasted above
}
```

`V::HAS_APPROX_RSQRT` is the other one worth knowing, because it differs by element type
rather than only by ISA. `f32` has a hardware reciprocal square root and `f64` does not,
where it becomes a square root followed by a division. So `a * rsqrt(b)` wins on `f32`
while `a / sqrt(b)` wins on `f64`, and a kernel that serves both should gate on it.

### Shorten the dependency chain, not the instruction count

On a superscalar core the limit is usually the latency of the longest dependency chain
rather than the number of instructions. LLVM cannot fix this for you, because reassociating
floating-point math changes results and it isn't allowed to.

Balanced trees beat serial chains at the same op count. `((x + y) + (z + w))` is two levels
deep where `(((x + y) + z) + w)` is three. Factoring can collapse a level outright:
`rx.mul_adde(ry + rz, ry * rz)` computes `rx*ry + rx*rz + ry*rz` two deep instead of three.
Hoisting a reciprocal out of a loop and multiplying by it turns a chain of divisions into
one, and division latency dwarfs multiplication.

Constants deserve the same attention. If a coefficient is a compile-time constant, fold the
negation into the constant and use `mul_adde` rather than `nmul_adde`, which saves an
operation. Check `FloatConsts` before computing anything, since `V::SQRT_EPSILON` and
friends already exist and cost nothing.

### Codegen traps that pass every test

`core::array::map` and `core::array::from_fn` fail to inline inside `target_feature` code
and fall back to scalar. Bare closures handed to `std` combinators often do the same. Both
put their bodies outside the feature context, which is the Rule 1 failure one level down.
Hand-roll the loop. `#[inline(always)]` helper functions are fine, it's the closures that
aren't.

Bounds checks are the other one. A `from_slice` or `copy_to_slice` check inside a hot loop
costs far more than the compare itself, because it breaks LLVM's block scheduling around
the surrounding SIMD. Hoist a single length assertion before the loop so the body schedules
as one block.

A rarer one, for when the codegen looks stranger than it should: LLVM occasionally tries to
vectorize the vectors, repacking code that is already SIMD into a wider form that ends up
slower. `block_autovectorization` is an optimizer barrier that stops it at the use site,
emitting no instructions of its own. Reach for it only after measuring a regression you can
trace to that, never preemptively.

### Look at what you got

Guessing is how the Rule 1 failure survives. Emitting assembly takes one command:

```text
cargo rustc --release -- --emit asm -C llvm-args=--x86-asm-syntax=intel
```

Then read the function you care about and check that the vector instructions you expected
are there, on the register width you expected, with no `call` into intrinsic stubs and no
spills to stack. The two listings at the top of this section came out of exactly that
command. For timing, use a real harness rather than eyeballing, and remember that a debug
build of SIMD code is slow by construction, so measure in release.

## Composite Types

The architecture section made the claim, so here is what it buys. Because the trait bounds
are the entire contract, a type that is not a hardware register at all can implement the
same hierarchy and flow through the functions you already wrote. Nothing in `dot2` or
`decay` from earlier needs to change.

`Dual` carries a value and its derivatives, so a kernel written for plain floats
differentiates itself:

```rust,ignore
use thermite_dual::AutoDiff;

// gaussian is the same function from the introduction. Untouched.
let r = gaussian.ad([V::splat(0.5)]);

let value = r.re.extract::<0>();      //  0.7788007830714049
let dydx  = r.dual[0].extract::<0>(); // -0.7788007830714049, which is -2x e^(-x^2)
```

`Compensated` is a double-double, tracking the rounding error of every operation in a second
limb. It buys **106 bits of significand against `f64`'s 53**, and it charges for them: an
addition is roughly eleven float operations, and a multiplication is two with hardware FMA
or around seventeen without, where it falls back to Dekker splitting.

```rust,ignore
// Summed naively in f64, 1e16 + 1.0 - 1e16 gives 0.0, because the 1.0
// falls off the bottom of the significand. Compensated gives 1.0.
let sum = values.iter().copied().fold(Compensated::<V>::ZERO, |a, x| a + x);
```

`Complex` pairs two real vectors and implements the same traits, so real-valued generic
code operates on complex data unchanged.

They nest. `Complex<Compensated<V>>` is complex arithmetic carried out in compensated
reals, and `Dual<Compensated<V>, N>` differentiates in double-double. The most practical
use of that last one is on hardware without `f64` at all, where `Compensated<f32>` recovers
most of double precision from single-precision units.

The honest status: all the companion crates are `publish = false` and pre-release. The
vector-trait surfaces are complete, but a handful of special functions still panic rather
than compute, mostly in the gamma family and the higher Bessel functions on the composite
types. Grep for `todo!` in the crate you depend on before relying on a specific function.

## The Companion Crates

`thermite-special` is the extended math library: the error-function family, gamma and
friends, activation functions such as gelu and swish, Lambert W, and elliptic integrals.
It follows the same policy system as the core math traits.

`thermite-geometry` provides geometric primitives in both layouts, split into `soa` and
`aos` modules, covering vectors, points, rays, bounds, matrices, quaternions, tangents, and
transforms, so you don't rebuild them per project.

`thermite-complex`, `thermite-dual`, and `thermite-compensated` are the composite types from
the previous section.

`thermite-sdf` is signed distance fields: primitives, boolean and smooth combinators,
transforms, and fractals, over the `SDF`, `GradientSdf`, and `BoundedSdf` traits.

`thermite-sort` is a SIMD sort. It beats the standard library's `sort_unstable` across the
input distributions it has been measured on, and it is dramatically faster on data that is
already ordered. Because it is written against the same trait bounds as everything else, it
sorts composite types too, so a double-double or complex sort comes free in their SoA forms.

`thermite-ffi` exposes the math library across a C ABI, with generated headers, for
callers that aren't Rust.

## Gotchas

Collected from the rest of the guide, plus the things that catch people who skipped it.

**Bare `f32` and `f64` are not vectors.** They don't implement `FloatVector`. Wrap with
`Vector::<f32>::splat(x)` or `x.as_vector()`, or use the `scalar_`-prefixed `ScalarMath`
methods for one-off work.

**Math trait names need an explicit import to be named.** The prelude brings them in
anonymously, so `x.exp()` resolves but `<V: TranscendentalMath>` does not compile until you
`use thermite::math::TranscendentalMath`.

**`>>` is a logical shift, even on signed vectors.** Use `srai`, `sra`, or `srav` when you
want the sign bit filled. A `(x << 31) >> 31` sign-mask idiom silently folds to zero
otherwise, and only assembly will tell you.

**`load` is an aligned load.** Loading a table straight out of a plain `Box` or `Vec` will
fault, nondeterministically, depending on where the allocator put it. Use `load_unaligned`
or an aligned container.

**`bitandnot` means different things at different layers.** At the `Vector` and `Mask`
layer, `a.bitandnot(b)` is `a & !b`. At the register layer it follows the x86 convention
and is `!lhs & rhs`. The vector implementations swap operands when they delegate, so never
infer one layer's behavior from the other.

**Masked variants take the mask first**, except `_m`, which takes `src` first and then the
mask. The generated documentation carries the exact signature for every method.

**`mul_add` is not the slow one by default.** Without hardware FMA it lowers to a
vectorized compensated emulation, not to `libm`. Only `disable_fast_fma` or
`strict_ieee754` make it genuinely expensive.

**`strict_ieee754` costs real performance.** It implies both `preserve_denormals` and
`disable_fast_fma`. Turn it on deliberately, knowing what you're buying.

**`PartialOrd` on vectors is feature-gated** behind `partial-ord`, and it's only meaningful
when every lane agrees on the ordering.

**`mix` takes the interpolant as `self`.** It's `t.mix(a, b)`, not `a.mix(b, t)`, which is
the reverse of how most APIs spell it. Both compile, and only one is right.

**`first_set` and `last_set` return `Option<usize>`**, not a bare index, because a mask with
no true lanes has no answer to give.

**`V::Native` describes the vector, not the machine.** A sub-native slot like `i16x2` is an
array of scalar lanes on every backend, so it reports `Scalar` even on an AVX2 host. That's
correct for that vector and wrong for any decision about the hardware's register budget. For
those, read the dispatched `S`.

**Debug builds of SIMD code are slow by construction.** Benchmark in release. Setting
`opt-level = 1` on the dev profile makes development bearable without a full release build.

## Where Everything Lives

Reference documentation for every trait and method is on docs.rs, and it is the right place
for "what does this method do". This guide is the tour, not the index.

The `#[dispatch]` and `dispatch_dyn!` macros carry their own documentation, including the
limitations on what can cross a `dispatch_dyn!` boundary. Worth reading once before you
build anything large.

Each companion crate carries its own README and module documentation. The issue tracker is
the honest record of what's finished and what isn't, which matters more than usual while
the companions remain pre-release.

