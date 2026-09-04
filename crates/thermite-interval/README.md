thermite-interval
=================

Rigorous SIMD interval arithmetic for
[Thermite](https://github.com/raygon-renderer/thermite).

`Interval<V, W>` carries a closed interval `[lo, hi]` per lane, and every
operation returns an enclosure of the exact real result (containment). Because
`Interval` implements the same vector traits as every other Thermite composite,
a generic kernel written against `FloatVector` bounds computes verified
enclosures with no changes to the kernel.

The two policies are independent of each other:

- The **widening policy `W`** (`Fastest` / `Balanced` / `Tightest`) lives on
  the type and governs the interval bookkeeping (how each operation rounds
  outward). Mixing tiers is a type error, so convert explicitly with
  `with_widening`.
- Thermite's usual **math policy `P`** keeps its role on the math-library
  methods, tuning the approximation algorithms. A loose algorithm over
  `Tightest` intervals gives a cheap value that is still enclosed, because
  the enclosure absorbs the algorithm's error bound.

Neither policy is allowed to produce an interval that misses the true result.
Outward rounding never touches the hardware rounding mode. It widens after a
nearest-rounded op, by an ulp step or an eps scale, or only where an
error-free transform proves the rounding actually erred (the `Tightest` tier).

```rust
use thermite::prelude::*;
use thermite::math::FloatConsts;
use thermite_interval::{Interval, Tightest};

type V = Vector<f64>;
type I = Interval<V, Tightest>;

// [2, 3] * [10, 11] - 1/3, outward-rounded at every step.
let x = I::bounds(V::splat(2.0), V::splat(3.0));
let y = I::bounds(V::splat(10.0), V::splat(11.0));
let r = x * y - I::FRAC_1_3; // the constant is a true 1-ulp enclosure of 1/3
```

Each surface documents which of two rigor tiers it falls under:

- **Rigorous**: arithmetic, `sqrt`, `square`, `abs`, `min`/`max`, the set
  operations, and the `BoundedFloatConsts` constant enclosures. These are built
  only on correctly rounded primitives and error-free transforms.
- **High confidence, not yet certified**: the transcendental library. Endpoint
  values come from Thermite's kernels, widened by a per-policy margin for the
  algorithm error. Those margins are engineering estimates over the kernels'
  documented accuracies. I have not proven them.

With the `dual` feature, `verify` provides a vectorized Krawczyk operator over
`Dual<Interval<V, W>, 1>` for certified root existence and uniqueness.
