//! Composing the per-register prefix scan into a whole-array prefix sum.
//!
//! `NumericVector::prefix_sum` scans the `LANES` elements of one register. Scanning an
//! array means carrying each chunk's running total into the next. The only part that
//! needs thought is extracting that carry, and there is a shortcut: after
//! `chunk.prefix_sum()` the **last lane already holds the chunk total**, so the carry
//! is a broadcast of that lane - no horizontal reduction required.
//!
//! `scanned.reverse().broadcast::<0>()` is how you say "broadcast the last lane"
//! without needing `LANES - 1` as a const-generic literal (which is not expressible on
//! stable). Two ops, both loop-invariant in shape. `broadcastv(LANES - 1)` is the
//! runtime-index alternative and may be one op on some backends, but it lowers to a
//! variable permute - measure before preferring it.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example prefix_sum_array
//! ```
//!
//! # Scaling further
//!
//! The chunk loop below is *serially dependent*: each chunk waits on the previous
//! chunk's carry, so it cannot pipeline and runs at roughly one chunk per carry
//! latency. That is fine until it stops being the bottleneck, which for a simple scan
//! is around memory bandwidth.
//!
//! For very large arrays the standard fix is two passes:
//!
//! 1. Split into blocks; compute each block's total independently (fully parallel,
//!    and `sum_elements` is the right tool there since only the total is wanted).
//! 2. Prefix-sum the per-block totals (a tiny array - the kernel below handles it).
//! 3. Re-scan each block, adding its block offset (fully parallel again).
//!
//! That trades one extra pass over memory for parallelism across blocks, and step 3
//! parallelises across threads as well as lanes.

use thermite::prelude::*;
use thermite::simd::{FloatSimd, SizedSimd};

/// In-place inclusive prefix sum over an `f32` slice, at the native width of whatever
/// ISA is selected at runtime.
///
/// `#[thermite::dispatch(S)]` is load-bearing, not decoration: without a dispatch
/// ancestor the body compiles at the base ISA and every intrinsic stays out-of-line.
#[thermite::dispatch(S)]
pub fn prefix_sum_f32<S: FloatSimd<f32>>(data: &mut [f32]) {
    type V<S> = Vector<<S as SizedSimd<f32, i32, u32>>::fxN>;

    // Splits into (unaligned head, aligned whole chunks, unaligned tail). A prefix sum
    // is order-dependent, so all three are visited front to back.
    let (head, chunks, tail) = data.try_aligned_simd_iter_mut::<V<S>>();

    // Head: below one full register, so scalar is all there is.
    let mut acc = 0.0f32;
    for x in head.iter_mut() {
        acc += *x;
        *x = acc;
    }

    let mut carry = V::<S>::splat(acc);
    for v in chunks {
        let scanned = v.prefix_sum() + carry;
        // The last lane is this chunk's running total; broadcast it for the next one.
        carry = scanned.reverse().broadcast::<0>();
        *v = scanned;
    }

    // Tail: every lane of `carry` holds the same running total, so lane 0 will do.
    let mut acc = carry.extract::<0>();
    for x in tail.iter_mut() {
        acc += *x;
        *x = acc;
    }
}

fn main() {
    // Awkward lengths and start offsets on purpose: shorter than one register, exactly
    // one register, several plus a partial, and misaligned starts so the head is
    // non-empty. Values stay small integers so the float sum is exact and the
    // comparison against the sequential oracle can be exact too - the vector scan
    // reassociates the additions, so it is NOT bit-identical to a sequential sum for
    // arbitrary float input.
    let mut checked = 0;
    for len in [0usize, 1, 3, 7, 8, 16, 17, 31, 64, 65, 127, 1000] {
        for offset in [0usize, 1, 3] {
            let mut buf = vec![0.0f32; len + offset];
            for (i, x) in buf.iter_mut().enumerate() {
                *x = (1 + (i % 5)) as f32;
            }
            let data = &mut buf[offset..];

            let mut want = data.to_vec();
            for i in 1..want.len() {
                want[i] += want[i - 1];
            }

            thermite::dispatch_dyn!(prefix_sum_f32(data));

            assert_eq!(data, &want[..], "len={len} offset={offset}");
            checked += 1;
        }
    }
    println!("{checked} length/offset combinations match the sequential oracle");

    let mut demo = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0];
    thermite::dispatch_dyn!(prefix_sum_f32(&mut demo));
    println!("{demo:?}");
    assert_eq!(demo, [3.0, 4.0, 8.0, 9.0, 14.0, 23.0, 25.0, 31.0, 36.0, 39.0]);
}
