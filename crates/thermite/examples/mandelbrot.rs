//! The Mandelbrot set, one register of pixels at a time.
//!
//! The escape-time loop is the textbook case for masks. Lanes leave the set at
//! different iterations, so there is no scalar `break` to write: instead an
//! `active` mask tracks which lanes are still bounded, the iteration count is
//! incremented only through that mask, and the loop exits when
//! [`any`](GenericMask::any) reports every lane has escaped.
//!
//! Nothing below names an instruction set or a lane count. `kernel` is generic
//! over `S: FloatSimd<f32>` and [`dispatch_dyn!`](thermite::dispatch_dyn) picks
//! the widest backend the running CPU supports, so the same source is an AVX2
//! kernel on a modern x86 machine, an SSE2 one on an older machine, and NEON on
//! AArch64.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example mandelbrot
//! ```
//!
//! It writes `mandelbrot.png` (or the first CLI argument).
//!
//! # What to look at
//!
//! The two lines that matter for performance are `#[thermite::dispatch(S)]` on
//! `kernel` and `#[inline(always)]` on `escape_time`. Without the first, the body
//! compiles at the base ISA and every intrinsic becomes an out-of-line call.
//! Without the second, `escape_time` is compiled featureless for the same reason.
//! Neither mistake produces a warning or a wrong image, only a slow one.
//!
//! The masked increment is worth a second look too:
//!
//! ```text
//! counts = counts.add_c(active, V::ONE);
//! ```
//!
//! `add_c` adds where the mask is set and leaves the other lanes alone, which is
//! one AND plus one add. The equivalent `active.select(counts + one, counts)`
//! computes the same thing through a blend, and blends are the more expensive
//! instruction on most backends.
//!
//! Every operation has three such forms, and the mask is always the **first**
//! argument: `_c` keeps `self` where the mask is clear, `_z` zeroes those lanes,
//! and `_m` merges from a third operand. The shading step below uses `_z` to
//! black out the interior of the set without a branch or a second pass.

use thermite::prelude::*;
use thermite::simd::{FloatSimd, SizedSimd};

const WIDTH: usize = 1200;
const HEIGHT: usize = 800;
const MAX_ITER: u32 = 512;

/// Bounds of the complex plane, chosen to frame the whole set with a little room.
const CENTER_RE: f32 = -0.6;
const CENTER_IM: f32 = 0.0;
const SPAN_RE: f32 = 3.2;

/// Escape-time iteration for one register of points.
///
/// Returns the iteration count per lane as a float, with lanes that never escape
/// holding `MAX_ITER`. Points are checked against a radius of 2, squared to `4.0`
/// so no square root is needed.
///
/// `#[inline(always)]` is required, not stylistic: this is called from inside a
/// dispatched body, and target features only reach a callee that actually gets
/// inlined.
#[inline(always)]
fn escape_time<V: FloatVector<Element = f32>>(c_re: V, c_im: V) -> V {
    let bailout = V::splat(4.0);

    let mut z_re = V::ZERO;
    let mut z_im = V::ZERO;
    let mut counts = V::ZERO;

    for _ in 0..MAX_ITER {
        // |z|^2, then the mask of lanes still inside the escape radius.
        let re2 = z_re * z_re;
        let im2 = z_im * z_im;
        let active = (re2 + im2).cmp_le(bailout);

        // Every lane has escaped, so the remaining iterations cannot change the
        // result. Checking a whole register at once is what makes this cheap.
        if !active.any() {
            break;
        }

        // z = z^2 + c, computed for all lanes. Masking the arithmetic would cost
        // more than letting escaped lanes run: they are already excluded from
        // `counts`, and the values they produce are simply discarded.
        //
        // `im` first, because it reads the pre-update `z_re`.
        z_im = (z_re + z_re).mul_adde(z_im, c_im);
        z_re = (re2 - im2) + c_re;

        // The only masked operation in the loop. Escaped lanes stop counting.
        counts = counts.add_c(active, V::ONE);
    }

    counts
}

/// Renders the whole image into `out`, one row at a time.
///
/// `#[thermite::dispatch(S)]` emits a `#[target_feature]` trampoline per backend
/// plus a const-folded ISA match. It is the only reason the body gets per-ISA
/// codegen at all.
#[thermite::dispatch(S)]
pub fn kernel<S: FloatSimd<f32>>(out: &mut [u8], width: usize, height: usize) {
    type V<S> = Vector<<S as SizedSimd<f32, i32, u32>>::fxN>;

    let lanes = V::<S>::LANES;

    let scale = SPAN_RE / width as f32;
    let left = CENTER_RE - SPAN_RE * 0.5;
    let top = CENTER_IM + scale * height as f32 * 0.5;

    let inv_max = 1.0 / MAX_ITER as f32;

    for y in 0..height {
        let c_im = V::<S>::splat(top - y as f32 * scale);

        // `indexed()` is `[0, 1, 2, ..., LANES-1]` in the element type, so this
        // walks the row a register at a time. `offset()` is the matching stride,
        // a splat of LANES, which keeps the advance out of the inner loop.
        let mut x_ramp = V::<S>::indexed();
        let step = V::<S>::offset();

        let row = &mut out[y * width..(y + 1) * width];

        for x in (0..width).step_by(lanes) {
            let c_re = x_ramp.mul_adde(V::<S>::splat(scale), V::<S>::splat(left));

            let counts = escape_time(c_re, c_im);

            // Map iterations to a byte. A linear ramp is useless here: almost
            // every exterior point escapes within a handful of iterations, so it
            // would put nearly the whole plane at one end of the range. `sqrt`
            // spreads those low counts out, which is where the visible structure
            // lives.
            let t = (counts * V::<S>::splat(inv_max)).sqrt();

            // Lanes that never escaped hold exactly MAX_ITER and are the set
            // itself. `mul_z` scales to 0..255 and zeroes the masked-off lanes in
            // one operation, so the interior comes out black with no branch and
            // no second pass.
            let escaped = counts.cmp_lt(V::<S>::splat(MAX_ITER as f32));
            let shade = t.mul_z(escaped, V::<S>::splat(255.0));

            // The last register of a row can run past the end, so only the lanes
            // that correspond to real pixels get written.
            let valid = (width - x).min(lanes);
            for (lane, px) in shade.as_slice()[..valid].iter().zip(&mut row[x..]) {
                *px = *lane as u8;
            }

            x_ramp += step;
        }
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "mandelbrot.png".into());

    let mut pixels = vec![0u8; WIDTH * HEIGHT];

    // The ISA is chosen here, once, at runtime. A single binary carries every
    // backend and runs the best one available on the machine it lands on.
    let isa = thermite::InstructionSet::get();
    thermite::dispatch_dyn!(kernel(&mut pixels, WIDTH, HEIGHT));

    // The image is self-checking without a reference file. The center pixel is
    // `CENTER_RE + 0i`, which sits well inside the main cardioid and so never
    // escapes, while the top-left corner is far outside and leaves immediately.
    let center = pixels[(HEIGHT / 2) * WIDTH + WIDTH / 2];
    assert_eq!(center, 0, "the interior of the set should be black");
    assert!(pixels[0] > 0, "the corner should have escaped immediately");

    image::save_buffer(
        &path,
        &pixels,
        WIDTH as u32,
        HEIGHT as u32,
        image::ExtendedColorType::L8,
    )
    .expect("failed to write PNG");

    println!("{WIDTH}x{HEIGHT}, {MAX_ITER} iterations max, dispatched to {isa:?} -> {path}");
}
