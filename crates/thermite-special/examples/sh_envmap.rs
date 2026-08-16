//! Projects an HDR environment map onto the spherical-harmonic basis and renders the
//! round trip, after "Introduction to Spherical Harmonics for Graphics Programmers"
//! (gpfault.net). It is that article's cubemap case study, on real HDR radiance
//! instead of an sRGB cubemap.
//!
//! The output is a pair of stacked cubemap crosses: the source environment on top, its
//! SH reconstruction below. Raising `--degree` walks the reconstruction from a single
//! flat ambient term up through recognisable lighting.
//!
//! ```text
//! cargo run -p thermite-special --example sh_envmap --release
//! cargo run -p thermite-special --example sh_envmap --release -- --degree 4
//! cargo run -p thermite-special --example sh_envmap --release -- --show-negative
//! cargo run -p thermite-special --example sh_envmap --release -- --degree 4 --dering
//! ```
//!
//! # What it demonstrates
//!
//! **Projection** is the inner product `$c_i = \int_{S^2} L(\omega) Y_i(\omega)\,d\omega$`,
//! evaluated as a Riemann sum over cubemap texels weighted by the solid angle each
//! subtends ([`solid_angle`], from Driscoll's derivation). **Reconstruction** is
//! `$L(\omega) \approx \sum_i c_i Y_i(\omega)$`. Both directions are the same SIMD
//! kernel ([`RealSpecialMath::spherical_harmonics`] evaluated over a register of
//! directions at a time), which is the point: one basis evaluation serves analysis and
//! synthesis alike.
//!
//! **Ringing** is on display too. SH is a truncated frequency-domain representation, so
//! a bright compact source (a sun, a window) makes the reconstruction overshoot and
//! undershoot around it, and radiance can go _negative_, which is physically
//! meaningless. Pass `--show-negative` to paint those texels magenta, and `--dering` to
//! apply the sinc window from Sloan's _Stupid Spherical Harmonics Tricks_, which tapers
//! the high bands toward zero and trades detail for non-negativity.
//!
//! # Conventions
//!
//! The environment is sampled in its own frame (equirectangular, `+y` up). The SH polar
//! axis is `+z`, which therefore lies on the horizon rather than at the zenith. That is
//! deliberate and harmless: the basis is orthonormal at any orientation, and projection
//! and reconstruction use the same one, so the round trip is unaffected.

// Indexed loops throughout: the kernels below run under `#[target_feature]`, where
// std combinators are prone to not inlining, and the scalar helpers index several
// arrays in step. Same allow the crate itself carries.
#![allow(clippy::needless_range_loop)]

use std::f32::consts::PI;

use thermite::prelude::*;
use thermite_special::{NO_PHASE, RealSpecialMath};

/// Face order is the usual cubemap one: +X, -X, +Y, -Y, +Z, -Z.
const FACES: usize = 6;

/// An RGB cubemap, `dim x dim` texels per face.
struct Cube {
    dim: usize,
    faces: Vec<Vec<[f32; 3]>>,
}

impl Cube {
    fn zeroed(dim: usize) -> Self {
        Self {
            dim,
            faces: (0..FACES).map(|_| vec![[0.0; 3]; dim * dim]).collect(),
        }
    }
}

/// Direction through the centre of texel `(col, row)` of `face`, normalised.
///
/// `u` and `v` run over `[-1, 1]` across the face. The per-face axis assignment is the
/// standard cubemap layout, so a cross assembled from these faces lines up the way a
/// reader expects.
fn face_direction(face: usize, u: f32, v: f32) -> [f32; 3] {
    let d = match face {
        0 => [1.0, -v, -u],  // +X
        1 => [-1.0, -v, u],  // -X
        2 => [u, 1.0, v],    // +Y
        3 => [u, -1.0, -v],  // -Y
        4 => [u, -v, 1.0],   // +Z
        _ => [-u, -v, -1.0], // -Z
    };

    let inv = 1.0 / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    [d[0] * inv, d[1] * inv, d[2] * inv]
}

/// Solid angle subtended by the texel centred at `(u, v)` on a `dim`-wide face.
///
/// Texels near a face's corners project onto much less of the sphere than texels near
/// its centre, so a uniform weight would bias the whole projection. Derivation:
/// <https://www.rorydriscoll.com/2012/01/15/cubemap-texel-solid-angle/>
fn solid_angle(u: f32, v: f32, dim: usize) -> f32 {
    fn area(a: f32, b: f32) -> f32 {
        (a * b).atan2((a * a + b * b + 1.0).sqrt())
    }

    let h = 1.0 / dim as f32; // half a texel, in the [-1, 1] face parameterisation
    let (u0, u1) = (u - h, u + h);
    let (v0, v1) = (v - h, v + h);

    area(u0, v0) - area(u0, v1) - area(u1, v0) + area(u1, v1)
}

/// Bilinear lookup into an equirectangular (lat-long) image, `+y` up.
fn sample_equirect(img: &[[f32; 3]], w: usize, h: usize, d: [f32; 3]) -> [f32; 3] {
    let u = 0.5 + d[0].atan2(-d[2]) / (2.0 * PI);
    let v = d[1].clamp(-1.0, 1.0).acos() / PI;

    let fx = u * w as f32 - 0.5;
    let fy = v * h as f32 - 0.5;

    let x0 = fx.floor();
    let y0 = fy.floor();
    let (tx, ty) = (fx - x0, fy - y0);

    // Longitude wraps, latitude clamps at the poles.
    let xi = |x: i64| x.rem_euclid(w as i64) as usize;
    let yi = |y: i64| y.clamp(0, h as i64 - 1) as usize;

    let (x0i, x1i) = (xi(x0 as i64), xi(x0 as i64 + 1));
    let (y0i, y1i) = (yi(y0 as i64), yi(y0 as i64 + 1));

    let mut out = [0.0; 3];
    for c in 0..3 {
        let a = img[y0i * w + x0i][c] * (1.0 - tx) + img[y0i * w + x1i][c] * tx;
        let b = img[y1i * w + x0i][c] * (1.0 - tx) + img[y1i * w + x1i][c] * tx;
        out[c] = a * (1.0 - ty) + b * ty;
    }
    out
}

/// Resamples an equirectangular environment into a cubemap.
fn build_cube(img: &[[f32; 3]], w: usize, h: usize, dim: usize) -> Cube {
    let mut cube = Cube::zeroed(dim);

    for face in 0..FACES {
        for row in 0..dim {
            let v = 2.0 * (row as f32 + 0.5) / dim as f32 - 1.0;
            for col in 0..dim {
                let u = 2.0 * (col as f32 + 0.5) / dim as f32 - 1.0;
                let d = face_direction(face, u, v);
                cube.faces[face][row * dim + col] = sample_equirect(img, w, h, d);
            }
        }
    }

    cube
}

/// Flattens a cube into per-texel `(direction, solid angle, radiance)`, which is all
/// either kernel needs and lets both walk one contiguous array.
fn cube_samples(cube: &Cube) -> (Vec<[f32; 3]>, Vec<f32>, Vec<[f32; 3]>) {
    let dim = cube.dim;
    let n = FACES * dim * dim;

    let mut dirs = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    let mut colors = Vec::with_capacity(n);

    for face in 0..FACES {
        for row in 0..dim {
            let v = 2.0 * (row as f32 + 0.5) / dim as f32 - 1.0;
            for col in 0..dim {
                let u = 2.0 * (col as f32 + 0.5) / dim as f32 - 1.0;
                dirs.push(face_direction(face, u, v));
                weights.push(solid_angle(u, v, dim));
                colors.push(cube.faces[face][row * dim + col]);
            }
        }
    }

    (dirs, weights, colors)
}

/// The projection, `$c_i = \sum_t L(\omega_t)\, Y_i(\omega_t)\, \Delta\omega_t$`, one
/// register of texels at a time.
///
/// The whole basis for a register of directions comes out of a single
/// `spherical_harmonics` call. The `3 * N` accumulators then stay in vector form until
/// the horizontal reduction at the very end, so the Riemann sum runs at lane width.
#[thermite::dispatch(S)]
fn project_kernel<S: FloatSimd<f32>, const L: usize, const N: usize>(
    dirs: &[[f32; 3]],
    weights: &[f32],
    colors: &[[f32; 3]],
) -> Vec<[f32; 3]> {
    type V<S> = Vector<<S as SizedSimd<f32, i32, u32>>::fxN>;

    let lanes = V::<S>::LANES;
    let mut acc = [[V::<S>::ZERO; N]; 3];

    let mut base = 0;
    while base < dirs.len() {
        let take = lanes.min(dirs.len() - base);

        // Scalar gather into lane arrays. A ragged tail is handled by leaving the
        // unused lanes at zero weight, which contributes nothing to the sums.
        let (mut lx, mut ly, mut lz) = ([0.0f32; 64], [0.0f32; 64], [0.0f32; 64]);
        let mut lw = [0.0f32; 64];
        let mut lc = [[0.0f32; 64]; 3];

        for k in 0..take {
            let d = dirs[base + k];
            lx[k] = d[0];
            ly[k] = d[1];
            lz[k] = d[2];
            lw[k] = weights[base + k];
            for c in 0..3 {
                lc[c][k] = colors[base + k][c];
            }
        }

        let x = V::<S>::from_slice(&lx[..lanes]);
        let y = V::<S>::from_slice(&ly[..lanes]);
        let z = V::<S>::from_slice(&lz[..lanes]);
        let w = V::<S>::from_slice(&lw[..lanes]);

        let mut basis = [V::<S>::ZERO; N];
        V::<S>::spherical_harmonics::<L, N, NO_PHASE>(x, y, z, &mut basis);

        for c in 0..3 {
            let radiance = V::<S>::from_slice(&lc[c][..lanes]) * w;
            for i in 0..N {
                acc[c][i] = basis[i].mul_adde(radiance, acc[c][i]);
            }
        }

        base += lanes;
    }

    (0..N)
        .map(|i| {
            [
                acc[0][i].sum_elements(),
                acc[1][i].sum_elements(),
                acc[2][i].sum_elements(),
            ]
        })
        .collect()
}

/// The reconstruction, `$L(\omega) \approx \sum_i c_i Y_i(\omega)$`, same basis kernel.
#[thermite::dispatch(S)]
fn reconstruct_kernel<S: FloatSimd<f32>, const L: usize, const N: usize>(
    dirs: &[[f32; 3]],
    coeffs: &[[f32; 3]],
) -> Vec<[f32; 3]> {
    type V<S> = Vector<<S as SizedSimd<f32, i32, u32>>::fxN>;

    let lanes = V::<S>::LANES;
    let mut out = Vec::with_capacity(dirs.len());

    let mut base = 0;
    while base < dirs.len() {
        let take = lanes.min(dirs.len() - base);

        let (mut lx, mut ly, mut lz) = ([0.0f32; 64], [0.0f32; 64], [0.0f32; 64]);
        for k in 0..take {
            let d = dirs[base + k];
            lx[k] = d[0];
            ly[k] = d[1];
            lz[k] = d[2];
        }

        let x = V::<S>::from_slice(&lx[..lanes]);
        let y = V::<S>::from_slice(&ly[..lanes]);
        let z = V::<S>::from_slice(&lz[..lanes]);

        let mut basis = [V::<S>::ZERO; N];
        V::<S>::spherical_harmonics::<L, N, NO_PHASE>(x, y, z, &mut basis);

        let mut rgb = [V::<S>::ZERO; 3];
        for c in 0..3 {
            for i in 0..N {
                rgb[c] = basis[i].mul_adde(V::<S>::splat(coeffs[i][c]), rgb[c]);
            }
        }

        let mut buf = [[0.0f32; 64]; 3];
        for c in 0..3 {
            rgb[c].copy_to_slice(&mut buf[c][..lanes]);
        }
        for k in 0..take {
            out.push([buf[0][k], buf[1][k], buf[2][k]]);
        }

        base += lanes;
    }

    out
}

/// Sloan's sinc window: scales band `l` by `$\left(\frac{\sin(\pi l / w)}{\pi l / w}\right)^n$`.
///
/// Truncating the series is a brick-wall low-pass, and brick walls ring. Tapering the
/// bands toward zero instead is a gentler filter, for the same reason a windowed FIR
/// beats a truncated one. It cannot _guarantee_ non-negativity, so `passes` exists to
/// apply it harder when one pass leaves negatives behind.
fn apply_window(coeffs: &mut [[f32; 3]], degree: usize, width: f32, passes: i32) {
    for l in 0..=degree {
        let scale = if l == 0 {
            1.0
        } else {
            let t = PI * l as f32 / width;
            (t.sin() / t).powi(passes)
        };

        for m in 0..(2 * l + 1) {
            let i = l * l + m;
            for c in 0..3 {
                coeffs[i][c] *= scale;
            }
        }
    }
}

/// Exposure, Reinhard tone map, sRGB transfer. Enough to put HDR radiance on screen.
fn tonemap(c: f32, exposure: f32) -> u8 {
    let x = (c * exposure).max(0.0);
    let m = x / (1.0 + x);
    let s = if m <= 0.003_130_8 {
        12.92 * m
    } else {
        1.055 * m.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Writes the source and reconstruction as two stacked cubemap crosses.
///
/// Cross layout, faces at `(col, row)` in a 4x3 grid:
///
/// ```text
///      +Y
///  -X  +Z  +X  -Z
///      -Y
/// ```
fn write_crosses(src: &Cube, sh: &Cube, path: &str, exposure: f32, show_negative: bool) -> std::io::Result<()> {
    // (face index, grid column, grid row)
    const SLOTS: [(usize, usize, usize); FACES] = [
        (2, 1, 0), // +Y
        (1, 0, 1), // -X
        (4, 1, 1), // +Z
        (0, 2, 1), // +X
        (5, 3, 1), // -Z
        (3, 1, 2), // -Y
    ];

    let dim = src.dim;
    let width = 4 * dim;
    let height = 6 * dim; // two crosses, three face-rows each
    let mut px = vec![0u8; width * height * 3];

    for (cube_index, cube) in [src, sh].into_iter().enumerate() {
        let y_base = cube_index * 3 * dim;

        for &(face, gc, gr) in &SLOTS {
            for row in 0..dim {
                for col in 0..dim {
                    let c = cube.faces[face][row * dim + col];
                    let negative = c[0] < 0.0 || c[1] < 0.0 || c[2] < 0.0;

                    let rgb = if show_negative && negative {
                        [255u8, 0, 255] // magenta: radiance the basis invented
                    } else {
                        [
                            tonemap(c[0], exposure),
                            tonemap(c[1], exposure),
                            tonemap(c[2], exposure),
                        ]
                    };

                    let x = gc * dim + col;
                    let y = y_base + gr * dim + row;
                    let o = (y * width + x) * 3;
                    px[o..o + 3].copy_from_slice(&rgb);
                }
            }
        }
    }

    image::save_buffer(path, &px, width as u32, height as u32, image::ColorType::Rgb8)
        .map_err(|e| std::io::Error::other(e.to_string()))
}

/// Runs the whole round trip at a degree known at compile time.
fn round_trip<const L: usize, const N: usize>(cube: &Cube, dering: Option<(f32, i32)>) -> (Vec<[f32; 3]>, Cube) {
    let (dirs, weights, colors) = cube_samples(cube);

    let mut coeffs = thermite::dispatch_dyn!(for<S> project_kernel::<S, L, N>(&dirs, &weights, &colors));

    if let Some((width, passes)) = dering {
        apply_window(&mut coeffs, L, width, passes);
    }

    let values = thermite::dispatch_dyn!(for<S> reconstruct_kernel::<S, L, N>(&dirs, &coeffs));

    let dim = cube.dim;
    let mut out = Cube::zeroed(dim);
    for face in 0..FACES {
        let start = face * dim * dim;
        out.faces[face].copy_from_slice(&values[start..start + dim * dim]);
    }

    (coeffs, out)
}

fn main() -> std::io::Result<()> {
    let mut hdri = "reference/autoshop_01_8k.hdr".to_string();
    let mut out = "renders/sh_envmap.png".to_string();
    let mut degree = 3usize;
    let mut dim = 128usize;
    let mut exposure = 1.0f32;
    let mut show_negative = false;
    let mut dering = false;
    let mut dering_width = 0.0f32;
    let mut dering_passes = 1i32;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next = || args.next().unwrap_or_default();
        match arg.as_str() {
            "--hdri" => hdri = next(),
            "-o" | "--out" => out = next(),
            "--degree" => degree = next().parse().unwrap_or(2),
            "--cube" => dim = next().parse().unwrap_or(128),
            "--exposure" => exposure = next().parse().unwrap_or(1.0),
            "--dering-width" => dering_width = next().parse().unwrap_or(0.0),
            "--dering-passes" => dering_passes = next().parse().unwrap_or(1),
            "--dering" => dering = true,
            "--show-negative" => show_negative = true,
            "-h" | "--help" => {
                println!(
                    "sh_envmap - project an HDR environment onto spherical harmonics\n\n\
                     --hdri PATH        equirectangular .hdr input (default {hdri})\n\
                     -o, --out PATH     output PNG (default {out})\n\
                     --degree L         maximum SH degree, 0..=5 (default 2)\n\
                     --cube N           cubemap face resolution (default 128)\n\
                     --exposure E       tone-map exposure (default 1.0)\n\
                     --dering           apply the sinc window\n\
                     --dering-width W   window width (default L + 1)\n\
                     --dering-passes N  window exponent (default 1)\n\
                     --show-negative    paint negative-radiance texels magenta"
                );
                return Ok(());
            }
            other => eprintln!("ignoring unknown argument {other:?}"),
        }
    }

    //let degree = degree.min(9);
    if dering_width <= 0.0 {
        dering_width = degree as f32 + 1.0;
    }
    let window = dering.then_some((dering_width, dering_passes));

    println!("loading {hdri}");
    let img = image::ImageReader::open(&hdri)
        .map_err(|e| std::io::Error::other(format!("{hdri}: {e}")))?
        .decode()
        .map_err(|e| std::io::Error::other(format!("{hdri}: {e}")))?
        .into_rgb32f();

    let (w, h) = (img.width() as usize, img.height() as usize);
    let texels: Vec<[f32; 3]> = img.pixels().map(|p| [p[0], p[1], p[2]]).collect();
    println!("  {w} x {h}");

    println!("building {dim}x{dim} cubemap");
    let cube = build_cube(&texels, w, h, dim);

    // The solid angles must sum to the area of the sphere. If they do not, the
    // projection is silently mis-weighted and every coefficient is wrong.
    let (_, weights, _) = cube_samples(&cube);
    let total: f64 = weights.iter().map(|&x| x as f64).sum();
    println!(
        "  solid angles sum to {total:.6} (4*pi = {:.6})",
        4.0 * std::f64::consts::PI
    );

    println!(
        "projecting to degree {degree} ({} coefficients)",
        (degree + 1) * (degree + 1)
    );
    let (coeffs, sh_cube) = match degree {
        0 => round_trip::<0, 1>(&cube, window),
        1 => round_trip::<1, 4>(&cube, window),
        2 => round_trip::<2, 9>(&cube, window),
        3 => round_trip::<3, 16>(&cube, window),
        4 => round_trip::<4, 25>(&cube, window),
        _ => round_trip::<5, 36>(&cube, window),
    };

    for (i, c) in coeffs.iter().enumerate() {
        let l = (i as f64).sqrt() as usize;
        let m = i as i64 - (l * (l + 1)) as i64;
        println!("  Y[{l:>2},{m:>3}]  {:>12.6} {:>12.6} {:>12.6}", c[0], c[1], c[2]);
    }

    let negatives = sh_cube
        .faces
        .iter()
        .flatten()
        .filter(|c| c[0] < 0.0 || c[1] < 0.0 || c[2] < 0.0)
        .count();
    let total_texels = FACES * dim * dim;
    println!(
        "negative-radiance texels: {negatives} / {total_texels} ({:.2}%)",
        100.0 * negatives as f64 / total_texels as f64
    );

    if let Some(parent) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_crosses(&cube, &sh_cube, &out, exposure, show_negative)?;
    println!("wrote {out}");

    Ok(())
}
