//! Moller-Trumbore ray-triangle intersection with an instance transform,
//! against glam and nalgebra: 4096 rays are pulled into object space by one
//! affine matrix, then tested against 8 triangles keeping the nearest hit.
//!
//! glam and nalgebra get the usual scalar early-out formulation at the
//! default target level (SSE2 unless RUSTFLAGS raises it), plus SoA variants
//! of the branchless kernel using each library's 4-wide type: glam's `Vec4`
//! and nalgebra over simba's `WideF32x4` lanes. Thermite runs the same
//! branchless kernel over SoA batches: on x86 at x86-v2 (SSE4.2, 128-bit) and
//! x86-v3 (AVX2+FMA, 256-bit); on aarch64 (with the `neon` feature) on the
//! NEON backend at the native 128-bit width (`f32x4`) and a 256-bit width
//! (`f32x8`) that is `ArrayRegister`-doubled from two 128-bit registers (2x128
//! double-pumped, emulated). The kernels are cross-checked at startup. The
//! host must support the forced ISA (SSE4.2 everywhere and AVX2 for v3 on x86;
//! NEON on aarch64).
//!
//! ```text
//! cargo bench --bench raytri
//! ```

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use glam::{Mat4, Quat, Vec3, Vec3A, Vec4};
use nalgebra::{Matrix4, Point3, Vector3};
use simba::simd::{SimdPartialOrd, SimdSigned, SimdValue, WideF32x4};

#[cfg(target_arch = "aarch64")]
use thermite::backend::neon::Neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::x86_v2::X86V2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::x86_v3::X86V3;
use thermite::prelude::*;
use thermite::simd::Simd;

/// Rays processed per timed iteration.
const N: usize = 4096;
/// Triangles each ray is tested against.
const TRIS: usize = 8;

/// Determinant cutoff below which the ray is treated as parallel.
const EPS: f32 = 1e-7;
const T_MIN: f32 = 1e-4;

/// Deterministic xorshift, values in `[lo, hi)`.
struct Rng(u64);

impl Rng {
    fn next(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = (self.0 >> 40) as f32 / (1u32 << 24) as f32;
        lo + unit * (hi - lo)
    }
}

/// SoA rays for the Thermite kernels.
struct Rays {
    ox: Vec<f32>,
    oy: Vec<f32>,
    oz: Vec<f32>,
    dx: Vec<f32>,
    dy: Vec<f32>,
    dz: Vec<f32>,
}

/// AoS rays - origin and direction interleaved, the way a renderer actually
/// stores them (and the way glam/nalgebra consume them here).
///
/// `#[repr(C)]` over six contiguous `f32`, so a `&[Ray]` is an interleaved
/// `oxoyozdxdydz...` span and a batch of `LANES` rays is exactly a 6-stream
/// AoS -> SoA load. The SoA kernels above get their de-interleaving for free
/// (someone else paid for it, off the clock); this one pays for it inline via
/// [`GenericVector::load_deinterleaved`], which is the honest comparison against
/// the AoS scalar kernels.
#[derive(Clone, Copy)]
#[repr(C)]
struct Ray {
    o: [f32; 3],
    d: [f32; 3],
}

/// `v0` plus precomputed edges `v1 - v0` and `v2 - v0`.
#[derive(Clone, Copy)]
struct Tri {
    v0: [f32; 3],
    e1: [f32; 3],
    e2: [f32; 3],
}

struct GlamTri {
    v0: Vec3A,
    e1: Vec3A,
    e2: Vec3A,
}

struct NaTri {
    v0: Point3<f32>,
    e1: Vector3<f32>,
    e2: Vector3<f32>,
}

/// Rays packed 4-wide for the glam SoA kernel.
struct GlamSoaRays {
    ox: Vec<Vec4>,
    oy: Vec<Vec4>,
    oz: Vec<Vec4>,
    dx: Vec<Vec4>,
    dy: Vec<Vec4>,
    dz: Vec<Vec4>,
}

/// Rays packed 4-wide for the nalgebra SoA kernel (AoSoA, one `Vector3` of
/// simba lanes per 4 rays).
struct NaSoaRays {
    o: Vec<Vector3<WideF32x4>>,
    d: Vec<Vector3<WideF32x4>>,
}

struct NaWideTri {
    v0: Vector3<WideF32x4>,
    e1: Vector3<WideF32x4>,
    e2: Vector3<WideF32x4>,
}

struct Scene {
    rays: Rays,
    rays_aos: Vec<Ray>,
    glam_o: Vec<Vec3A>,
    glam_d: Vec<Vec3A>,
    na_o: Vec<Point3<f32>>,
    na_d: Vec<Vector3<f32>>,
    glam_soa: GlamSoaRays,
    na_soa: NaSoaRays,
    tris: Vec<Tri>,
    glam_tris: Vec<GlamTri>,
    na_tris: Vec<NaTri>,
    na_wide_tris: Vec<NaWideTri>,
    /// Column-major affine world-to-object matrix, shared by all kernels.
    m: [f32; 16],
    glam_m: Mat4,
    na_m: Matrix4<f32>,
}

fn pack4(xs: &[f32]) -> WideF32x4 {
    let mut v = WideF32x4::splat(0.0);
    for (l, &x) in xs.iter().enumerate() {
        v.replace(l, x);
    }
    v
}

fn make_scene() -> Scene {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);

    let mut rays = Rays {
        ox: Vec::with_capacity(N),
        oy: Vec::with_capacity(N),
        oz: Vec::with_capacity(N),
        dx: Vec::with_capacity(N),
        dy: Vec::with_capacity(N),
        dz: Vec::with_capacity(N),
    };

    // Origins behind the scene looking roughly +z, so some rays hit and some
    // miss.
    for _ in 0..N {
        rays.ox.push(rng.next(-2.0, 2.0));
        rays.oy.push(rng.next(-2.0, 2.0));
        rays.oz.push(rng.next(-6.0, -4.0));
        rays.dx.push(rng.next(-0.35, 0.35));
        rays.dy.push(rng.next(-0.35, 0.35));
        rays.dz.push(rng.next(0.8, 1.2));
    }

    let mut tris = Vec::with_capacity(TRIS);
    for _ in 0..TRIS {
        let c = [rng.next(-1.5, 1.5), rng.next(-1.5, 1.5), rng.next(0.5, 3.0)];
        let mut v = [[0.0f32; 3]; 3];
        for vert in v.iter_mut() {
            let o = [rng.next(-0.9, 0.9), rng.next(-0.9, 0.9), rng.next(-0.2, 0.2)];
            *vert = [c[0] + o[0], c[1] + o[1], c[2] + o[2]];
        }
        tris.push(Tri {
            v0: v[0],
            e1: [v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]],
            e2: [v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]],
        });
    }

    // Mild instance transform, so the hit/miss mix survives it.
    let glam_m = Mat4::from_scale_rotation_translation(
        Vec3::new(0.9, 1.1, 1.0),
        Quat::from_rotation_y(0.3) * Quat::from_rotation_x(-0.2),
        Vec3::new(0.2, -0.1, 0.3),
    );
    let m = glam_m.to_cols_array();
    let na_m = Matrix4::from_column_slice(&m);

    let rays_aos: Vec<Ray> = (0..N)
        .map(|i| Ray {
            o: [rays.ox[i], rays.oy[i], rays.oz[i]],
            d: [rays.dx[i], rays.dy[i], rays.dz[i]],
        })
        .collect();

    let glam_o: Vec<Vec3A> = (0..N).map(|i| Vec3A::new(rays.ox[i], rays.oy[i], rays.oz[i])).collect();
    let glam_d: Vec<Vec3A> = (0..N).map(|i| Vec3A::new(rays.dx[i], rays.dy[i], rays.dz[i])).collect();
    let na_o: Vec<Point3<f32>> = (0..N)
        .map(|i| Point3::new(rays.ox[i], rays.oy[i], rays.oz[i]))
        .collect();
    let na_d: Vec<Vector3<f32>> = (0..N)
        .map(|i| Vector3::new(rays.dx[i], rays.dy[i], rays.dz[i]))
        .collect();

    let glam_tris = tris
        .iter()
        .map(|t| GlamTri {
            v0: Vec3A::from_array(t.v0),
            e1: Vec3A::from_array(t.e1),
            e2: Vec3A::from_array(t.e2),
        })
        .collect();
    let na_tris = tris
        .iter()
        .map(|t| NaTri {
            v0: Point3::new(t.v0[0], t.v0[1], t.v0[2]),
            e1: Vector3::new(t.e1[0], t.e1[1], t.e1[2]),
            e2: Vector3::new(t.e2[0], t.e2[1], t.e2[2]),
        })
        .collect();

    let glam_soa = GlamSoaRays {
        ox: rays.ox.chunks_exact(4).map(Vec4::from_slice).collect(),
        oy: rays.oy.chunks_exact(4).map(Vec4::from_slice).collect(),
        oz: rays.oz.chunks_exact(4).map(Vec4::from_slice).collect(),
        dx: rays.dx.chunks_exact(4).map(Vec4::from_slice).collect(),
        dy: rays.dy.chunks_exact(4).map(Vec4::from_slice).collect(),
        dz: rays.dz.chunks_exact(4).map(Vec4::from_slice).collect(),
    };
    let na_soa = NaSoaRays {
        o: (0..N)
            .step_by(4)
            .map(|i| {
                Vector3::new(
                    pack4(&rays.ox[i..i + 4]),
                    pack4(&rays.oy[i..i + 4]),
                    pack4(&rays.oz[i..i + 4]),
                )
            })
            .collect(),
        d: (0..N)
            .step_by(4)
            .map(|i| {
                Vector3::new(
                    pack4(&rays.dx[i..i + 4]),
                    pack4(&rays.dy[i..i + 4]),
                    pack4(&rays.dz[i..i + 4]),
                )
            })
            .collect(),
    };
    let na_wide_tris = tris
        .iter()
        .map(|t| NaWideTri {
            v0: Vector3::new(
                WideF32x4::splat(t.v0[0]),
                WideF32x4::splat(t.v0[1]),
                WideF32x4::splat(t.v0[2]),
            ),
            e1: Vector3::new(
                WideF32x4::splat(t.e1[0]),
                WideF32x4::splat(t.e1[1]),
                WideF32x4::splat(t.e1[2]),
            ),
            e2: Vector3::new(
                WideF32x4::splat(t.e2[0]),
                WideF32x4::splat(t.e2[1]),
                WideF32x4::splat(t.e2[2]),
            ),
        })
        .collect();

    Scene {
        rays,
        rays_aos,
        glam_o,
        glam_d,
        na_o,
        na_d,
        glam_soa,
        na_soa,
        tris,
        glam_tris,
        na_tris,
        na_wide_tris,
        m,
        glam_m,
        na_m,
    }
}

fn ray_tri_glam(o: &[Vec3A], d: &[Vec3A], tris: &[GlamTri], m: &Mat4) -> f32 {
    let mut acc = 0.0f32;
    for (&o, &d) in o.iter().zip(d) {
        let o = m.transform_point3a(o);
        let d = m.transform_vector3a(d);
        let mut t_near = f32::INFINITY;
        for tri in tris {
            let p = d.cross(tri.e2);
            let det = tri.e1.dot(p);
            if det.abs() <= EPS {
                continue;
            }
            let inv = 1.0 / det;
            let s = o - tri.v0;
            let u = s.dot(p) * inv;
            if u < 0.0 || u > 1.0 {
                continue;
            }
            let q = s.cross(tri.e1);
            let v = d.dot(q) * inv;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let t = tri.e2.dot(q) * inv;
            if t > T_MIN {
                t_near = t_near.min(t);
            }
        }
        if t_near < f32::INFINITY {
            acc += t_near;
        }
    }
    acc
}

fn ray_tri_nalgebra(o: &[Point3<f32>], d: &[Vector3<f32>], tris: &[NaTri], m: &Matrix4<f32>) -> f32 {
    let mut acc = 0.0f32;
    for (o, d) in o.iter().zip(d) {
        let o = m.transform_point(o);
        let d = m.transform_vector(d);
        let mut t_near = f32::INFINITY;
        for tri in tris {
            let p = d.cross(&tri.e2);
            let det = tri.e1.dot(&p);
            if det.abs() <= EPS {
                continue;
            }
            let inv = 1.0 / det;
            let s = o - tri.v0;
            let u = s.dot(&p) * inv;
            if u < 0.0 || u > 1.0 {
                continue;
            }
            let q = s.cross(&tri.e1);
            let v = d.dot(&q) * inv;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let t = tri.e2.dot(&q) * inv;
            if t > T_MIN {
                t_near = t_near.min(t);
            }
        }
        if t_near < f32::INFINITY {
            acc += t_near;
        }
    }
    acc
}

/// Same branchless kernel as the Thermite one, but with `Vec4` as the 4-lane
/// register.
fn ray_tri_glam_soa(rays: &GlamSoaRays, tris: &[Tri], m: &[f32; 16]) -> f32 {
    fn dot(a: &[Vec4; 3], b: &[Vec4; 3]) -> Vec4 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn cross(a: &[Vec4; 3], b: &[Vec4; 3]) -> [Vec4; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    let m00 = Vec4::splat(m[0]);
    let m01 = Vec4::splat(m[4]);
    let m02 = Vec4::splat(m[8]);
    let m03 = Vec4::splat(m[12]);
    let m10 = Vec4::splat(m[1]);
    let m11 = Vec4::splat(m[5]);
    let m12 = Vec4::splat(m[9]);
    let m13 = Vec4::splat(m[13]);
    let m20 = Vec4::splat(m[2]);
    let m21 = Vec4::splat(m[6]);
    let m22 = Vec4::splat(m[10]);
    let m23 = Vec4::splat(m[14]);

    let eps = Vec4::splat(EPS);
    let t_min = Vec4::splat(T_MIN);
    let inf = Vec4::splat(f32::INFINITY);

    let mut acc = Vec4::ZERO;
    for i in 0..rays.ox.len() {
        let (ox, oy, oz) = (rays.ox[i], rays.oy[i], rays.oz[i]);
        let (dx, dy, dz) = (rays.dx[i], rays.dy[i], rays.dz[i]);
        let o = [
            ox * m00 + oy * m01 + oz * m02 + m03,
            ox * m10 + oy * m11 + oz * m12 + m13,
            ox * m20 + oy * m21 + oz * m22 + m23,
        ];
        let d = [
            dx * m00 + dy * m01 + dz * m02,
            dx * m10 + dy * m11 + dz * m12,
            dx * m20 + dy * m21 + dz * m22,
        ];

        let mut t_near = inf;
        for tri in tris {
            let v0 = [Vec4::splat(tri.v0[0]), Vec4::splat(tri.v0[1]), Vec4::splat(tri.v0[2])];
            let e1 = [Vec4::splat(tri.e1[0]), Vec4::splat(tri.e1[1]), Vec4::splat(tri.e1[2])];
            let e2 = [Vec4::splat(tri.e2[0]), Vec4::splat(tri.e2[1]), Vec4::splat(tri.e2[2])];

            let p = cross(&d, &e2);
            let det = dot(&e1, &p);
            let inv = Vec4::ONE / det;
            let s = [o[0] - v0[0], o[1] - v0[1], o[2] - v0[2]];
            let u = dot(&s, &p) * inv;
            let q = cross(&s, &e1);
            let v = dot(&d, &q) * inv;
            let t = dot(&e2, &q) * inv;

            let hit = det.abs().cmpgt(eps)
                & u.cmpge(Vec4::ZERO)
                & v.cmpge(Vec4::ZERO)
                & (u + v).cmple(Vec4::ONE)
                & t.cmpgt(t_min);
            t_near = t_near.min(Vec4::select(hit, t, inf));
        }

        acc += Vec4::select(t_near.cmplt(inf), t_near, Vec4::ZERO);
    }
    acc.element_sum()
}

/// nalgebra's SIMD story: the same `Vector3` code as the scalar version, with
/// simba's 4-wide lane type as the scalar.
fn ray_tri_nalgebra_soa(rays: &NaSoaRays, tris: &[NaWideTri], m: &[f32; 16]) -> f32 {
    let m00 = WideF32x4::splat(m[0]);
    let m01 = WideF32x4::splat(m[4]);
    let m02 = WideF32x4::splat(m[8]);
    let m03 = WideF32x4::splat(m[12]);
    let m10 = WideF32x4::splat(m[1]);
    let m11 = WideF32x4::splat(m[5]);
    let m12 = WideF32x4::splat(m[9]);
    let m13 = WideF32x4::splat(m[13]);
    let m20 = WideF32x4::splat(m[2]);
    let m21 = WideF32x4::splat(m[6]);
    let m22 = WideF32x4::splat(m[10]);
    let m23 = WideF32x4::splat(m[14]);

    let zero = WideF32x4::splat(0.0);
    let one = WideF32x4::splat(1.0);
    let eps = WideF32x4::splat(EPS);
    let t_min = WideF32x4::splat(T_MIN);
    let inf = WideF32x4::splat(f32::INFINITY);

    let mut acc = zero;
    for (o, d) in rays.o.iter().zip(&rays.d) {
        let o = Vector3::new(
            m00 * o.x + m01 * o.y + m02 * o.z + m03,
            m10 * o.x + m11 * o.y + m12 * o.z + m13,
            m20 * o.x + m21 * o.y + m22 * o.z + m23,
        );
        let d = Vector3::new(
            m00 * d.x + m01 * d.y + m02 * d.z,
            m10 * d.x + m11 * d.y + m12 * d.z,
            m20 * d.x + m21 * d.y + m22 * d.z,
        );

        let mut t_near = inf;
        for tri in tris {
            let p = d.cross(&tri.e2);
            let det = tri.e1.dot(&p);
            let inv = one / det;
            let s = o - tri.v0;
            let u = s.dot(&p) * inv;
            let q = s.cross(&tri.e1);
            let v = d.dot(&q) * inv;
            let t = tri.e2.dot(&q) * inv;

            let hit = det.simd_abs().simd_gt(eps)
                & u.simd_ge(zero)
                & v.simd_ge(zero)
                & (u + v).simd_le(one)
                & t.simd_gt(t_min);
            t_near = t_near.simd_min(t.select(hit, inf));
        }

        acc += t_near.select(t_near.simd_lt(inf), zero);
    }
    acc.extract(0) + acc.extract(1) + acc.extract(2) + acc.extract(3)
}

#[inline(always)]
fn dot3<V: FloatVector>(a: &[V; 3], b: &[V; 3]) -> V {
    a[0].mul_adde(b[0], a[1].mul_adde(b[1], a[2] * b[2]))
}

#[inline(always)]
fn cross3<V: FloatVector>(a: &[V; 3], b: &[V; 3]) -> [V; 3] {
    [
        a[2].nmul_adde(b[1], a[1] * b[2]),
        a[0].nmul_adde(b[2], a[2] * b[0]),
        a[1].nmul_adde(b[0], a[0] * b[1]),
    ]
}

/// The branchless Moller-Trumbore inner loop for one batch of rays already in
/// object space: every lane evaluates every triangle, masks pick the winners.
/// Shared verbatim by the SoA and AoS kernels so they differ ONLY in how the
/// batch was loaded.
#[inline(always)]
fn nearest_hit<V: FloatVector<Element = f32>>(o: &[V; 3], d: &[V; 3], tris: &[Tri], eps: V, t_min: V) -> V {
    let mut t_near = V::INFINITY;

    let mut k = 0;
    while k < tris.len() {
        let tri = &tris[k];
        let v0 = [V::splat(tri.v0[0]), V::splat(tri.v0[1]), V::splat(tri.v0[2])];
        let e1 = [V::splat(tri.e1[0]), V::splat(tri.e1[1]), V::splat(tri.e1[2])];
        let e2 = [V::splat(tri.e2[0]), V::splat(tri.e2[1]), V::splat(tri.e2[2])];

        let p = cross3(d, &e2);
        let det = dot3(&e1, &p);
        let inv = V::ONE / det;
        let s = [o[0] - v0[0], o[1] - v0[1], o[2] - v0[2]];
        let u = dot3(&s, &p) * inv;
        let q = cross3(&s, &e1);
        let v = dot3(d, &q) * inv;
        let t = dot3(&e2, &q) * inv;

        // No separate u > 1 test; u + v <= 1 with v >= 0 covers it.
        let hit =
            det.abs().cmp_gt(eps) & u.cmp_ge(V::ZERO) & v.cmp_ge(V::ZERO) & (u + v).cmp_le(V::ONE) & t.cmp_gt(t_min);
        t_near = t_near.min(hit.select(t, V::INFINITY));
        k += 1;
    }

    let found = t_near.cmp_lt(V::INFINITY);
    found.select(t_near, V::ZERO)
}

/// AoS variant: the rays arrive interleaved and are de-interleaved inline, in
/// one call per batch - the AoS -> SoA transpose that the SoA kernel gets handed
/// for free. Everything downstream is bit-identical to the SoA kernel.
///
/// The load is `load_deinterleaved_grouped::<2, 2>`, not a flat 6-stream
/// `load_deinterleaved::<6>`, and the distinction is worth real time. A `Ray` is
/// not six independent streams; it is TWO streams (origin, direction) of THREE
/// components each, and the grouped call says so. Backends with structural loads
/// then split it per chunk into `LD3`s - the transpose happens in the load unit -
/// while the flat view, whose stream count of 6 no `LDn` covers, falls back to a
/// register shuffle network. NEON `f32x4`: **12 instructions (two `LD3`) grouped,
/// versus ~25 flat**; double-pumped `f32x8`: 25 versus 44. On x86 (no structural
/// loads) both spell the same 6-stream shuffle network, so nothing is lost.
///
/// The general lesson: describe the layout you actually have, and let the backend
/// decide what it can do with it.
#[inline(always)]
fn ray_tri_kernel_aos<V: FloatVector<Element = f32>>(rays: &[Ray], tris: &[Tri], m: &[f32; 16]) -> f32 {
    let m00 = V::splat(m[0]);
    let m01 = V::splat(m[4]);
    let m02 = V::splat(m[8]);
    let m03 = V::splat(m[12]);
    let m10 = V::splat(m[1]);
    let m11 = V::splat(m[5]);
    let m12 = V::splat(m[9]);
    let m13 = V::splat(m[13]);
    let m20 = V::splat(m[2]);
    let m21 = V::splat(m[6]);
    let m22 = V::splat(m[10]);
    let m23 = V::splat(m[14]);

    let eps = V::splat(EPS);
    let t_min = V::splat(T_MIN);

    let mut acc = V::ZERO;
    let mut i = 0;
    while i + V::lanes() <= rays.len() {
        // `Ray` is `#[repr(C)]` over two 3-component vectors, so a batch of
        // `LANES` rays is 2 streams x 3 components: one grouped load, which is
        // two `LD3`s wherever the hardware has them.
        let [og, dg] = unsafe { V::load_deinterleaved_grouped::<2, 2>(rays.as_ptr().add(i) as *const f32) };

        let (ox, oy, oz) = (og.head, og.tail[0], og.tail[1]);
        let (dx, dy, dz) = (dg.head, dg.tail[0], dg.tail[1]);

        let o = [
            ox.mul_adde(m00, oy.mul_adde(m01, oz.mul_adde(m02, m03))),
            ox.mul_adde(m10, oy.mul_adde(m11, oz.mul_adde(m12, m13))),
            ox.mul_adde(m20, oy.mul_adde(m21, oz.mul_adde(m22, m23))),
        ];
        let d = [
            dx.mul_adde(m00, dy.mul_adde(m01, dz * m02)),
            dx.mul_adde(m10, dy.mul_adde(m11, dz * m12)),
            dx.mul_adde(m20, dy.mul_adde(m21, dz * m22)),
        ];

        acc += nearest_hit::<V>(&o, &d, tris, eps, t_min);
        i += V::lanes();
    }
    acc.sum_elements()
}

/// Branchless Moller-Trumbore: every lane evaluates every triangle, masks pick
/// the winners.
#[inline(always)]
fn ray_tri_kernel<V: FloatVector<Element = f32>>(rays: &Rays, tris: &[Tri], m: &[f32; 16]) -> f32 {
    // Rows of the column-major affine matrix, splatted across lanes.
    let m00 = V::splat(m[0]);
    let m01 = V::splat(m[4]);
    let m02 = V::splat(m[8]);
    let m03 = V::splat(m[12]);
    let m10 = V::splat(m[1]);
    let m11 = V::splat(m[5]);
    let m12 = V::splat(m[9]);
    let m13 = V::splat(m[13]);
    let m20 = V::splat(m[2]);
    let m21 = V::splat(m[6]);
    let m22 = V::splat(m[10]);
    let m23 = V::splat(m[14]);

    let eps = V::splat(EPS);
    let t_min = V::splat(T_MIN);

    let mut acc = V::ZERO;
    let mut i = 0;
    while i + V::lanes() <= rays.ox.len() {
        let (ox, oy, oz, dx, dy, dz) = unsafe {
            (
                V::load_unaligned(rays.ox.as_ptr().add(i)),
                V::load_unaligned(rays.oy.as_ptr().add(i)),
                V::load_unaligned(rays.oz.as_ptr().add(i)),
                V::load_unaligned(rays.dx.as_ptr().add(i)),
                V::load_unaligned(rays.dy.as_ptr().add(i)),
                V::load_unaligned(rays.dz.as_ptr().add(i)),
            )
        };

        // Origin transforms as a point, direction as a vector.
        let o = [
            ox.mul_adde(m00, oy.mul_adde(m01, oz.mul_adde(m02, m03))),
            ox.mul_adde(m10, oy.mul_adde(m11, oz.mul_adde(m12, m13))),
            ox.mul_adde(m20, oy.mul_adde(m21, oz.mul_adde(m22, m23))),
        ];
        let d = [
            dx.mul_adde(m00, dy.mul_adde(m01, dz * m02)),
            dx.mul_adde(m10, dy.mul_adde(m11, dz * m12)),
            dx.mul_adde(m20, dy.mul_adde(m21, dz * m22)),
        ];

        acc += nearest_hit::<V>(&o, &d, tris, eps, t_min);
        i += V::lanes();
    }
    acc.sum_elements()
}

macro_rules! thermite_kernel {
    ($mod:ident, $tf:literal, $V:ty) => {
        mod $mod {
            use super::*;

            #[target_feature(enable = $tf)]
            pub unsafe fn ray_tri(rays: &Rays, tris: &[Tri], m: &[f32; 16]) -> f32 {
                ray_tri_kernel::<$V>(rays, tris, m)
            }

            #[target_feature(enable = $tf)]
            pub unsafe fn ray_tri_aos(rays: &[Ray], tris: &[Tri], m: &[f32; 16]) -> f32 {
                ray_tri_kernel_aos::<$V>(rays, tris, m)
            }
        }
    };
}

// Native width per backend: 128-bit on v2, 256-bit on v3.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
thermite_kernel!(v2, "sse4.2", Vector<<X86V2 as Simd>::f32x4>);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
thermite_kernel!(v3, "avx2,fma", Vector<<X86V3 as Simd>::f32x8>);

// NEON: native 128-bit width (f32x4) plus a 256-bit width (f32x8) that is
// `ArrayRegister`-doubled from two 128-bit registers (2x128 double-pumped,
// emulated).
#[cfg(target_arch = "aarch64")]
thermite_kernel!(neon128, "neon", Vector<<Neon as Simd>::f32x4>);
#[cfg(target_arch = "aarch64")]
thermite_kernel!(neon256, "neon", Vector<<Neon as Simd>::f32x8>);

fn bench(c: &mut Criterion) {
    let scene = make_scene();

    // Everyone has to agree on the sum of nearest-hit distances, modulo FMA
    // rounding and the odd boundary hit flipping.
    let r_glam = ray_tri_glam(&scene.glam_o, &scene.glam_d, &scene.glam_tris, &scene.glam_m);
    let r_na = ray_tri_nalgebra(&scene.na_o, &scene.na_d, &scene.na_tris, &scene.na_m);
    let r_glam_soa = ray_tri_glam_soa(&scene.glam_soa, &scene.tris, &scene.m);
    let r_na_soa = ray_tri_nalgebra_soa(&scene.na_soa, &scene.na_wide_tris, &scene.m);
    assert!(r_glam > 0.0, "scene produced no hits");
    for (name, r) in [("nalgebra", r_na), ("glam-soa", r_glam_soa), ("nalgebra-soa", r_na_soa)] {
        let rel = (r - r_glam).abs() / r_glam;
        assert!(rel < 1e-2, "{name} disagrees with glam: {r} vs {r_glam}");
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let r_v2 = unsafe { v2::ray_tri(&scene.rays, &scene.tris, &scene.m) };
        let r_v3 = unsafe { v3::ray_tri(&scene.rays, &scene.tris, &scene.m) };
        // The AoS kernels must agree EXACTLY with their SoA twins: same math,
        // same order, only the load differs. Anything else is a de-interleave bug.
        let r_v2_aos = unsafe { v2::ray_tri_aos(&scene.rays_aos, &scene.tris, &scene.m) };
        let r_v3_aos = unsafe { v3::ray_tri_aos(&scene.rays_aos, &scene.tris, &scene.m) };
        assert_eq!(r_v2, r_v2_aos, "thermite-v2-aos disagrees with thermite-v2");
        assert_eq!(r_v3, r_v3_aos, "thermite-v3-aos disagrees with thermite-v3");

        for (name, r) in [("thermite-v2", r_v2), ("thermite-v3", r_v3)] {
            let rel = (r - r_glam).abs() / r_glam;
            assert!(rel < 1e-2, "{name} disagrees with glam: {r} vs {r_glam}");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let r_neon128 = unsafe { neon128::ray_tri(&scene.rays, &scene.tris, &scene.m) };
        let r_neon256 = unsafe { neon256::ray_tri(&scene.rays, &scene.tris, &scene.m) };

        let r_neon128_aos = unsafe { neon128::ray_tri_aos(&scene.rays_aos, &scene.tris, &scene.m) };
        let r_neon256_aos = unsafe { neon256::ray_tri_aos(&scene.rays_aos, &scene.tris, &scene.m) };
        assert_eq!(
            r_neon128, r_neon128_aos,
            "thermite-neon-128-aos disagrees with its SoA twin"
        );
        assert_eq!(
            r_neon256, r_neon256_aos,
            "thermite-neon-256-aos disagrees with its SoA twin"
        );

        for (name, r) in [("thermite-neon-128", r_neon128), ("thermite-neon-256", r_neon256)] {
            let rel = (r - r_glam).abs() / r_glam;
            assert!(rel < 1e-2, "{name} disagrees with glam: {r} vs {r_glam}");
        }
    }

    let mut g = c.benchmark_group("raytri/f32");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("glam", |b| {
        b.iter(|| {
            black_box(ray_tri_glam(
                black_box(&scene.glam_o),
                black_box(&scene.glam_d),
                &scene.glam_tris,
                &scene.glam_m,
            ))
        })
    });
    g.bench_function("nalgebra", |b| {
        b.iter(|| {
            black_box(ray_tri_nalgebra(
                black_box(&scene.na_o),
                black_box(&scene.na_d),
                &scene.na_tris,
                &scene.na_m,
            ))
        })
    });
    g.bench_function("glam-soa", |b| {
        b.iter(|| black_box(ray_tri_glam_soa(black_box(&scene.glam_soa), &scene.tris, &scene.m)))
    });
    g.bench_function("nalgebra-soa", |b| {
        b.iter(|| {
            black_box(ray_tri_nalgebra_soa(
                black_box(&scene.na_soa),
                &scene.na_wide_tris,
                &scene.m,
            ))
        })
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        g.bench_function("thermite-v2", |b| {
            b.iter(|| unsafe { black_box(v2::ray_tri(black_box(&scene.rays), &scene.tris, &scene.m)) })
        });
        g.bench_function("thermite-v3", |b| {
            b.iter(|| unsafe { black_box(v3::ray_tri(black_box(&scene.rays), &scene.tris, &scene.m)) })
        });
        // Same kernels, but fed AoS rays and de-interleaving inline - the cost
        // the SoA rows do not pay, and the layout glam/nalgebra actually use.
        g.bench_function("thermite-v2-aos", |b| {
            b.iter(|| unsafe { black_box(v2::ray_tri_aos(black_box(&scene.rays_aos), &scene.tris, &scene.m)) })
        });
        g.bench_function("thermite-v3-aos", |b| {
            b.iter(|| unsafe { black_box(v3::ray_tri_aos(black_box(&scene.rays_aos), &scene.tris, &scene.m)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    {
        g.bench_function("thermite-neon-128", |b| {
            b.iter(|| unsafe { black_box(neon128::ray_tri(black_box(&scene.rays), &scene.tris, &scene.m)) })
        });
        g.bench_function("thermite-neon-256", |b| {
            b.iter(|| unsafe { black_box(neon256::ray_tri(black_box(&scene.rays), &scene.tris, &scene.m)) })
        });
        // AoS rays, de-interleaved inline: on NEON this lowers to the structural
        // loads, where the transpose happens in the load unit.
        g.bench_function("thermite-neon-128-aos", |b| {
            b.iter(|| unsafe { black_box(neon128::ray_tri_aos(black_box(&scene.rays_aos), &scene.tris, &scene.m)) })
        });
        g.bench_function("thermite-neon-256-aos", |b| {
            b.iter(|| unsafe { black_box(neon256::ray_tri_aos(black_box(&scene.rays_aos), &scene.tris, &scene.m)) })
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
