//! Correctness gate for the elliptic integrals, driven entirely through the public
//! entry points, `V::carlson_p` / `V::ellint_p` and their request structs. That is
//! also the dispatched path, so what runs here is what ships: the kernels underneath
//! (the AGM behind the complete first/second kinds, the five Carlson duplications, the
//! phi-range reduction) are reached the way a caller reaches them.
//!
//! Reference values come from A&S / Mathematica. Where no table exists (phi outside `[0, pi/2]`, the `D`
//! kind, and `R_J` with a negative parameter) the test builds its own oracle from an
//! identity or a numerical Cauchy principal value, so nothing here depends on the
//! implementation being right in two places at once.
//!
//! Every case runs on each backend the host arch provides. The algorithms are shared,
//! but the FMA/`rsqrt` capabilities under them are not, so an x86-only run leaves the
//! non-FMA and exact-`rsqrt` lowerings untested.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
#![allow(clippy::approx_constant)]
// Reference values are pasted from mpmath at full width. Trimming them to f64's
// shortest round-trip would only invite a transcription error.
#![allow(clippy::excessive_precision)]

fn close(got: f64, want: f64, tol: f64) -> bool {
    (got - want).abs() <= tol * want.abs().max(1.0)
}

// Jacobi zeta from mpmath at 30 digits, via the defining difference
// Z = E(phi,k) - E(k) F(phi,k)/K(k), which is a different computation from the Carlson
// form under test, so this is a real cross-check rather than a restatement.
// (phi, k, Z(phi, k))
const JACOBI_ZETA: &[(f64, f64, f64)] = &[
    (0.0, 0.5, 0.0),
    (0.3, 0.5, 0.036766998867488845911),
    (0.7, 0.5, 0.065615536956845927299),
    (1.0, 0.9, 0.2804533264624154805),
    (-0.7, 0.8, -0.1932025746760059664),
    (2.0, 0.6, -0.078633525006053302798),
    (3.0, 0.99, -0.097602115566158220906),
    (4.5, 0.3, 0.0095954090261178052864),
    (0.7, 0.0, 0.0),
    (1.2, 0.999, 0.55959701221741850373),
];

// Heuman's lambda from mpmath at 30 digits, via the defining three-term combination.
// Covers both arms: entries at |phi| <= pi/2 take the Carlson form, the rest the Legendre
// identity, and 1.5707963 sits just short of the endpoint where the R_J parameter is most
// delicate.
// (phi, k, Lambda_0(phi, k))
const HEUMAN_LAMBDA: &[(f64, f64, f64)] = &[
    (0.3, 0.5, 0.27619645372589324643),
    (0.7, 0.5, 0.6035962502087692917),
    (1.0, 0.9, 0.68787980092097212311),
    (1.5, 0.1, 0.9956640865531333951),
    (1.5707963, 0.7, 0.99999998990261178138),
    (1.8, 0.5, 1.0639612315238180472),
    (2.5, 0.3, 1.4150608561304022021),
    (3.0, 0.9, 1.8945758000922666049),
    (-1.0, 0.6, -0.77268315426127576343),
    (0.9, 0.99, 0.58180966333069250961),
];

// (k, K(k), E(k))
const KE: &[(f64, f64, f64)] = &[
    (0.0, 1.570796326794897, 1.570796326794897),
    (0.5, 1.685750354812596, 1.467462209339427),
    (0.7071067811865476, 1.854074677301372, 1.350643881047676),
    (0.8660254037844386, 2.156515647499643, 1.211056027568459),
    (0.9486832980505138, 2.578092113348173, 1.104774732704073),
];

// AGM spot values from mpmath at 30 digits. The last two straddle the range where the
// starting ratio, not the quadratic tail, sets the iteration count.
// (a, b, AGM(a, b))
const AGM: &[(f64, f64, f64)] = &[
    (1.0, 2.0, 1.4567910310469068692),
    (1.0, 1.4142135623730950488, 1.1981402347355922074),
    (24.0, 6.0, 13.458171481725615421),
    (3.0, 3.0, 3.0),
    (1.0, 1.0e-8, 0.079305210334345310798),
    (1.0e-30, 1.0, 0.022292230559453832048),
    (0.5, 1.0e12, 52870140222.89687512),
];

// Jacobi elliptic functions from mpmath at 30 digits. Covers k = 0 (the trigonometric
// degeneracy), k approaching 1, both signs of u, and the two points where a member of the
// triple is exactly zero: u = K (cn = 0) and u = 2K (sn = 0).
// (u, k, sn, cn, dn)
const JACOBI: &[(f64, f64, f64, f64, f64)] = &[
    (0.0, 0.5, 0.0, 1.0, 1.0),
    (0.7, 0.0, 0.64421768723769101971, 0.76484218728448845486, 1.0),
    (
        0.7,
        0.5,
        0.63429327633511237202,
        0.77309251684133431103,
        0.94837651273058064585,
    ),
    (
        -0.7,
        0.5,
        -0.63429327633511237202,
        0.77309251684133431103,
        0.94837651273058064585,
    ),
    (
        1.9,
        0.9,
        0.98573306130364143008,
        0.16831616634462503784,
        0.46146242404001938958,
    ),
    (
        3.366,
        0.5,
        0.0055006749505070420888,
        -0.99998487117310373055,
        0.99999621781473364535,
    ),
    (
        -5.1,
        0.75,
        0.90729112913363234275,
        -0.42050304041162232807,
        0.73277747572842527664,
    ),
    (
        0.3,
        0.1,
        0.29547798521037720328,
        0.95534954872864000884,
        0.99956336847773696364,
    ),
    (
        6.0,
        0.99,
        0.61351195068720361726,
        -0.78968543507144807728,
        0.79441386250828918745,
    ),
    (
        -8.0,
        0.8,
        -0.018787076551251492614,
        0.99982350730249257144,
        0.99988704826244286462,
    ),
    (
        2.0,
        0.999,
        0.96443754803169331087,
        0.26431083206447528786,
        0.26780508840374556204,
    ),
    (
        1.0,
        0.9999999,
        0.76159417303675454936,
        0.64805425359028542507,
        0.64805434309291762445,
    ),
    (
        1.685750354812596,
        0.5,
        1.0,
        -8.2623790858790638229e-18,
        0.86602540378443864676,
    ),
    (3.501507605831505, 0.6, -8.6427771956862322199e-17, -1.0, 1.0),
];

// Incomplete F and E from the reference value tables.
// Note |k| > 1 cases: valid as long as 1 - k^2 sin^2(phi) >= 0; handled naturally.
// (phi, k, F(phi,k), E(phi,k))
const INC: &[(f64, f64, f64, f64)] = &[
    (
        0.3430906586047127,
        2.712952582080266,
        0.4340870330108736,
        0.2852345328295404,
    ),
    (
        1.302990057703935,
        0.1279518954120547,
        1.307312511398114,
        1.298690225567921,
    ),
    (
        0.6523628380743488,
        -1.429437513650137,
        0.8005154258533936,
        0.5508100202571943,
    ),
    (
        0.4046022501376546,
        -1.981659235625333,
        0.4656721451084328,
        0.3575401358115371,
    ),
    (
        0.630370432896175,
        0.8641142168759754,
        0.6632598061016007,
        0.6003112504412838,
    ),
];

// Complete third kind Pi(n, k) from `elliptic_pim_values` (param m = k^2, so k = sqrt(m)).
// Includes n < 0 cases: Boost shifts those via A&S 17.7.17 to dodge cancellation, whereas we
// feed p = 1 - n (> 1, positive) straight into R_J, so this checks that the direct path is still
// accurate. (n, m, Pi(n,k)), and the n = 0 row must equal K(sqrt(m)).
//
// The (0.5, 0.50) row is n == m == k^2, which makes the R_J parameter p = 1 - n coincide with
// the argument y = 1 - k^2, the degenerate point where a `carlson_rc` without its
// small-|t| series loses ~7 digits. Kept here as a guard on full precision.
const PIC: &[(f64, f64, f64)] = &[
    (-10.0, 0.25, 0.4892245275965397),
    (-3.0, 0.50, 0.8760028274011437),
    (-1.0, 0.75, 1.440034318657551),
    (0.0, 0.25, 1.685750354812596), // == K(0.5); n = 0 zeroes the R_J term
    (0.5, 0.50, 2.701287762095351), // n == k^2: R_J p == arg degeneracy (regression guard)
    (0.5, 0.95, 4.633308147279891),
];

// Incomplete third kind Pi(phi, n, k) from the reference value tables.
// All entries have p = 1 - n sin^2(phi) > 0. (phi, n, k, Pi)
const PI3: &[(f64, f64, f64, f64)] = &[
    (
        1.087095515757691,
        0.157358332363011,
        0.8160487832898813,
        1.31594514075427,
    ),
    (
        0.7128175949111615,
        1.926593468907062,
        0.2994546721661018,
        1.25394623148424,
    ),
    (
        0.630370432896175,
        1.465981775919188,
        1.008702896970963,
        0.8737159913132074,
    ),
    (
        0.9695030752034163,
        -0.4072847419780592,
        -0.6962608926846425,
        0.9442477901112342,
    ),
];

// (phi, k) walking in toward the domain boundary |k sin(phi)| = 1, with F and E at 40 dps.
// phi_max is asin(1/|k|), or pi/2 when |k| <= 1; `gap` is the relative step back from it.
// The |k| <= 1 rows are the regression guard for computing 1 - k^2 sin^2(phi) as
// (1 - k^2) + k^2 cos^2(phi): the direct form loses everything here (1.8e-2 at the
// boundary) because k*sin(phi) rounds to exactly 1 before the subtraction.
// (k, phi, F, E, tol)
const NEAR_BOUNDARY: &[(f64, f64, f64, f64, f64)] = &[
    (1.0, 1.5707963267948966, 38.02500337382921, 1.0, 1.0e-12),
    (1.0, 1.5707963267933258, 27.872580217643021, 1.0, 1.0e-12),
    (1.0, 1.5707947559985698, 14.057075033199597, 0.9999999999987663, 1.0e-12),
    (
        1.0,
        1.5692255304681018,
        7.1493195486358628,
        0.99999876629970353,
        1.0e-12,
    ),
    (0.5, 1.5707963267948966, 1.685750354812596, 1.4674622093394271, 1.0e-12),
    (0.5, 1.5692255304681018, 1.6839365556969933, 1.4661018596297779, 1.0e-12),
    // |k| > 1 keeps a genuine cancellation (both addends have magnitude k^2 and opposite
    // signs), so these carry a pole-adjacent tolerance rather than a full-precision one.
    (1.5, 0.7297276562269663, 1.206444990100978, 0.55909966061115069, 1.0e-7),
    (1.5, 0.7297269264993101, 1.2053024662251757, 0.55909965998972036, 1.0e-7),
    (
        2.712952582080266,
        0.3775047783488644,
        0.60032703487807298,
        0.2946859614688079,
        1.0e-7,
    ),
    (
        2.712952582080266,
        0.3775044008440861,
        0.59977988434550101,
        0.29468596112153287,
        1.0e-7,
    ),
];

// Independent Cauchy-PV oracle for R_J(x,y,z,p) with p < 0 (pole at t0 = -p):
//   R_J = 1.5 [ int_0^{2 t0} (G(t)-G(t0))/(t-t0) dt + int_{2 t0}^inf G(t)/(t-t0) dt ],
//   G(t) = [(t+x)(t+y)(t+z)]^{-1/2}.   (PV int_0^{2 t0} 1/(t-t0) dt = 0, so both are regular.)
fn rj_pv_reference(x: f64, y: f64, z: f64, p: f64) -> f64 {
    assert!(p < 0.0);
    let t0 = -p;
    let g = |t: f64| ((t + x) * (t + y) * (t + z)).powf(-0.5);
    let gp = |t: f64| g(t) * (-0.5) * (1.0 / (t + x) + 1.0 / (t + y) + 1.0 / (t + z)); // G'(t)

    // Composite Simpson (even n) on [0, 2 t0] of (G(t)-G(t0))/(t-t0); at t==t0 use lim = G'(t0).
    let n = 200_000usize;
    let b = 2.0 * t0;
    let h = b / n as f64;
    let f1 = |t: f64| {
        let d = t - t0;
        if d.abs() < 1e-12 { gp(t0) } else { (g(t) - g(t0)) / d }
    };
    let mut s1 = f1(0.0) + f1(b);
    for i in 1..n {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        s1 += w * f1(i as f64 * h);
    }
    s1 *= h / 3.0;

    // Tail [2 t0, inf): t = 2 t0 + u/(1-u), u in [0,1), dt = du/(1-u)^2; integrand -> 0 at u=1.
    let hu = 1.0 / n as f64;
    let f2 = |u: f64| {
        let om = 1.0 - u;
        let t = 2.0 * t0 + u / om;
        g(t) / (t - t0) / (om * om)
    };
    let mut s2 = f2(0.0); // u=1 endpoint contributes 0
    for i in 1..n {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        s2 += w * f2(i as f64 * hu);
    }
    s2 *= hu / 3.0;

    1.5 * (s1 + s2)
}

/// Stamps the whole suite out for one backend, at that backend's native f32/f64 widths.
macro_rules! elliptic_tests {
    ($modname:ident, $backend:ty, $f32reg:ident, $f64reg:ident) => {
        mod $modname {
            use super::*;

            use thermite::Vector;
            use thermite::math::policy::policies::{CheckOverflow, Precision};
            use thermite::prelude::*;
            use thermite_special::elliptic::{
                CarlsonKind, CarlsonRc, CarlsonRd, CarlsonRf, CarlsonRg, CarlsonRj, EllintD, EllintDInc, EllintE,
                EllintEInc, EllintF, EllintK, EllintPi, EllintPiInc, EllipticConsts, EllipticKind, HeumanLambda,
                JacobiZeta,
            };
            use thermite_special::{RealSpecialMathWithPolicy, SpecialMathWithPolicy};

            type F32 = Vector<<$backend as Simd>::$f32reg>;
            type F64 = Vector<<$backend as Simd>::$f64reg>;

            /// Splat a scalar to every lane. All the checks read lane 0 back, and the rest
            /// of the vector is along for the ride, which is what makes a splat enough.
            fn v(x: f64) -> F64 {
                F64::splat(x)
            }

            /// A Legendre integral through the public dispatched entry, at lane 0.
            fn ell<K: EllipticKind<Output = F64>>(kind: K) -> f64 {
                F64::ellint_p::<Precision, K>(kind).extract::<0>()
            }

            /// A Carlson symmetric integral through the public dispatched entry, at lane 0.
            fn car<K: CarlsonKind<Output = F64>>(kind: K) -> f64 {
                F64::carlson_p::<Precision, K>(kind).extract::<0>()
            }

            // K and E complete, which is the AGM path.
            #[test]
            fn ellint_complete_ke_matches_reference() {
                for &(k, want_k, want_e) in KE {
                    let gk = ell(EllintK { k: v(k) });
                    let ge = ell(EllintE { k: v(k) });
                    assert!(close(gk, want_k, 1.0e-13), "K({k}): got {gk}, want {want_k}");
                    assert!(close(ge, want_e, 1.0e-13), "E({k}): got {ge}, want {want_e}");
                }
            }

            #[test]
            fn jacobi_zeta_matches_the_defining_difference() {
                for &(phi, k, want) in JACOBI_ZETA {
                    let got = ell(JacobiZeta { phi: v(phi), k: v(k) });
                    assert!(close(got, want, 1.0e-13), "Z({phi}, {k}): got {got}, want {want}");
                }
            }

            // Structural properties, independent of the table: odd in phi, pi-periodic, and
            // exactly zero at every multiple of pi/2: the last is where the defining
            // difference cancels worst and the Carlson form should not care.
            #[test]
            fn jacobi_zeta_symmetries() {
                for &(phi, k) in &[(0.7f64, 0.5f64), (1.3, 0.9), (2.2, 0.3)] {
                    let z = ell(JacobiZeta { phi: v(phi), k: v(k) });
                    let neg = ell(JacobiZeta {
                        phi: v(-phi),
                        k: v(k),
                    });
                    assert_eq!(neg, -z, "Z odd in phi at ({phi}, {k})");

                    let shifted = ell(JacobiZeta {
                        phi: v(phi + core::f64::consts::PI),
                        k: v(k),
                    });
                    assert!(close(shifted, z, 1.0e-12), "Z pi-periodic at ({phi}, {k})");
                }
                for &k in &[0.3f64, 0.7, 0.95] {
                    for n in 0..4 {
                        let phi = core::f64::consts::FRAC_PI_2 * n as f64;
                        let z = ell(JacobiZeta { phi: v(phi), k: v(k) });
                        assert!(z.abs() < 1.0e-15, "Z({phi}, {k}) should vanish, got {z}");
                    }
                }
                // k = 0 makes Z identically zero. k = 1 is the pinned hyperbolic-free limit
                // sin(phi) * sign(cos phi), neither of which the Carlson form can reach.
                assert_eq!(
                    ell(JacobiZeta {
                        phi: v(1.1),
                        k: v(0.0)
                    }),
                    0.0,
                    "Z(phi, 0)"
                );
                let one = F64::ellint_p::<CheckOverflow<Precision, true>, _>(JacobiZeta {
                    phi: v(1.1),
                    k: v(1.0),
                })
                .extract::<0>();
                assert!(
                    close(one, 1.1f64.sin(), 1.0e-15),
                    "Z(1.1, 1) = sin(1.1), got {one}"
                );
            }

            #[test]
            fn heuman_lambda_matches_the_defining_combination() {
                for &(phi, k, want) in HEUMAN_LAMBDA {
                    let got = ell(HeumanLambda { phi: v(phi), k: v(k) });
                    assert!(
                        close(got, want, 1.0e-13),
                        "Lambda({phi}, {k}): got {got}, want {want}"
                    );
                }
            }

            // The endpoints that define it, and the seam between its two arms. The R_J
            // parameter `1 - k^2/delta^2` is exactly zero at phi = pi/2, so the naive
            // spelling rounds negative just short of it and silently switches R_J to its
            // principal-value branch. These pin that it does not.
            #[test]
            fn heuman_lambda_endpoints_and_arm_seam() {
                for &k in &[0.1f64, 0.5, 0.9] {
                    let zero = ell(HeumanLambda { phi: v(0.0), k: v(k) });
                    assert_eq!(zero, 0.0, "Lambda(0, {k}) should be 0");

                    let half = ell(HeumanLambda {
                        phi: v(core::f64::consts::FRAC_PI_2),
                        k: v(k),
                    });
                    assert!(
                        close(half, 1.0, 1.0e-14),
                        "Lambda(pi/2, {k}) should be 1, got {half}"
                    );

                    // Straddle the pi/2 seam: the two arms must agree across it.
                    let eps = 1.0e-9;
                    let lo = ell(HeumanLambda {
                        phi: v(core::f64::consts::FRAC_PI_2 - eps),
                        k: v(k),
                    });
                    let hi = ell(HeumanLambda {
                        phi: v(core::f64::consts::FRAC_PI_2 + eps),
                        k: v(k),
                    });
                    assert!(
                        close(lo, hi, 1.0e-8),
                        "arms disagree at pi/2 for k={k}: {lo} vs {hi}"
                    );
                }
            }

            /// The Jacobi triple through its public entry, at lane 0.
            fn jac(u: f64, k: f64) -> (f64, f64, f64) {
                let (sn, cn, dn) = F64::jacobi_elliptic_p::<Precision>(v(u), v(k));
                (sn.extract::<0>(), cn.extract::<0>(), dn.extract::<0>())
            }

            #[test]
            fn jacobi_elliptic_matches_reference() {
                for &(u, k, wsn, wcn, wdn) in JACOBI {
                    let (sn, cn, dn) = jac(u, k);
                    // Absolute tolerance on purpose: all three are bounded by 1 and all three
                    // have zeros, so a relative gate at a zero would be testing how well the
                    // zero's location is known rather than the function.
                    assert!(close(sn, wsn, 1.0e-14), "sn({u}, {k}): got {sn}, want {wsn}");
                    assert!(close(cn, wcn, 1.0e-14), "cn({u}, {k}): got {cn}, want {wcn}");
                    assert!(close(dn, wdn, 1.0e-14), "dn({u}, {k}): got {dn}, want {wdn}");
                }
            }

            // sn^2 + cn^2 = 1 and k^2 sn^2 + dn^2 = 1 hold identically. They are independent
            // of the reference table and catch a triple that is self-consistently wrong.
            #[test]
            fn jacobi_elliptic_identities() {
                for ki in 0..20 {
                    let k = ki as f64 / 20.0;
                    for ui in 0..20 {
                        let u = -6.0 + 12.0 * ui as f64 / 19.0;
                        let (sn, cn, dn) = jac(u, k);
                        let pyth = sn * sn + cn * cn - 1.0;
                        let delta = k * k * sn * sn + dn * dn - 1.0;
                        assert!(pyth.abs() < 1.0e-14, "sn^2+cn^2-1 = {pyth} at u={u}, k={k}");
                        assert!(delta.abs() < 1.0e-14, "k^2 sn^2+dn^2-1 = {delta} at u={u}, k={k}");
                    }
                }
            }

            // sn is odd in u, cn and dn are even. The kernel gets this from the sign of the
            // single sine at the bottom of the ladder, so it is worth pinning.
            #[test]
            fn jacobi_elliptic_parity() {
                for &(u, k) in &[(0.7, 0.5), (2.3, 0.9), (5.0, 0.25)] {
                    let (sp, cp, dp) = jac(u, k);
                    let (sm, cm, dm) = jac(-u, k);
                    assert_eq!(sm, -sp, "sn parity at u={u}, k={k}");
                    assert_eq!(cm, cp, "cn parity at u={u}, k={k}");
                    assert_eq!(dm, dp, "dn parity at u={u}, k={k}");
                }
            }

            // The two moduli where the ladder degenerates: k = 0 is pure trigonometry, and
            // k = 1 is the hyperbolic limit the ladder cannot walk to and so substitutes.
            #[test]
            fn jacobi_elliptic_degenerate_moduli() {
                for &u in &[0.0f64, 0.4, 1.7, -3.2] {
                    let (sn, cn, dn) = jac(u, 0.0);
                    assert!(close(sn, u.sin(), 1.0e-15), "sn(u,0) != sin u at u={u}");
                    assert!(close(cn, u.cos(), 1.0e-15), "cn(u,0) != cos u at u={u}");
                    assert_eq!(dn, 1.0, "dn(u,0) != 1 at u={u}");

                    let f = |u: f64, k: f64| {
                        let (sn, cn, dn) = F64::jacobi_elliptic_p::<CheckOverflow<Precision, true>>(v(u), v(k));
                        (sn.extract::<0>(), cn.extract::<0>(), dn.extract::<0>())
                    };
                    let (sn, cn, dn) = f(u, 1.0);
                    let sech = 1.0 / u.cosh();
                    assert!(close(sn, u.tanh(), 1.0e-15), "sn(u,1) != tanh u at u={u}");
                    assert!(close(cn, sech, 1.0e-15), "cn(u,1) != sech u at u={u}");
                    assert!(close(dn, sech, 1.0e-15), "dn(u,1) != sech u at u={u}");

                    // Only k^2 enters, so the sign of the modulus is irrelevant, and |k| > 1
                    // leaves the domain through a negative square root rather than a guard.
                    let (sp, cp, dp) = f(u, 0.6);
                    let (sm, cm, dm) = f(u, -0.6);
                    assert_eq!((sp, cp, dp), (sm, cm, dm), "modulus sign at u={u}");
                    assert!(f(u, 1.5).0.is_nan(), "|k| > 1 should be NaN at u={u}");
                }
            }

            /// The general AGM through its own public entry, at lane 0.
            fn agm(a: f64, b: f64) -> f64 {
                F64::agm_p::<Precision>(v(a), v(b)).extract::<0>()
            }

            #[test]
            fn agm_matches_reference() {
                for &(a, b, want) in AGM {
                    let got = agm(a, b);
                    assert!(close(got, want, 1.0e-14), "AGM({a}, {b}): got {got}, want {want}");
                    // Symmetric in its arguments, and to the last bit: the recurrence's first
                    // pass is symmetric, so the whole thing is.
                    assert_eq!(agm(b, a), got, "AGM({b}, {a}) != AGM({a}, {b})");
                }
            }

            // K(k) = pi / (2 AGM(1, k')). The complete integral runs its own copy of the
            // recurrence with the `E` accumulator attached, so agreement here is the check
            // that the two did not drift apart.
            #[test]
            fn agm_matches_complete_k() {
                for &(k, want_k, _) in KE {
                    let kp = (1.0 - k * k).sqrt();
                    let got = core::f64::consts::FRAC_PI_2 / agm(1.0, kp);
                    assert!(
                        close(got, want_k, 1.0e-14),
                        "K({k}) via AGM: got {got}, want {want_k}"
                    );
                }
            }

            // Homogeneous: AGM(ca, cb) = c AGM(a, b). Scaling by a power of two is exact on
            // both sides, so this holds bit for bit and is what a caller working near the
            // overflow threshold is told to rely on.
            #[test]
            fn agm_is_homogeneous_in_powers_of_two() {
                for &(a, b, _) in AGM {
                    for c in [0.25f64, 4.0, 2f64.powi(-100), 2f64.powi(100)] {
                        let scaled = agm(a * c, b * c);
                        let want = agm(a, b) * c;
                        assert_eq!(scaled, want, "AGM({a}*{c}, {b}*{c}) != {c} * AGM({a}, {b})");
                    }
                }
            }

            #[test]
            fn agm_edge_cases() {
                let f = |a: f64, b: f64| F64::agm_p::<CheckOverflow<Precision, true>>(v(a), v(b)).extract::<0>();

                // AGM(a, 0) = 0: the iteration only walks toward it, so this is the pin.
                assert_eq!(f(3.0, 0.0), 0.0, "AGM(3, 0)");
                assert_eq!(f(0.0, 3.0), 0.0, "AGM(0, 3)");
                assert_eq!(f(0.0, 0.0), 0.0, "AGM(0, 0)");

                assert_eq!(f(f64::INFINITY, 2.0), f64::INFINITY, "AGM(inf, 2)");
                assert_eq!(f(2.0, f64::INFINITY), f64::INFINITY, "AGM(2, inf)");

                // No limit: zero against infinity, and anything negative.
                assert!(f(0.0, f64::INFINITY).is_nan(), "AGM(0, inf)");
                assert!(f(-1.0, 2.0).is_nan(), "AGM(-1, 2)");
                assert!(f(1.0, -2.0).is_nan(), "AGM(1, -2)");
                assert!(f(f64::NAN, 1.0).is_nan(), "AGM(NaN, 1)");
            }

            // The hardcoded EllipticConsts::CARLSON_THRESH literals must equal sqrt(sqrt(sqrt(3*eps))).
            #[test]
            fn carlson_thresh_const_matches_runtime() {
                let e64 = f64::EPSILON;
                assert_eq!(
                    <f64 as EllipticConsts>::CARLSON_THRESH,
                    (e64 + e64 + e64).sqrt().sqrt().sqrt()
                );
                let e32 = f32::EPSILON;
                assert_eq!(
                    <f32 as EllipticConsts>::CARLSON_THRESH,
                    (e32 + e32 + e32).sqrt().sqrt().sqrt()
                );
            }

            // Every Carlson kind at its degenerate point, where the value is exact in closed form:
            // R_F(x,x,x) = x^-1/2, R_D(x,x,x) = R_J(x,x,x,x) = x^-3/2, R_G(x,x,x) = x^1/2,
            // R_C(x,x) = x^-1/2. Also pins the request-struct arities: each kind takes exactly its
            // own arguments, so the wrong shape would not compile.
            #[test]
            fn carlson_spot_values() {
                let q = v(4.0);
                assert!(
                    close(car(CarlsonRf { x: q, y: q, z: q }), 0.5, 1.0e-14),
                    "R_F(4,4,4)"
                );
                assert!(
                    close(car(CarlsonRd { x: q, y: q, z: q }), 0.125, 1.0e-14),
                    "R_D(4,4,4)"
                );
                assert!(
                    close(car(CarlsonRg { x: q, y: q, z: q }), 2.0, 1.0e-14),
                    "R_G(4,4,4)"
                );
                assert!(close(car(CarlsonRc { x: q, y: q }), 0.5, 1.0e-14), "R_C(4,4)");
                assert!(
                    close(
                        car(CarlsonRj {
                            x: q,
                            y: q,
                            z: q,
                            p: q
                        }),
                        0.125,
                        1.0e-13
                    ),
                    "R_J(4,4,4,4)"
                );
            }

            // f32 R_C exercises whichever closed-form lowering this backend picks: the
            // hardware-rsqrt one where `HAS_APPROX_RSQRT` is set (x86), the sqrt+div one
            // otherwise, with the f64 cases only ever reaching the latter. Covers both closed
            // forms, t > 0 (atan) via R_C(1,2) = atan(1) = pi/4, and t < 0 (ln/atanh) via
            // R_C(2,1) = ln(1 + sqrt 2), plus the small-|t| series via R_C(4,4) = 1/2.
            #[test]
            fn carlson_rc_f32_closed_forms() {
                let rc = |x: f32, y: f32| {
                    F32::carlson_p::<Precision, _>(CarlsonRc {
                        x: F32::splat(x),
                        y: F32::splat(y),
                    })
                    .extract::<0>() as f64
                };
                assert!(
                    close(rc(1.0, 2.0), core::f64::consts::FRAC_PI_4, 1.0e-6),
                    "R_C(1,2) = pi/4"
                );
                assert!(
                    close(rc(2.0, 1.0), (1.0 + 2.0_f64.sqrt()).ln(), 1.0e-6),
                    "R_C(2,1) = ln(1+sqrt2)"
                );
                assert!(close(rc(4.0, 4.0), 0.5, 1.0e-6), "R_C(4,4) = 1/2 (series)");
            }

            // R_G via the complete second-kind identity E(k) = 2 R_G(0, 1-k^2, 1), with E from the
            // (independently validated) AGM. Exercises the zero-argument path (lo = 0, mid = 1-k^2 > 0).
            #[test]
            fn carlson_rg_matches_complete_e() {
                for &(k, _, want_e) in KE {
                    let omk2 = v((1.0 - k) * (1.0 + k)); // 1 - k^2
                    let got = 2.0
                        * car(CarlsonRg {
                            x: v(0.0),
                            y: omk2,
                            z: v(1.0),
                        });
                    assert!(
                        close(got, want_e, 1.0e-13),
                        "2 R_G(0,1-k^2,1) vs E({k}): got {got}, want {want_e}"
                    );
                }
            }

            // Cross-check the Carlson primitives against the complete integrals, which reach them
            // by a different route (the AGM):
            //   K(k) = R_F(0, 1-k^2, 1)
            //   E(k) = R_F(0, 1-k^2, 1) - (k^2/3) R_D(0, 1-k^2, 1)
            #[test]
            fn carlson_matches_complete_ke() {
                for &(k, _, _) in KE {
                    let want_k = ell(EllintK { k: v(k) });
                    let want_e = ell(EllintE { k: v(k) });
                    let omk2 = v((1.0 - k) * (1.0 + k)); // 1 - k^2
                    let (zero, one) = (v(0.0), v(1.0));

                    let rf = car(CarlsonRf {
                        x: zero,
                        y: omk2,
                        z: one,
                    });
                    let rd = car(CarlsonRd {
                        x: zero,
                        y: omk2,
                        z: one,
                    });
                    let e = rf - (k * k / 3.0) * rd;

                    assert!(close(rf, want_k, 1.0e-13), "R_F vs K({k})");
                    assert!(close(e, want_e, 1.0e-13), "R_F/R_D vs E({k})");
                }
            }

            #[test]
            fn ellint_incomplete_matches_reference() {
                for &(phi, k, want_f, want_e) in INC {
                    let f = ell(EllintF { phi: v(phi), k: v(k) });
                    let e = ell(EllintEInc { phi: v(phi), k: v(k) });
                    assert!(close(f, want_f, 1.0e-12), "F({phi},{k}): got {f}, want {want_f}");
                    assert!(close(e, want_e, 1.0e-12), "E({phi},{k}): got {e}, want {want_e}");
                }
            }

            // The D kind has no reference table. It is defined as D = (F - E) / k^2, so the
            // validated F/E rows are the oracle. The tolerance carries the subtraction's
            // cancellation, and the k = 0.128 row loses ~2 digits in the *oracle*, not in D.
            // (k = 0 is 0/0 and is excluded by definition, not by convenience.)
            #[test]
            fn ellint_d_matches_f_minus_e() {
                for &(k, want_k, want_e) in KE {
                    if k == 0.0 {
                        continue;
                    }
                    let got = ell(EllintD { k: v(k) });
                    let want = (want_k - want_e) / (k * k);
                    assert!(close(got, want, 1.0e-11), "D({k}): got {got}, want {want}");
                }
                for &(phi, k, want_f, want_e) in INC {
                    let got = ell(EllintDInc { phi: v(phi), k: v(k) });
                    let want = (want_f - want_e) / (k * k);
                    assert!(close(got, want, 1.0e-11), "D({phi},{k}): got {got}, want {want}");
                }
            }

            // phi-range reduction: I(phi + m*pi) = I(phi) + 2m * I_complete, for all kinds. We take
            // the validated phi in [0, pi/2] reference rows, shift phi by +/- m*pi, and check the
            // reduced result matches the identity (no external data needed for phi > pi/2).
            #[test]
            fn ellint_phi_range_reduction() {
                for &(phi, k, want_f, want_e) in INC {
                    // |k| > 1 rows constrain phi < pi/2 and have NaN complete values, so skip: the
                    // identity does not apply there (and m = 0 keeps them correct anyway).
                    if k.abs() > 1.0 {
                        continue;
                    }
                    let comp_k = ell(EllintK { k: v(k) });
                    let comp_e = ell(EllintE { k: v(k) });
                    for m in [-2i32, -1, 1, 3] {
                        let shifted = v(phi + m as f64 * core::f64::consts::PI);
                        let f = ell(EllintF {
                            phi: shifted,
                            k: v(k),
                        });
                        let e = ell(EllintEInc {
                            phi: shifted,
                            k: v(k),
                        });
                        let ef = want_f + 2.0 * m as f64 * comp_k;
                        let ee = want_e + 2.0 * m as f64 * comp_e;
                        assert!(close(f, ef, 1.0e-11), "F({phi}+{m}pi,{k}): got {f}, want {ef}");
                        assert!(close(e, ee, 1.0e-11), "E({phi}+{m}pi,{k}): got {e}, want {ee}");
                    }
                }
            }

            /// The same identity at *large* `|phi|`, where the reduced angle has lost most of
            /// its digits to `m*PI` rounding, the point being that the result does not care.
            /// The periodic term grows with `m` at the same rate the reduction error does, so
            /// the identity keeps holding to near f64 relative accuracy regardless.
            #[test]
            fn ellint_phi_range_reduction_large() {
                // A representative interior point rather than the whole table: this is about the
                // reduction, not the integrand.
                let (phi, k) = (0.7, 0.5);

                let comp_k = ell(EllintK { k: v(k) });
                let base = ell(EllintF { phi: v(phi), k: v(k) });

                for m in [1_000i64, 100_000, 10_000_000, 1_000_000_000] {
                    let f = ell(EllintF {
                        phi: v(phi + m as f64 * core::f64::consts::PI),
                        k: v(k),
                    });

                    let want = base + 2.0 * m as f64 * comp_k;

                    // Relative, because the value grows with m.
                    let err = (f - want).abs() / want.abs();
                    assert!(err < 1e-12, "F(phi + {m}*pi): got {f}, want {want} (rel {err})");
                }
            }

            #[test]
            fn ellint_pi_complete_matches_reference() {
                for &(n, m, want) in PIC {
                    let k = m.sqrt();
                    let got = ell(EllintPi { n: v(n), k: v(k) });
                    assert!(
                        close(got, want, 1.0e-13),
                        "Pi_complete({n},k={k}): got {got}, want {want}"
                    );
                }
            }

            #[test]
            fn ellint_pi_matches_reference() {
                for &(phi, n, k, want) in PI3 {
                    let got = ell(EllintPiInc {
                        n: v(n),
                        phi: v(phi),
                        k: v(k),
                    });
                    assert!(
                        close(got, want, 1.0e-12),
                        "Pi({phi},{n},{k}): got {got}, want {want}"
                    );
                }
            }

            // Reduction applies to the third kind too: Pi(n, phi + m*pi, k) = Pi(n, phi, k) + 2m Pi(n, k).
            #[test]
            fn ellint_pi_phi_range_reduction() {
                for &(phi, n, k, want) in PI3 {
                    // |k| > 1 makes the complete Pi NaN; the identity does not apply (see F/E test).
                    if k.abs() > 1.0 {
                        continue;
                    }
                    let comp = ell(EllintPi { n: v(n), k: v(k) });
                    for m in [-1i32, 1, 2] {
                        let got = ell(EllintPiInc {
                            n: v(n),
                            phi: v(phi + m as f64 * core::f64::consts::PI),
                            k: v(k),
                        });
                        let exp = want + 2.0 * m as f64 * comp;
                        assert!(
                            close(got, exp, 1.0e-11),
                            "Pi({n},{phi}+{m}pi,{k}): got {got}, want {exp}"
                        );
                    }
                }
            }

            // The degenerate and singular points of every kind, against exact closed forms
            // rather than table entries. Each of these used to return a capped finite number
            // (the duplication loop's iteration limit leaking out as the answer), a 0/0 NaN,
            // or an inf*0 NaN. The infinities are genuine divergences, and a large finite
            // number is the worse answer there because it survives an `is_finite` check;
            // E(1) = 1 and D(0) = pi/4 are the opposite case, ordinary finite values that
            // the general formula only reaches as a limit, landing 21% low and NaN without
            // the pin.
            #[test]
            fn domain_edges() {
                let inf = f64::INFINITY;
                let hp = core::f64::consts::FRAC_PI_2;
                let (n, u) = (v(0.0), v(1.0)); // zero and one, named to dodge the `z` field

                // Carlson: R_F/R_D/R_J diverge once two arguments vanish (and R_D/R_J also on
                // their own singular argument); R_G stays finite and collapses to sqrt(hi)/2.
                let edges: &[(&str, f64, f64)] = &[
                    ("R_F(0,0,1)", car(CarlsonRf { x: n, y: n, z: u }), inf),
                    (
                        "R_F(0,1,1)",
                        car(CarlsonRf { x: n, y: u, z: u }),
                        core::f64::consts::FRAC_PI_2,
                    ),
                    ("R_C(1,0)", car(CarlsonRc { x: u, y: n }), inf),
                    (
                        "R_C(0,1)",
                        car(CarlsonRc { x: n, y: u }),
                        core::f64::consts::FRAC_PI_2,
                    ),
                    ("R_D(0,0,1)", car(CarlsonRd { x: n, y: n, z: u }), inf),
                    ("R_D(1,1,0)", car(CarlsonRd { x: u, y: u, z: n }), inf),
                    (
                        "R_D(0,1,1)",
                        car(CarlsonRd { x: n, y: u, z: u }),
                        2.3561944901923449,
                    ),
                    (
                        "R_J(0,0,1,1)",
                        car(CarlsonRj {
                            x: n,
                            y: n,
                            z: u,
                            p: u,
                        }),
                        inf,
                    ),
                    (
                        "R_J(1,1,1,0)",
                        car(CarlsonRj {
                            x: u,
                            y: u,
                            z: u,
                            p: n,
                        }),
                        inf,
                    ),
                    (
                        "R_J(0,1,1,1)",
                        car(CarlsonRj {
                            x: n,
                            y: u,
                            z: u,
                            p: u,
                        }),
                        2.3561944901923449,
                    ),
                    ("R_G(0,0,1)", car(CarlsonRg { x: n, y: n, z: u }), 0.5),
                    ("R_G(0,0,0)", car(CarlsonRg { x: n, y: n, z: n }), 0.0),
                    (
                        "R_G(0,1,1)",
                        car(CarlsonRg { x: n, y: u, z: u }),
                        core::f64::consts::FRAC_PI_4,
                    ),
                    // Legendre at |k| = 1: K and D diverge, but E(1) = 1 exactly.
                    ("K(1)", ell(EllintK { k: v(1.0) }), inf),
                    ("E(1)", ell(EllintE { k: v(1.0) }), 1.0),
                    ("D(1)", ell(EllintD { k: v(1.0) }), inf),
                    ("K(-1)", ell(EllintK { k: v(-1.0) }), inf),
                    ("E(-1)", ell(EllintE { k: v(-1.0) }), 1.0),
                    (
                        "E(pi/2,1)",
                        ell(EllintEInc {
                            phi: v(hp),
                            k: v(1.0),
                        }),
                        1.0,
                    ),
                    ("Pi(n=0.5,k=1)", ell(EllintPi { n: v(0.5), k: v(1.0) }), inf),
                    // n = 1 puts the R_J parameter p = 1 - n at zero: a pole, not a NaN.
                    ("Pi(n=1,k=0.5)", ell(EllintPi { n: v(1.0), k: v(0.5) }), inf),
                    // k = 0 makes complete D a 0/0 whose limit is pi/4.
                    ("D(0)", ell(EllintD { k: v(0.0) }), core::f64::consts::FRAC_PI_4),
                    ("K(0)", ell(EllintK { k: v(0.0) }), core::f64::consts::FRAC_PI_2),
                    ("E(0)", ell(EllintE { k: v(0.0) }), core::f64::consts::FRAC_PI_2),
                    (
                        "D(0.7,0)",
                        ell(EllintDInc {
                            phi: v(0.7),
                            k: v(0.0),
                        }),
                        0.10363756750288495,
                    ),
                    (
                        "F(0,0.5)",
                        ell(EllintF {
                            phi: v(0.0),
                            k: v(0.5),
                        }),
                        0.0,
                    ),
                    // Out of domain stays NaN: the edge pins must not turn these into values.
                    ("K(1.5)", ell(EllintK { k: v(1.5) }), f64::NAN),
                    (
                        "F(1.4,1.5)",
                        ell(EllintF {
                            phi: v(1.4),
                            k: v(1.5),
                        }),
                        f64::NAN,
                    ),
                ];

                for &(label, got, want) in edges {
                    let ok = if want.is_nan() {
                        got.is_nan()
                    } else if want.is_infinite() {
                        got == want
                    } else {
                        close(got, want, 1.0e-13)
                    };
                    assert!(ok, "{label}: got {got}, want {want}");
                }
            }

            // Accuracy walking in toward |k sin(phi)| = 1. See the note above NEAR_BOUNDARY.
            #[test]
            fn ellint_near_domain_boundary() {
                for &(k, phi, want_f, want_e, tol) in NEAR_BOUNDARY {
                    let f = ell(EllintF { phi: v(phi), k: v(k) });
                    let e = ell(EllintEInc { phi: v(phi), k: v(k) });
                    let (ef, ee) = ((f - want_f).abs() / want_f, (e - want_e).abs() / want_e);
                    assert!(ef < tol, "F({phi},{k}): got {f}, want {want_f} (rel {ef:.2e})");
                    assert!(ee < tol, "E({phi},{k}): got {e}, want {want_e} (rel {ee:.2e})");
                }
            }

            // The `check_overflow: false` arms of every edge fix are otherwise never compiled
            // by the suite. Ordinary inputs must be unaffected by the flag, which only decides
            // whether the singular points get pinned.
            #[test]
            fn unchecked_policy_matches_on_ordinary_inputs() {
                type Unchecked = CheckOverflow<Precision, false>;
                for &(k, want_k, want_e) in KE {
                    let gk = F64::ellint_p::<Unchecked, _>(EllintK { k: v(k) }).extract::<0>();
                    let ge = F64::ellint_p::<Unchecked, _>(EllintE { k: v(k) }).extract::<0>();
                    assert!(close(gk, want_k, 1.0e-13), "unchecked K({k}): got {gk}");
                    assert!(close(ge, want_e, 1.0e-13), "unchecked E({k}): got {ge}");
                }
                for &(phi, k, want_f, want_e) in INC {
                    let f = F64::ellint_p::<Unchecked, _>(EllintF { phi: v(phi), k: v(k) }).extract::<0>();
                    let e = F64::ellint_p::<Unchecked, _>(EllintEInc { phi: v(phi), k: v(k) }).extract::<0>();
                    assert!(close(f, want_f, 1.0e-12), "unchecked F({phi},{k}): got {f}");
                    assert!(close(e, want_e, 1.0e-12), "unchecked E({phi},{k}): got {e}");
                }
                let q = v(4.0);
                let rf = F64::carlson_p::<Unchecked, _>(CarlsonRf { x: q, y: q, z: q }).extract::<0>();
                let rg = F64::carlson_p::<Unchecked, _>(CarlsonRg { x: q, y: q, z: q }).extract::<0>();
                let rj = F64::carlson_p::<Unchecked, _>(CarlsonRj {
                    x: q,
                    y: q,
                    z: q,
                    p: q,
                })
                .extract::<0>();
                assert!(close(rf, 0.5, 1.0e-14), "unchecked R_F: got {rf}");
                assert!(close(rg, 2.0, 1.0e-14), "unchecked R_G: got {rg}");
                assert!(close(rj, 0.125, 1.0e-13), "unchecked R_J: got {rj}");
            }

            #[test]
            fn carlson_rj_negative_p() {
                for &(x, y, z, p) in &[
                    (1.0, 2.0, 4.0, -0.5),
                    (0.5, 1.0, 2.0, -0.25),
                    (1.0, 3.0, 5.0, -2.0),
                ] {
                    let want = rj_pv_reference(x, y, z, p);
                    let got = car(CarlsonRj {
                        x: v(x),
                        y: v(y),
                        z: v(z),
                        p: v(p),
                    });
                    assert!(
                        close(got, want, 1.0e-6),
                        "R_J({x},{y},{z},{p}): got {got}, PV oracle {want}"
                    );
                }
            }
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    elliptic_tests!(v3, X86V3, f32x8, f64x4);
    elliptic_tests!(v2, X86V2, f32x4, f64x2);
    elliptic_tests!(v1, X86V1, f32x4, f64x2);
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    elliptic_tests!(wasm, Wasm, f32x4, f64x2);
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;
    elliptic_tests!(neon, Neon, f32x4, f64x2);
}
