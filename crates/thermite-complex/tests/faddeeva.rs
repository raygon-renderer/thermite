//! The Faddeeva function `w(z)` over C (`special` feature).
//!
//! Checked against reference values computed by mpmath at 60 digits, an oracle that
//! shares no code with the implementation. The tolerances are the measured worst case
//! for each tier (see `faddeeva`'s module docs), loosened by roughly an order.
//!
//! Accuracy is asserted **normwise**, relative to `|w|`. That is the guarantee the
//! algorithm actually makes; `Re w` alone is far less accurate near the real axis at
//! large `x`, which is a property of `w` (where `Re w = exp(-x^2)` is astronomically
//! smaller than `|w|`) rather than of the approximation. The test
//! `re_is_exp_neg_x_squared_on_the_real_axis` pins exactly how far that goes.

#![cfg(feature = "special")]

use thermite::math::policy::policies::{Performance, Precision, Reference, UltraPerformance};
use thermite::prelude::*;

use thermite_complex::Complex;
use thermite_complex::prelude::{ComplexSpecialMath, ComplexSpecialMathWithPolicy};

type V = Vector<f64>;
type C = Complex<V>;

fn c(re: f64, im: f64) -> C {
    Complex::new(V::splat(re), V::splat(im))
}

fn parts(z: C) -> (f64, f64) {
    (z.re.extract::<0>(), z.im.extract::<0>())
}

/// Normwise relative error, `|got - want| / |want|`.
fn rel(got: C, want: (f64, f64)) -> f64 {
    let (re, im) = parts(got);
    let d = ((re - want.0).powi(2) + (im - want.1).powi(2)).sqrt();
    let n = (want.0 * want.0 + want.1 * want.1).sqrt();

    if n == 0.0 { d } else { d / n }
}

/// Reference values `(Re z, Im z, Re w, Im w)`, from mpmath at 60 digits.
///
/// Regenerate with `faddeeva_reference.py`, in this directory.
#[rustfmt::skip]
const REF: [(f64, f64, f64, f64); 118] = [
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 1e-08, 0.9999999887162084, 0.0),
    (0.0, 0.001, 0.9988726200811514, 0.0),
    (0.0, 0.1, 0.8964569799691267, 0.0),
    (0.0, 1.0, 0.427583576155807, 0.0),
    (0.0, 5.0, 0.11070463773306863, 0.0),
    (0.0, 30.0, 0.01879588886141675, 0.0),
    (1e-08, 0.0, 0.9999999999999999, 1.1283791670955125e-08),
    (1e-08, 1e-08, 0.9999999887162083, 1.1283791470955128e-08),
    (1e-08, 0.001, 0.9988726200811513, 1.1263814218553502e-08),
    (1e-08, 0.1, 0.8964569799691265, 9.490877711016872e-09),
    (1e-08, 1.0, 0.427583576155807, 2.7321201478389856e-09),
    (1e-08, 5.0, 0.11070463773306863, 2.133278976482631e-10),
    (1e-08, 30.0, 0.01879588886141675, 6.2583541050748405e-12),
    (0.001, 0.0, 0.9999990000005, 0.0011283784148430353),
    (0.001, 1e-08, 0.999998988716731, 0.0011283783948430556),
    (0.001, 0.001, 0.9988716223354113, 0.0011263806715998664),
    (0.001, 0.1, 0.8964561784212955, 0.0009490871918132702),
    (0.001, 1.0, 0.4275834217842832, 0.0002732119355569463),
    (0.001, 5.0, 0.11070463369237986, 2.133278901192952e-05),
    (0.001, 30.0, 0.018795888840590205, 6.258354098148009e-07),
    (0.1, 0.0, 0.9900498337491681, 0.11208866436449538),
    (0.1, 1e-08, 0.9900498226895538, 0.11208866238439574),
    (0.1, 0.001, 0.9889448418417955, 0.1118908768860108),
    (0.1, 0.1, 0.8884785624756437, 0.09433165105728511),
    (0.1, 1.0, 0.42604361081205644, 0.02724214085161446),
    (0.1, 5.0, 0.11066424464977836, 0.0021325263291299993),
    (0.1, 30.0, 0.018795680598257373, 6.258284837523776e-05),
    (0.5, 0.0, 0.7788007830714049, 0.47892517290104347),
    (0.5, 1e-08, 0.778800776576865, 0.47892516511303573),
    (0.5, 0.001, 0.7781517183125491, 0.4781471751215842),
    (0.5, 0.1, 0.7175877421575944, 0.40847440160301646),
    (0.5, 1.0, 0.3912340214521361, 0.127202410884648),
    (0.5, 5.0, 0.1097030279891138, 0.010573056535802454),
    (0.5, 30.0, 0.018790683663577543, 0.0003128311437578235),
    (1.0, 0.0, 0.36787944117144233, 0.6071577058413937),
    (1.0, 1e-08, 0.36787944203080475, 0.607157698483805),
    (1.0, 0.001, 0.36796500994105386, 0.6064224679353174),
    (1.0, 0.1, 0.37317014831126744, 0.5385548078594318),
    (1.0, 1.0, 0.3047442052569126, 0.20821893820283163),
    (1.0, 5.0, 0.10679773839806538, 0.02060408871468425),
    (1.0, 30.0, 0.018775085315541475, 0.0006251434914879298),
    (2.0, 0.0, 0.01831563888873418, 0.3400262170660662),
    (2.0, 1e-08, 0.01831564120599118, 0.34002621633344066),
    (2.0, 0.001, 0.01854723637040555, 0.3399528312073786),
    (2.0, 0.1, 0.04020139816145129, 0.3315826873345631),
    (2.0, 1.0, 0.14023958136627795, 0.2222134401798991),
    (2.0, 5.0, 0.09649811260664139, 0.03735165315636875),
    (2.0, 30.0, 0.018712949469146704, 0.0012461537277561356),
    (3.5, 0.0, 4.785117392129009e-06, 0.1688298885799677),
    (3.5, 1e-08, 4.78565169265864e-06, 0.16882988857963274),
    (3.5, 0.001, 5.821505111750621e-05, 0.168829836909092),
    (3.5, 0.1, 0.005339924882782207, 0.1686453008191369),
    (3.5, 1.0, 0.04769823536579902, 0.15298765763500582),
    (3.5, 5.0, 0.07600262630785862, 0.051819629671132954),
    (3.5, 30.0, 0.018544172890514035, 0.0021611215780780613),
    (5.0, 0.0, 1.3887943864964021e-11, 0.11524596183093659),
    (5.0, 1e-08, 2.5469245600349706e-10, 0.11524596183093659),
    (5.0, 0.001, 2.4080463967103415e-05, 0.11524595667450373),
    (5.0, 0.1, 0.002406911716942712, 0.11519442455072769),
    (5.0, 1.0, 0.023003132594059963, 0.11033283255357997),
    (5.0, 5.0, 0.056965439888176976, 0.055838742775391026),
    (5.0, 30.0, 0.018289230001030467, 0.003044918133625361),
    (8.0, 0.0, 1.603810890548638e-28, 0.07108811174448088),
    (8.0, 1e-08, 9.0306208161815e-11, 0.07108811174448088),
    (8.0, 0.001, 9.030620666703193e-06, 0.07108811058762611),
    (8.0, 0.1, 0.0009029126289382924, 0.07107654514487521),
    (8.0, 1.0, 0.008883661074217763, 0.06995040848005314),
    (8.0, 5.0, 0.032031988644396714, 0.050673023009802066),
    (8.0, 30.0, 0.017551082670589687, 0.004675444860494793),
    (15.0, 0.0, 1.921947727823849e-98, 0.03769678605913683),
    (15.0, 1e-08, 2.524414678592424e-11, 0.03769678605913683),
    (15.0, 0.001, 2.5244146671943456e-06, 0.03769678588970271),
    (15.0, 0.1, 0.00025243007030017377, 0.03769509179478866),
    (15.0, 1.0, 0.0025130683012635035, 0.03752811696561413),
    (15.0, 5.0, 0.011342898608733479, 0.0338919776577923),
    (15.0, 30.0, 0.015043711469150045, 0.007515179733162054),
    (50.0, 0.0, 0.0, 0.011286049784700271),
    (50.0, 1e-08, 2.258113745145637e-12, 0.011286049784700271),
    (50.0, 0.001, 2.2581137442411246e-07, 0.01128604978018133),
    (50.0, 0.1, 2.258104700056489e-05, 0.011286004595471076),
    (50.0, 1.0, 0.00022572095950627497, 0.011281532653784522),
    (50.0, 5.0, 0.0011178626541078892, 0.011174196798275594),
    (50.0, 30.0, 0.004979564747361252, 0.00829683329940245),
    (1000.0, 0.0, 0.0, 0.0005641898656429712),
    (1000.0, 1e-08, 5.6419042983424735e-15, 0.0005641898656429712),
    (1000.0, 0.001, 5.641904298336832e-10, 0.000564189865642407),
    (1000.0, 0.1, 5.641904241923234e-08, 0.0005641898600010585),
    (1000.0, 1.0, 5.641898656424071e-07, 0.0005641893014522593),
    (1000.0, 5.0, 2.8208816268837396e-06, 0.0005641757612136804),
    (1000.0, 30.0, 1.691049339772852e-05, 0.0005636825500805207),
    (-0.5, 0.0, 0.7788007830714049, -0.47892517290104347),
    (-0.5, 0.25, 0.6383373967914265, -0.3258148398263451),
    (-0.5, 3.0, 0.175105212623158, -0.026636168446230884),
    (-2.0, 0.0, 0.01831563888873418, -0.3400262170660662),
    (-2.0, 0.25, 0.0682634892706679, -0.31570766271099415),
    (-2.0, 3.0, 0.13075746966984858, -0.08111265047745665),
    (-6.0, 0.0, 2.3195228302435696e-16, -0.09539620896911076),
    (-6.0, 0.25, 0.0040859383398352545, -0.09521807564156685),
    (-6.0, 3.0, 0.03855459744859336, -0.07536948707088668),
    (0.0, -0.001, 1.0011293799198486, 0.0),
    (0.0, -0.5, 1.952360489182557, 0.0),
    (0.0, -2.0, 108.94090438997797, 0.0),
    (0.0, -8.0, 1.2470298161623233e+28, 0.0),
    (0.5, -0.001, 0.7794506266311085, 0.4797047779857289),
    (0.5, -0.5, 1.2220084158685705, 1.1893393085928645),
    (0.5, -2.0, -35.63530351200189, 77.38014237534543),
    (0.5, -8.0, -1.413078573475713e+27, 9.608526563189276e+27),
    (2.0, -0.001, 0.018083784988234364, 0.34009935607428),
    (2.0, -0.5, -0.12293249482276238, 0.32755513633331257),
    (2.0, -2.0, -0.4389528271292429, 2.1098962103309815),
    (2.0, -8.0, 1.905378484910617e+26, 1.2594666898390968e+26),
    (6.0, -0.001, -1.6375340027141455e-05, 0.09539620611327662),
    (6.0, -0.5, -0.008124885586461947, 0.09468791486012608),
    (6.0, -2.0, -0.0291701442903109, 0.0852596706015393),
    (6.0, -8.0, -521897623801.07513, 2845041450369.9263),
    (1.99146684283, -1.35481012811, -2.2637964826927196e-12, -4.3776652032696615e-12),
    (1.9, -1.3, -0.09274226408463683, -0.0948308147618205),
];

/// The lower half-plane is reached by `w(z) = 2exp(-z^2) - w(-z)`, whose relative
/// accuracy degrades near the zeros of `w` (all of which live there) as the two terms
/// cancel. Absolute accuracy survives, which is the standard guarantee. Poppe & Wijers
/// state the same for Algorithm 680.
fn is_near_a_zero(x: f64, y: f64) -> bool {
    y < 0.0 && (x - 1.99146684283).abs() < 0.2 && (y + 1.35481012811).abs() < 0.2
}

#[test]
fn faddeeva_w_reference_best() {
    let mut worst = 0.0f64;
    let mut worst_at = (0.0, 0.0);

    for &(x, y, wr, wi) in &REF {
        let got = c(x, y).faddeeva_w_p::<Precision>();

        if is_near_a_zero(x, y) {
            let (re, im) = parts(got);
            assert!(
                (re - wr).abs() < 1e-14 && (im - wi).abs() < 1e-14,
                "w({x} + {y}i) = ({re}, {im}), want ({wr}, {wi})"
            );
            continue;
        }

        let e = rel(got, (wr, wi));
        if e > worst {
            worst = e;
            worst_at = (x, y);
        }
    }

    assert!(worst < 1e-12, "worst relative error {worst:e} at z = {worst_at:?}");
}

#[test]
fn faddeeva_w_reference_default_policy() {
    // Performance maps to Average -> N = 24, measured at 4.2e-10.
    for &(x, y, wr, wi) in &REF {
        if is_near_a_zero(x, y) {
            continue;
        }

        let e = rel(c(x, y).faddeeva_w_p::<Performance>(), (wr, wi));
        assert!(e < 1e-8, "w({x} + {y}i) relative error {e:e}");
    }
}

#[test]
fn faddeeva_w_reference_ultra_performance() {
    // UltraPerformance maps to Worst -> N = 8, measured at 3.1e-4.
    for &(x, y, wr, wi) in &REF {
        if is_near_a_zero(x, y) {
            continue;
        }

        let e = rel(c(x, y).faddeeva_w_p::<UltraPerformance>(), (wr, wi));
        assert!(e < 5e-3, "w({x} + {y}i) relative error {e:e}");
    }
}

#[test]
fn w_of_zero_is_one() {
    // Not exact, since the origin is an ordinary point of the approximation rather than a
    // special case, so this holds only to the tier's accuracy. Im w(0) *is* exact: the
    // polynomial has real coefficients and Z is real at z = 0, so nothing ever writes a
    // non-zero imaginary part.
    let (re, im) = parts(c(0.0, 0.0).faddeeva_w_p::<Precision>());
    assert!((re - 1.0).abs() < 1e-12, "Re w(0) = {re}");
    assert_eq!(im, 0.0, "Im w(0) = {im}");

    let (re, _) = parts(c(0.0, 0.0).faddeeva_w_p::<Reference>());
    assert!((re - 1.0).abs() < 1e-15, "Re w(0) at Reference = {re}");
}

#[test]
fn imaginary_axis_is_erfcx_of_a_real() {
    // w(iy) = erfcx(y) = e^{y^2} erfc(y), real for real y.
    for y in [0.25f64, 1.0, 4.0, 20.0] {
        let (re, im) = parts(c(0.0, y).faddeeva_w_p::<Precision>());
        let want = libm::exp(y * y) * libm::erfc(y);

        assert!((re - want).abs() <= 1e-12 * want, "w({y}i) re = {re}, want {want}");
        assert!(im.abs() < 1e-15, "w({y}i) should be real, im = {im}");
    }
}

#[test]
fn erfcx_matches_w_of_iz() {
    // erfcx(z) = w(iz), by definition, so this pins the argument rotation.
    for (x, y) in [(0.5f64, 0.25f64), (3.0, -1.5), (-2.0, 0.75)] {
        let a = c(x, y).erfcx_p::<Precision>();
        let b = c(-y, x).faddeeva_w_p::<Precision>();

        assert_eq!(parts(a), parts(b), "erfcx({x} + {y}i)");
    }
}

#[test]
fn conjugate_symmetry_across_the_imaginary_axis() {
    // w(-conj z) = conj(w(z)): Re w is even in x, Im w is odd in x.
    for (x, y) in [(1.5f64, 0.5f64), (7.0, 0.01), (0.25, 3.0)] {
        let (ar, ai) = parts(c(-x, y).faddeeva_w_p::<Precision>());
        let (br, bi) = parts(c(x, y).faddeeva_w_p::<Precision>());

        assert!((ar - br).abs() <= 1e-15 * br.abs().max(1.0), "Re w not even in x");
        assert!((ai + bi).abs() <= 1e-15 * bi.abs().max(1.0), "Im w not odd in x");
    }
}

#[test]
fn reflection_holds_across_the_real_axis() {
    // w(z) + w(-z) = 2 e^{-z^2}, the identity the lower half-plane is built on.
    for (x, y) in [(1.0f64, 0.5f64), (3.0, 2.0), (0.25, 4.0)] {
        let (ar, ai) = parts(c(x, -y).faddeeva_w_p::<Precision>());
        let (br, bi) = parts(c(-x, y).faddeeva_w_p::<Precision>());

        // z = x - iy, so -z^2 = y^2 - x^2 + 2ixy
        let m = 2.0 * libm::exp(y * y - x * x);
        let (wr, wi) = (m * libm::cos(2.0 * x * y), m * libm::sin(2.0 * x * y));

        let scale = m.max(1.0);
        assert!((ar + br - wr).abs() <= 1e-13 * scale, "re at {x} - {y}i");
        assert!((ai + bi - wi).abs() <= 1e-13 * scale, "im at {x} - {y}i");
    }
}

#[test]
fn re_is_exp_neg_x_squared_on_the_real_axis() {
    // On the axis every (iy)^n term of the correction vanishes, so `Re w` reduces to
    // its seed and is *bit-exact* `exp(-x^2)`, not merely accurate to the tier. This
    // is the sharpest statement of what the near-axis path buys.
    for x in [0.5f64, 1.0, 2.0, 3.0, 6.0, 12.0, 25.0] {
        let (re, _) = parts(c(x, 0.0).faddeeva_w_p::<Precision>());

        assert_eq!(re, libm::exp(-x * x), "Re w({x}) should be exactly exp(-x^2)");
    }

    // Outside the gate it falls back to the direct evaluation, which cannot represent
    // exp(-x^2) at all next to |w| - by x = 1000 the true value is 1e-434000 and the
    // computed one is pure roundoff. Documented, not a regression.
    let (re, _) = parts(c(1500.0, 0.0).faddeeva_w_p::<Precision>());
    assert!(re.abs() < 1e-15, "far past the gate Re w should be roundoff, got {re}");
}

/// The near-real-axis correction, which is the whole reason `Re w` is usable at small
/// `y`. Reference values from mpmath at 50 digits.
///
/// `Performance` does not get the correction (it is gated at `Best`), so it doubles as
/// the control: the same points, the same code, without the fix.
#[test]
fn real_axis_correction_recovers_the_real_part() {
    #[rustfmt::skip]
    let refs: [(f64, f64, f64); 8] = [
        // (x, y, Re w)
        (4.0,    1e-8, 1.1292767024031586e-07),
        (10.0,   1e-8, 5.728717562239308e-11),
        (10.0,   1e-6, 5.728717562239248e-09),
        (100.0,  1e-8, 5.642742331498062e-13),
        (300.0,  1e-8, 6.268877632985457e-14),
        // just inside REAL_AXIS_X = 1e3; the boundary itself is pinned separately below
        (999.0,  1e-8, 5.653205072233062e-15),
        (2.0,    1e-7, 0.018315662061303015),
        (6.0,    1e-9, 1.6375572486099643e-11),
    ];

    for &(x, y, want) in &refs {
        let good = parts(c(x, y).faddeeva_w_p::<Precision>()).0;
        let ctrl = parts(c(x, y).faddeeva_w_p::<Performance>()).0;

        let (e_good, e_ctrl) = ((good - want).abs() / want.abs(), (ctrl - want).abs() / want.abs());

        // The bound is the tier's, not the correction's: the seed's imaginary part comes
        // from the same N-term evaluation, so `Best` (N=32) lands at ~1e-9 in the worst
        // case here while `Reference` (N=40) reaches 2e-10.
        assert!(e_good < 1e-8, "Re w({x} + {y}i) = {good}, want {want}, rel {e_good:e}");
        assert!(
            (parts(c(x, y).faddeeva_w_p::<Reference>()).0 - want).abs() / want.abs() < 1e-9,
            "Re w({x} + {y}i) at Reference"
        );

        // The correction must actually be doing something, not merely not hurting.
        assert!(
            e_good < e_ctrl,
            "correction did not improve Re w({x} + {y}i): {e_good:e} vs uncorrected {e_ctrl:e}"
        );
    }
}

/// Outside the gate the correction must be off, and `w` must stay continuous across the
/// boundary, since a seam there would show up as a discontinuity in a Voigt profile.
#[test]
fn correction_is_continuous_across_its_gate() {
    // REAL_AXIS_Y is 1e-5 for f64; step across it and across REAL_AXIS_X = 1e3.
    for x in [10.0f64, 100.0, 999.0] {
        let below = parts(c(x, 9.99e-6).faddeeva_w_p::<Precision>());
        let above = parts(c(x, 1.001e-5).faddeeva_w_p::<Precision>());

        let scale = (above.0 * above.0 + above.1 * above.1).sqrt();
        assert!(
            (below.0 - above.0).abs() <= 1e-8 * scale && (below.1 - above.1).abs() <= 1e-8 * scale,
            "seam in y at x={x}: {below:?} vs {above:?}"
        );
    }

    for y in [1e-8f64, 1e-6] {
        let inside = parts(c(999.0, y).faddeeva_w_p::<Precision>());
        let outside = parts(c(1001.0, y).faddeeva_w_p::<Precision>());

        // Different x, so compare against how much w itself should change: ~0.2%.
        let rel = (inside.1 - outside.1).abs() / outside.1.abs();
        assert!(rel < 5e-3, "seam in x at y={y}: {inside:?} vs {outside:?}, rel {rel:e}");
    }

    // The x bound is deliberate rather than incidental: past it the correction's own
    // error (which grows like x^2) overtakes the direct evaluation's, so `Re w` at
    // x = 1000 is expected to be the *uncorrected* value.
    let at_bound = parts(c(1000.0, 1e-8).faddeeva_w_p::<Precision>()).0;
    let want = 5.6419042983424735e-15;
    assert!(
        (at_bound - want).abs() / want > 1e-6,
        "x = REAL_AXIS_X should fall outside the gate, but Re w looks corrected: {at_bound}"
    );
}

#[test]
fn im_holds_relative_accuracy_where_re_does_not() {
    // The two halves of the documented caveat, on the same points. Far out on the real
    // axis `Im w` *is* `|w|`, so it keeps the tier's full relative accuracy, while
    // `Re w = exp(-x^2)` has fallen so far below `|w|` that the normwise error swamps
    // it completely. By x = 15, `Re w` is 1e-98 and not even the sign is meaningful.
    for x in [8.0f64, 15.0, 50.0, 1000.0] {
        let &(_, _, want_re, want_im) = REF.iter().find(|r| r.0 == x && r.1 == 0.0).unwrap();
        let (re, im) = parts(c(x, 0.0).faddeeva_w_p::<Precision>());

        assert!(
            (im - want_im).abs() <= 1e-12 * want_im.abs(),
            "Im w({x}) = {im}, want {want_im}"
        );

        // Absolute, not relative: this is the part the method does not promise.
        assert!(
            (re - want_re).abs() <= 1e-12 * want_im.abs(),
            "Re w({x}) = {re} is not even absolutely close to {want_re}"
        );
    }
}

/// `w` satisfies `w'(z) = -2z w(z) + 2i/sqrt(pi)`.
///
/// Two things at once: it exercises the `Complex<Dual<..>>` impl, which nothing else
/// reaches, and it is an independent check on `w` itself: the ODE is a property of the
/// function, and the approximation is under no obligation to satisfy it. That it does,
/// to 1e-11, says the derivative of the rational form tracks the derivative of `w` and
/// not merely its value.
#[cfg(feature = "dual")]
#[test]
fn satisfies_its_differential_equation_through_dual() {
    use thermite_dual::Dual;

    type D = Dual<V, 1>;
    type CD = Complex<D>;

    for &(x, y) in &[(0.8f64, 0.6f64), (2.5, 0.1), (-1.5, 3.0), (0.0, 0.0), (6.0, 0.25)] {
        let z: CD = Complex::new(Dual::variable(V::splat(x), 0), Dual::constant(V::splat(y)));
        let w = z.faddeeva_w_p::<Precision>();

        let got = (w.re.dual[0].extract::<0>(), w.im.dual[0].extract::<0>());

        // -2z*w + 2i/sqrt(pi), with w the value part
        let (wr, wi) = (w.re.value().extract::<0>(), w.im.value().extract::<0>());
        let want = (
            -2.0 * (x * wr - y * wi),
            -2.0 * (x * wi + y * wr) + 2.0 / core::f64::consts::PI.sqrt(),
        );

        let scale = (want.0 * want.0 + want.1 * want.1).sqrt().max(1.0);
        assert!(
            (got.0 - want.0).abs() <= 1e-11 * scale && (got.1 - want.1).abs() <= 1e-11 * scale,
            "w'({x} + {y}i): got {got:?}, want {want:?}"
        );
    }
}

#[test]
fn f32_clamps_to_its_own_ladder() {
    // f32 caps at N = 16 (~5 ulp) whatever the policy asks for, because f32 Horner
    // roundoff floors there. So `Precision` and `Performance` must agree exactly, and
    // both must land within the f32 tier's bound rather than the f64 one.
    type W = Vector<f32>;
    type CF = Complex<W>;

    let f = |re: f32, im: f32| Complex::new(W::splat(re), W::splat(im));
    let parts32 = |z: CF| (z.re.extract::<0>(), z.im.extract::<0>());

    for &(x, y, wr, wi) in &REF {
        // Skip what f32 cannot represent at all: the reflection overflows well before
        // f64's does, and the tiny-Re-w points are meaningless here.
        if y < -8.0 || x.abs() > 1e18 || (y * y - x * x) > 80.0 || is_near_a_zero(x, y) {
            continue;
        }

        let (re, im) = parts32(f(x as f32, y as f32).faddeeva_w_p::<Precision>());
        let n = (wr * wr + wi * wi).sqrt();
        let d = (((re as f64) - wr).powi(2) + ((im as f64) - wi).powi(2)).sqrt();

        assert!(d <= 1e-5 * n, "f32 w({x} + {y}i) relative error {:e}", d / n);

        // Identical, not merely close: both policies resolve to the same N. Only outside
        // the near-real-axis box, though, because the correction is gated at `Best`, so inside
        // it the two policies deliberately run different code.
        if y.abs() >= 1e-3 || x.abs() >= 1e2 {
            assert_eq!(parts32(f(x as f32, y as f32).faddeeva_w_p::<Performance>()), (re, im));
        }
    }
}

#[test]
fn lanes_are_independent() {
    // Alternating half-planes: the reflection is masked, so a lane must not be able to
    // pull its neighbour across the real axis with it.
    let elems: Vec<num_complex::Complex<f64>> = (0..C::LANES)
        .map(|i| num_complex::Complex::new(0.5 + i as f64, if i % 2 == 0 { 1.0 } else { -1.0 }))
        .collect();

    let z = unsafe { C::load_unaligned(elems.as_ptr().cast()) };
    let got = z.faddeeva_w_p::<Precision>();

    for (i, e) in elems.iter().enumerate() {
        let (x, y) = (e.re, e.im);
        let (wr, wi) = parts(c(x, y).faddeeva_w_p::<Precision>());
        let lane = got.extractv(i);

        assert!(
            (lane.re - wr).abs() <= 1e-15 * wr.abs().max(1.0) && (lane.im - wi).abs() <= 1e-15 * wi.abs().max(1.0),
            "lane {i} at {x} + {y}i: got ({}, {}), want ({wr}, {wi})",
            lane.re,
            lane.im
        );
    }
}

/// The Voigt function `K(x, y) = Re w(x + iy)`, against its defining convolution
/// integral rather than against `w`, an oracle that shares no code with anything here:
///
///   K(x, y) = (y/pi) * integral exp(-t^2) / ((x - t)^2 + y^2) dt
///
/// mpmath's adaptive quadrature at 50 digits, which agrees with `Re w` to 1e-50.
///
/// The per-point tolerances are the behaviour, not slack. Inside the correction gate
/// (`y < 1e-5`, `|x| < 1e3`) `K` holds the tier's full relative accuracy. Outside it,
/// `K` is `Re w` from the direct evaluation and follows `eps * |w| / K`, which is
/// `1` wherever `K` is of order `|w|`, and grows as `K` falls into the Lorentz wing.
#[test]
fn voigt_matches_the_convolution_integral() {
    #[rustfmt::skip]
    let refs: [(f64, f64, f64, f64); 7] = [
        // x,   y,      K(x, y),                 tolerance
        (0.0,   1.0,    0.427583576155807,       1e-13), // K == |w|, no loss
        (1.0,   0.5,    0.3549003328675779,      1e-12),
        (2.5,   0.1,    0.014698406828789557,    1e-10),
        (5.0,   0.01,   0.00024080339195117517,  1e-9),  // K/|w| ~ 2e-3
        (0.5,   1e-4,   0.7787358415658242,      1e-13), // x small, K still ~ |w|
        (3.0,   1e-6,   0.00012348836881971166,  1e-12), // inside the gate: full accuracy
        (8.0,   1e-3,   9.030620666703193e-06,   1e-8),  // K/|w| ~ 1e-4
    ];

    for &(x, y, want, tol) in &refs {
        let got = c(x, y).voigt_p::<Precision>().extract::<0>();

        assert!(
            (got - want).abs() <= tol * want.abs(),
            "K({x}, {y}) = {got}, want {want}, rel {:e} > {tol:e}",
            (got - want).abs() / want.abs()
        );
    }
}

/// The far wing at vanishing `y`, which is what the correction exists for and where
/// every rational Voigt approximation in the literature fails.
///
/// Note how fast the Lorentz term takes over: at `x = 12`, `K(x, 0)` is `exp(-144) =
/// 2.9e-63`, but by `y = 1e-30` it is already `4.0e-33`, thirty orders larger, and
/// entirely `y/(sqrt(pi)x^2)`. Getting both regimes right at once is the whole trick,
/// and neither is representable as a perturbation of the other.
///
/// Reference values from mpmath at 60 digits.
#[test]
fn voigt_in_the_far_wing() {
    #[rustfmt::skip]
    let refs: [(f64, f64, f64); 16] = [
        (3.0,  0.0,   0.00012340980408667956),
        (3.0,  1e-30, 0.00012340980408667956),
        (3.0,  1e-12, 0.0001234098041652443),
        (3.0,  1e-8,  0.00012341058973403064),
        (6.0,  0.0,   2.3195228302435696e-16),
        (6.0,  1e-30, 2.3195228302435696e-16),
        (6.0,  1e-12, 1.6607292816840975e-14),
        (6.0,  1e-8,  1.6375363729044922e-10),
        (12.0, 0.0,   2.8946403116483003e-63),
        (12.0, 1e-30, 3.959521872939645e-33),
        (12.0, 1e-12, 3.959521872939645e-15),
        (12.0, 1e-8,  3.9595218729396455e-11),
        (25.0, 0.0,   3.6808558548018004e-272),
        (25.0, 1e-30, 9.048785365110864e-34),
        (25.0, 1e-12, 9.048785365110862e-16),
        (25.0, 1e-8,  9.048785365110863e-12),
    ];

    for &(x, y, want) in &refs {
        let got = c(x, y).voigt_p::<Precision>().extract::<0>();

        assert!(
            (got - want).abs() <= 1e-9 * want,
            "K({x}, {y}) = {got}, want {want}, rel {:e}",
            (got - want).abs() / want
        );
    }
}

/// `K` is even in `x` and strictly positive for `y > 0`, both by construction from the
/// convolution. The second is the one that catches a broken `Re w`: the classic failure
/// of a rational Voigt approximation at small `y` is to return a *negative* value.
#[test]
fn voigt_is_even_and_positive() {
    for y in [1e-8f64, 1e-6, 1e-3, 1.0] {
        for x in [0.0f64, 0.5, 3.0, 7.0, 20.0, 100.0] {
            let a = c(x, y).voigt_p::<Precision>().extract::<0>();
            let b = c(-x, y).voigt_p::<Precision>().extract::<0>();

            assert!(a > 0.0, "K({x}, {y}) = {a} is not positive");
            assert_eq!(a, b, "K is not even in x at ({x}, {y})");
        }
    }
}

/// The generated default-policy forms exist and agree with the explicit `Performance`
/// ones, which is what `decl_complex_math!` promises: `faddeeva_w()` is
/// `faddeeva_w_p::<DefaultPolicy>()`, exactly as `norm()` is `norm_p::<DefaultPolicy>()`.
#[test]
fn default_policy_forms_match_the_explicit_ones() {
    for (x, y) in [(0.5f64, 0.25f64), (3.0, -1.5), (-2.0, 0.75), (7.0, 1e-7)] {
        assert_eq!(
            parts(c(x, y).faddeeva_w()),
            parts(c(x, y).faddeeva_w_p::<Performance>())
        );
        assert_eq!(parts(c(x, y).erfcx()), parts(c(x, y).erfcx_p::<Performance>()));
        assert_eq!(
            c(x, y).voigt().extract::<0>(),
            c(x, y).voigt_p::<Performance>().extract::<0>()
        );
    }
}

/// Generic code can bound on `ComplexSpecialMath` the way it bounds on `ComplexMath`.
#[test]
fn is_usable_as_a_generic_bound() {
    fn line_shape<T: ComplexSpecialMath + Copy>(z: T) -> T::Real {
        z.voigt()
    }

    assert_eq!(
        line_shape(c(2.0, 0.1)).extract::<0>(),
        c(2.0, 0.1).voigt().extract::<0>()
    );
}

/// The f32 correction gate (`y < 1e-3`, `|x| < 1e2`) is set by analogy with f64's rather
/// than derived, so this pins that it is at least never a regression. Measured against a
/// 50-digit oracle, the correction wins by one to four orders at every point inside it.
#[test]
fn f32_real_axis_correction_is_never_a_regression() {
    type W = Vector<f32>;
    let f = |re: f32, im: f32| Complex::new(W::splat(re), W::splat(im));

    #[rustfmt::skip]
    let refs: [(f32, f32, f64); 6] = [
        (2.0,  1e-4, 0.018338810176746257),
        (4.0,  1e-4, 4.037490347118378e-06),
        (6.0,  1e-5, 1.6375340556961204e-07),
        (10.0, 1e-4, 5.728717561645333e-07),
        (20.0, 1e-3, 1.4157965831846981e-06),
        (99.0, 1e-4, 5.757330398098576e-09),
    ];

    for &(x, y, want) in &refs {
        let good = f(x, y).voigt_p::<Precision>().extract::<0>() as f64;
        let ctrl = f(x, y).voigt_p::<Performance>().extract::<0>() as f64;

        let (e_good, e_ctrl) = ((good - want).abs() / want, (ctrl - want).abs() / want);

        assert!(
            e_good <= e_ctrl,
            "f32 correction regressed K({x}, {y}): {e_good:e} vs {e_ctrl:e}"
        );
        assert!(e_good < 1e-2, "f32 K({x}, {y}) rel {e_good:e}");
    }
}
