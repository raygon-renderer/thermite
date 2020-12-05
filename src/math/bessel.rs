use crate::*;

pub use bessel_internal::*;

#[dispatch(thermite = "crate")]
mod bessel_internal {
    use crate::*;

    #[inline(always)]
    pub fn bessel_j<S: Simd, E: SimdVectorizedMathInternal<S>, P: Policy>(x: E::Vf, n: u32) -> E::Vf {
        match n {
            0 => bessel_j0::<S, E, P>(x),
            1 => bessel_j1::<S, E, P>(x),
            _ => unimplemented!(),
        }
    }

    #[inline(always)]
    pub fn bessel_y<S: Simd, E: SimdVectorizedMathInternal<S>, P: Policy>(x: E::Vf, n: u32) -> E::Vf {
        match n {
            0 => bessel_y0::<S, E, P>(x),
            _ => unimplemented!(),
        }
    }

    #[inline(always)]
    pub fn bessel_j0<S: Simd, E: SimdVectorizedMathInternal<S>, P: Policy>(mut x: E::Vf) -> E::Vf {
        x = x.abs(); // even function

        let x1 = E::Vf::splat_as(2.4048255576957727686e+00);
        let x2 = E::Vf::splat_as(5.5200781102863106496e+00);
        let x11 = E::Vf::splat_as(6.160e+02);
        let x12 = E::Vf::splat_as(-1.42444230422723137837e-03);
        let x21 = E::Vf::splat_as(1.4130e+03);
        let x22 = E::Vf::splat_as(5.46860286310649596604e-04);
        let den = E::Vf::splat_as(-1.0 / 256.0);

        let y00 = E::Vf::one();
        let mut y04 = unsafe { E::Vf::undefined() };
        let mut y48 = unsafe { E::Vf::undefined() };
        let mut y8i = unsafe { E::Vf::undefined() };

        let le4 = x.le(E::Vf::splat_as(4.0));
        let le8 = x.le(E::Vf::splat_as(8.0));

        // between 4 and 8
        // if le4 AND le8, then be48 is false. If (NOT le4) AND le8, then be48 is true
        // a value cannot be le4 AND (NOT le8)
        let be48 = le4 ^ le8;

        // 0 < x <= 4
        if P::POLICY.avoid_branching || le4.any() {
            let r = (x * x).poly_rational_p::<P>(
                &[
                    E::cast_from(-4.1298668500990866786e+11f64),
                    E::cast_from(2.7282507878605942706e+10f64),
                    E::cast_from(-6.2140700423540120665e+08f64),
                    E::cast_from(6.6302997904833794242e+06f64),
                    E::cast_from(-3.6629814655107086448e+04f64),
                    E::cast_from(1.0344222815443188943e+02f64),
                    E::cast_from(-1.2117036164593528341e-01f64),
                ],
                &[
                    E::cast_from(2.3883787996332290397e+12f64),
                    E::cast_from(2.6328198300859648632e+10f64),
                    E::cast_from(1.3985097372263433271e+08f64),
                    E::cast_from(4.5612696224219938200e+05f64),
                    E::cast_from(9.3614022392337710626e+02f64),
                    E::cast_from(1.0f64),
                    E::cast_from(0.0f64),
                ],
            );

            y04 = (x + x1) * (den.mul_adde(x11, x) - x12) * r;
        };

        // 4 < x <= 8
        if P::POLICY.avoid_branching || be48.any() {
            let y = E::Vf::splat_as(-1.0 / 64.0).mul_adde(x * x, E::Vf::one());

            // y <= 1 given above operation
            let r = y.poly_p::<P>(&[
                E::cast_from(-1.8319397969392084011e+03f64),
                E::cast_from(-1.2254078161378989535e+04f64),
                E::cast_from(-7.2879702464464618998e+03f64),
                E::cast_from(1.0341910641583726701e+04f64),
                E::cast_from(1.1725046279757103576e+04f64),
                E::cast_from(4.4176707025325087628e+03f64),
                E::cast_from(7.4321196680624245801e+02f64),
                E::cast_from(4.8591703355916499363e+01f64),
            ]) / y.poly_p::<P>(&[
                E::cast_from(-3.5783478026152301072e+05f64),
                E::cast_from(2.4599102262586308984e+05f64),
                E::cast_from(-8.4055062591169562211e+04f64),
                E::cast_from(1.8680990008359188352e+04f64),
                E::cast_from(-2.9458766545509337327e+03f64),
                E::cast_from(3.3307310774649071172e+02f64),
                E::cast_from(-2.5258076240801555057e+01f64),
                E::cast_from(1.0f64),
            ]);

            y48 = den.mul_adde(x21, x) * (x + x2) * r;
        }

        // 8 < x <= inf
        if P::POLICY.avoid_branching || !le8.all() {
            // 1 / sqrt(pi)
            let frac_one_sqrt_pi = E::Vf::splat_as(5.641895835477562869480794515607725858e-01f64);

            let factor = frac_one_sqrt_pi / x.sqrt();

            let y = E::Vf::splat_as(8.0) / x;
            let y2 = y * y;

            // y2 <= 1.0 given above division, no need to do poly_rational
            let rc = y2.poly_p::<P>(&[
                E::cast_from(2.2779090197304684302e+04f64),
                E::cast_from(4.1345386639580765797e+04f64),
                E::cast_from(2.1170523380864944322e+04f64),
                E::cast_from(3.4806486443249270347e+03f64),
                E::cast_from(1.5376201909008354296e+02f64),
                E::cast_from(8.8961548424210455236e-01f64),
            ]) / y2.poly_p::<P>(&[
                E::cast_from(2.2779090197304684318e+04f64),
                E::cast_from(4.1370412495510416640e+04f64),
                E::cast_from(2.1215350561880115730e+04f64),
                E::cast_from(3.5028735138235608207e+03f64),
                E::cast_from(1.5711159858080893649e+02f64),
                E::cast_from(1.0f64),
            ]);

            let rs = y2.poly_p::<P>(&[
                E::cast_from(-8.9226600200800094098e+01f64),
                E::cast_from(-1.8591953644342993800e+02f64),
                E::cast_from(-1.1183429920482737611e+02f64),
                E::cast_from(-2.2300261666214198472e+01f64),
                E::cast_from(-1.2441026745835638459e+00f64),
                E::cast_from(-8.8033303048680751817e-03f64),
            ]) / y2.poly_p::<P>(&[
                E::cast_from(5.7105024128512061905e+03f64),
                E::cast_from(1.1951131543434613647e+04f64),
                E::cast_from(7.2642780169211018836e+03f64),
                E::cast_from(1.4887231232283756582e+03f64),
                E::cast_from(9.0593769594993125859e+01f64),
                E::cast_from(1.0f64),
            ]);

            let (sx, cx) = x.sin_cos_p::<P>();

            y8i = factor * rc.mul_sube(sx + cx, y * rs * (sx - cx));
        }

        let mut y = y8i;
        y = le8.select(y48, y);
        y = le4.select(y04, y);
        y = x.eq(E::Vf::zero()).select(y00, y);

        y
    }

    #[inline(always)]
    pub fn bessel_j1<S: Simd, E: SimdVectorizedMathInternal<S>, P: Policy>(x: E::Vf) -> E::Vf {
        let w = x.abs();

        let x1 = E::Vf::splat_as(3.8317059702075123156e+00f64);
        let x2 = E::Vf::splat_as(7.0155866698156187535e+00f64);
        let x11 = E::Vf::splat_as(9.810e+02f64);
        let x12 = E::Vf::splat_as(-3.2527979248768438556e-04f64);
        let x21 = E::Vf::splat_as(1.7960e+03f64);
        let x22 = E::Vf::splat_as(-3.8330184381246462950e-05f64);
        let den = E::Vf::splat_as(-1.0 / 256.0);

        let y00 = E::Vf::zero();
        let mut y04 = unsafe { E::Vf::undefined() };
        let mut y48 = unsafe { E::Vf::undefined() };
        let mut y8i = unsafe { E::Vf::undefined() };

        let le4 = w.le(E::Vf::splat_as(4.0));
        let le8 = w.le(E::Vf::splat_as(8.0));

        // between 4 and 8
        // if le4 AND le8, then be48 is false. If (NOT le4) AND le8, then be48 is true
        // a value cannot be le4 AND (NOT le8)
        let be48 = le4 ^ le8;

        // 0 < w <= 4
        if P::POLICY.avoid_branching || le4.any() {
            let r = (x * x).poly_rational_p::<P>(
                &[
                    E::cast_from(-1.4258509801366645672e+11f64),
                    E::cast_from(6.6781041261492395835e+09f64),
                    E::cast_from(-1.1548696764841276794e+08f64),
                    E::cast_from(9.8062904098958257677e+05f64),
                    E::cast_from(-4.4615792982775076130e+03f64),
                    E::cast_from(1.0650724020080236441e+01f64),
                    E::cast_from(-1.0767857011487300348e-02f64),
                ],
                &[
                    E::cast_from(4.1868604460820175290e+12f64),
                    E::cast_from(4.2091902282580133541e+10f64),
                    E::cast_from(2.0228375140097033958e+08f64),
                    E::cast_from(5.9117614494174794095e+05f64),
                    E::cast_from(1.0742272239517380498e+03f64),
                    E::cast_from(1.0f64),
                    E::cast_from(0.0f64),
                ],
            );

            y04 = r * w * (w + x1) * (den.mul_adde(x11, w) - x12);
        }

        // 4 < w <= 8
        if P::POLICY.avoid_branching || be48.any() {
            let r = if P::POLICY.extra_precision {
                let y = E::Vf::one() / (x * x);

                // reverse coefficients and evaluate at 1/(x*x),
                // this preserves precision by ensuring powers of
                // x are between 0 and 1. Since x > 4 here, do it
                // unconditionally
                y.poly_p::<P>(&[
                    E::cast_from(4.6179191852758252278e+00f64),
                    E::cast_from(-7.5023342220781607561e+03f64),
                    E::cast_from(5.0793266148011179143e+06f64),
                    E::cast_from(-1.8113931269860667829e+09f64),
                    E::cast_from(3.5580665670910619166e+11f64),
                    E::cast_from(-3.6658018905416665164e+13f64),
                    E::cast_from(1.6608531731299018674e+15f64),
                    E::cast_from(-1.7527881995806511112e+16f64),
                ]) / y.poly_p::<P>(&[
                    E::cast_from(1.0f64),
                    E::cast_from(1.3886978985861357615e+03f64),
                    E::cast_from(1.1267125065029138050e+06f64),
                    E::cast_from(6.4872502899596389593e+08f64),
                    E::cast_from(2.7622777286244082666e+11f64),
                    E::cast_from(8.4899346165481429307e+13f64),
                    E::cast_from(1.7128800897135812012e+16f64),
                    E::cast_from(1.7253905888447681194e+18f64),
                ])
            } else {
                let y = x * x;

                y.poly_p::<P>(&[
                    E::cast_from(-1.7527881995806511112e+16f64),
                    E::cast_from(1.6608531731299018674e+15f64),
                    E::cast_from(-3.6658018905416665164e+13f64),
                    E::cast_from(3.5580665670910619166e+11f64),
                    E::cast_from(-1.8113931269860667829e+09f64),
                    E::cast_from(5.0793266148011179143e+06f64),
                    E::cast_from(-7.5023342220781607561e+03f64),
                    E::cast_from(4.6179191852758252278e+00f64),
                ]) / y.poly_p::<P>(&[
                    E::cast_from(1.7253905888447681194e+18f64),
                    E::cast_from(1.7128800897135812012e+16f64),
                    E::cast_from(8.4899346165481429307e+13f64),
                    E::cast_from(2.7622777286244082666e+11f64),
                    E::cast_from(6.4872502899596389593e+08f64),
                    E::cast_from(1.1267125065029138050e+06f64),
                    E::cast_from(1.3886978985861357615e+03f64),
                    E::cast_from(1.0f64),
                ]);
            };

            y48 = r * w * (w + x2) * (den.mul_adde(x21, w) - x22);
        }

        // 8 < w <= inf
        if P::POLICY.avoid_branching || !le8.all() {
            // 1 / sqrt(pi)
            let frac_one_sqrt_pi = E::Vf::splat_as(5.641895835477562869480794515607725858e-01f64);

            let y = E::Vf::splat_as(8.0) / w;
            let y2 = y * y;

            let factor = frac_one_sqrt_pi / w.sqrt();

            // y2 <= 1.0 given above division, no need for poly_rational
            let rc = y2.poly_p::<P>(&[
                E::cast_from(-4.4357578167941278571e+06f64),
                E::cast_from(-9.9422465050776411957e+06f64),
                E::cast_from(-6.6033732483649391093e+06f64),
                E::cast_from(-1.5235293511811373833e+06f64),
                E::cast_from(-1.0982405543459346727e+05f64),
                E::cast_from(-1.6116166443246101165e+03f64),
                E::cast_from(0.0f64),
            ]) / y2.poly_p::<P>(&[
                E::cast_from(-4.4357578167941278568e+06f64),
                E::cast_from(-9.9341243899345856590e+06f64),
                E::cast_from(-6.5853394797230870728e+06f64),
                E::cast_from(-1.5118095066341608816e+06f64),
                E::cast_from(-1.0726385991103820119e+05f64),
                E::cast_from(-1.4550094401904961825e+03f64),
                E::cast_from(1.0f64),
            ]);
            let rs = y2.poly_p::<P>(&[
                E::cast_from(3.3220913409857223519e+04f64),
                E::cast_from(8.5145160675335701966e+04f64),
                E::cast_from(6.6178836581270835179e+04f64),
                E::cast_from(1.8494262873223866797e+04f64),
                E::cast_from(1.7063754290207680021e+03f64),
                E::cast_from(3.5265133846636032186e+01f64),
                E::cast_from(0.0f64),
            ]) / y2.poly_p::<P>(&[
                E::cast_from(7.0871281941028743574e+05f64),
                E::cast_from(1.8194580422439972989e+06f64),
                E::cast_from(1.4194606696037208929e+06f64),
                E::cast_from(4.0029443582266975117e+05f64),
                E::cast_from(3.7890229745772202641e+04f64),
                E::cast_from(8.6383677696049909675e+02f64),
                E::cast_from(1.0f64),
            ]);

            // in the original, this used x instead of w, but that produced incorrect results
            let (sx, cx) = w.sin_cos_p::<P>();

            y8i = factor * rc.mul_adde(sx - cx, y * rs * (sx + cx));
        }

        let mut y = y8i;
        y = le8.select(y48, y);
        y = le4.select(y04, y);
        y = x.eq(E::Vf::zero()).select(y00, y);

        y.combine_sign(x)
    }

    pub fn bessel_y0<S: Simd, E: SimdVectorizedMathInternal<S>, P: Policy>(x: E::Vf) -> E::Vf {
        let x1 = E::Vf::splat_as(8.9357696627916752158e-01f64);
        let x2 = E::Vf::splat_as(3.9576784193148578684e+00f64);
        let x3 = E::Vf::splat_as(7.0860510603017726976e+00f64);
        let x11 = E::Vf::splat_as(2.280e+02f64);
        let x12 = E::Vf::splat_as(2.9519662791675215849e-03f64);
        let x21 = E::Vf::splat_as(1.0130e+03f64);
        let x22 = E::Vf::splat_as(6.4716931485786837568e-04f64);
        let x31 = E::Vf::splat_as(1.8140e+03f64);
        let x32 = E::Vf::splat_as(1.1356030177269762362e-04f64);

        let yi0 = E::Vf::nan();
        let y00 = E::Vf::neg_infinity();
        let mut y03 = unsafe { E::Vf::undefined() };
        let mut y355 = unsafe { E::Vf::undefined() };
        let mut y558 = unsafe { E::Vf::undefined() };
        let mut y8i = unsafe { E::Vf::undefined() };

        let le3 = x.le(E::Vf::splat_as(3.0f64));
        let le55 = x.le(E::Vf::splat_as(5.5f64));
        let le8 = x.le(E::Vf::splat_as(8.0));

        // if le3 AND le55, then NOT between.
        let be355 = le3 ^ le55;
        // if le55 AND le8, then NOT between.
        let be558 = le55 ^ le8;

        todo!()
    }
}
