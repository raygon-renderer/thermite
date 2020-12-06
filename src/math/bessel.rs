use crate::*;

pub use bessel_internal::*;

mod bessel_internal {
    use crate::*;

    const FRAC_1_SQRT_PI: f64 = 5.641895835477562869480794515607725858e-01;

    #[inline(always)]
    pub fn bessel_j<S: Simd, E: SimdVectorizedMathInternal<S>, P: Policy>(x: E::Vf, n: u32) -> E::Vf {
        match n {
            0 => bessel_j0::<S, E, P>(x),
            1 => bessel_j1::<S, E, P>(x),
            _ => unimplemented!(),
        }
    }

    #[inline(always)]
    pub fn bessel_y<S: Simd, E: SimdVectorizedMathInternal<S>, P: Policy>(mut x: E::Vf, n: u32) -> E::Vf {
        match n {
            0 => bessel_y0::<S, E, P>(x),
            _ => unimplemented!(),
        }
    }

    #[inline(always)]
    pub fn bessel_j0<S: Simd, E: SimdVectorizedMathInternal<S>, P: Policy>(mut x: E::Vf) -> E::Vf {
        let x1 = E::Vf::splat_as(2.4048255576957727686e+00);
        let x2 = E::Vf::splat_as(5.5200781102863106496e+00);
        let x11 = E::Vf::splat_as(6.160e+02);
        let x12 = E::Vf::splat_as(-1.42444230422723137837e-03);
        let x21 = E::Vf::splat_as(1.4130e+03);
        let x22 = E::Vf::splat_as(5.46860286310649596604e-04);
        let den = E::Vf::splat_as(-1.0 / 256.0);

        let p1 = &[
            E::cast_from(-4.1298668500990866786e+11f64),
            E::cast_from(2.7282507878605942706e+10f64),
            E::cast_from(-6.2140700423540120665e+08f64),
            E::cast_from(6.6302997904833794242e+06f64),
            E::cast_from(-3.6629814655107086448e+04f64),
            E::cast_from(1.0344222815443188943e+02f64),
            E::cast_from(-1.2117036164593528341e-01f64),
        ];
        let q1 = &[
            E::cast_from(2.3883787996332290397e+12f64),
            E::cast_from(2.6328198300859648632e+10f64),
            E::cast_from(1.3985097372263433271e+08f64),
            E::cast_from(4.5612696224219938200e+05f64),
            E::cast_from(9.3614022392337710626e+02f64),
            E::cast_from(1.0f64),
            E::cast_from(0.0f64),
        ];

        let p2 = &[
            E::cast_from(-1.8319397969392084011e+03f64),
            E::cast_from(-1.2254078161378989535e+04f64),
            E::cast_from(-7.2879702464464618998e+03f64),
            E::cast_from(1.0341910641583726701e+04f64),
            E::cast_from(1.1725046279757103576e+04f64),
            E::cast_from(4.4176707025325087628e+03f64),
            E::cast_from(7.4321196680624245801e+02f64),
            E::cast_from(4.8591703355916499363e+01f64),
        ];

        let q2 = &[
            E::cast_from(-3.5783478026152301072e+05f64),
            E::cast_from(2.4599102262586308984e+05f64),
            E::cast_from(-8.4055062591169562211e+04f64),
            E::cast_from(1.8680990008359188352e+04f64),
            E::cast_from(-2.9458766545509337327e+03f64),
            E::cast_from(3.3307310774649071172e+02f64),
            E::cast_from(-2.5258076240801555057e+01f64),
            E::cast_from(1.0f64),
        ];

        let pc = &[
            E::cast_from(2.2779090197304684302e+04f64),
            E::cast_from(4.1345386639580765797e+04f64),
            E::cast_from(2.1170523380864944322e+04f64),
            E::cast_from(3.4806486443249270347e+03f64),
            E::cast_from(1.5376201909008354296e+02f64),
            E::cast_from(8.8961548424210455236e-01f64),
        ];

        let qc = &[
            E::cast_from(2.2779090197304684318e+04f64),
            E::cast_from(4.1370412495510416640e+04f64),
            E::cast_from(2.1215350561880115730e+04f64),
            E::cast_from(3.5028735138235608207e+03f64),
            E::cast_from(1.5711159858080893649e+02f64),
            E::cast_from(1.0f64),
        ];

        let ps = &[
            E::cast_from(-8.9226600200800094098e+01f64),
            E::cast_from(-1.8591953644342993800e+02f64),
            E::cast_from(-1.1183429920482737611e+02f64),
            E::cast_from(-2.2300261666214198472e+01f64),
            E::cast_from(-1.2441026745835638459e+00f64),
            E::cast_from(-8.8033303048680751817e-03f64),
        ];

        let qs = &[
            E::cast_from(5.7105024128512061905e+03f64),
            E::cast_from(1.1951131543434613647e+04f64),
            E::cast_from(7.2642780169211018836e+03f64),
            E::cast_from(1.4887231232283756582e+03f64),
            E::cast_from(9.0593769594993125859e+01f64),
            E::cast_from(1.0f64),
        ];

        x = x.abs(); // even function

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
            let r = (x * x).poly_rational_p::<P>(p1, q1);

            y04 = (x + x1) * (den.mul_adde(x11, x) - x12) * r;
        };

        // 4 < x <= 8
        if P::POLICY.avoid_branching || be48.any() {
            let y = E::Vf::splat_as(-1.0 / 64.0).mul_adde(x * x, E::Vf::one());

            // y <= 1 given above operation
            let r = y.poly_p::<P>(p2) / y.poly_p::<P>(q2);

            y48 = den.mul_adde(x21, x) * (x + x2) * r;
        }

        // 8 < x <= inf
        if P::POLICY.avoid_branching || !le8.all() {
            let factor = E::Vf::splat_as(FRAC_1_SQRT_PI) / x.sqrt();

            let y = E::Vf::splat_as(8.0) / x;
            let y2 = y * y;

            // y2 <= 1.0 given above division, no need to do poly_rational
            let rc = y2.poly_p::<P>(pc) / y2.poly_p::<P>(qc);
            let rs = y2.poly_p::<P>(ps) / y2.poly_p::<P>(qs);

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
        let x1 = E::Vf::splat_as(3.8317059702075123156e+00f64);
        let x2 = E::Vf::splat_as(7.0155866698156187535e+00f64);
        let x11 = E::Vf::splat_as(9.810e+02f64);
        let x12 = E::Vf::splat_as(-3.2527979248768438556e-04f64);
        let x21 = E::Vf::splat_as(1.7960e+03f64);
        let x22 = E::Vf::splat_as(-3.8330184381246462950e-05f64);
        let den = E::Vf::splat_as(-1.0 / 256.0);

        let p1 = &[
            E::cast_from(-1.4258509801366645672e+11f64),
            E::cast_from(6.6781041261492395835e+09f64),
            E::cast_from(-1.1548696764841276794e+08f64),
            E::cast_from(9.8062904098958257677e+05f64),
            E::cast_from(-4.4615792982775076130e+03f64),
            E::cast_from(1.0650724020080236441e+01f64),
            E::cast_from(-1.0767857011487300348e-02f64),
        ];

        let q1 = &[
            E::cast_from(4.1868604460820175290e+12f64),
            E::cast_from(4.2091902282580133541e+10f64),
            E::cast_from(2.0228375140097033958e+08f64),
            E::cast_from(5.9117614494174794095e+05f64),
            E::cast_from(1.0742272239517380498e+03f64),
            E::cast_from(1.0f64),
            E::cast_from(0.0f64),
        ];

        let p2 = &[
            E::cast_from(4.6179191852758252278e+00f64),
            E::cast_from(-7.5023342220781607561e+03f64),
            E::cast_from(5.0793266148011179143e+06f64),
            E::cast_from(-1.8113931269860667829e+09f64),
            E::cast_from(3.5580665670910619166e+11f64),
            E::cast_from(-3.6658018905416665164e+13f64),
            E::cast_from(1.6608531731299018674e+15f64),
            E::cast_from(-1.7527881995806511112e+16f64),
        ];
        let q2 = &[
            E::cast_from(1.0f64),
            E::cast_from(1.3886978985861357615e+03f64),
            E::cast_from(1.1267125065029138050e+06f64),
            E::cast_from(6.4872502899596389593e+08f64),
            E::cast_from(2.7622777286244082666e+11f64),
            E::cast_from(8.4899346165481429307e+13f64),
            E::cast_from(1.7128800897135812012e+16f64),
            E::cast_from(1.7253905888447681194e+18f64),
        ];

        let pc = &[
            E::cast_from(-4.4357578167941278571e+06f64),
            E::cast_from(-9.9422465050776411957e+06f64),
            E::cast_from(-6.6033732483649391093e+06f64),
            E::cast_from(-1.5235293511811373833e+06f64),
            E::cast_from(-1.0982405543459346727e+05f64),
            E::cast_from(-1.6116166443246101165e+03f64),
            E::cast_from(0.0f64),
        ];

        let qc = &[
            E::cast_from(-4.4357578167941278568e+06f64),
            E::cast_from(-9.9341243899345856590e+06f64),
            E::cast_from(-6.5853394797230870728e+06f64),
            E::cast_from(-1.5118095066341608816e+06f64),
            E::cast_from(-1.0726385991103820119e+05f64),
            E::cast_from(-1.4550094401904961825e+03f64),
            E::cast_from(1.0f64),
        ];

        let ps = &[
            E::cast_from(3.3220913409857223519e+04f64),
            E::cast_from(8.5145160675335701966e+04f64),
            E::cast_from(6.6178836581270835179e+04f64),
            E::cast_from(1.8494262873223866797e+04f64),
            E::cast_from(1.7063754290207680021e+03f64),
            E::cast_from(3.5265133846636032186e+01f64),
            E::cast_from(0.0f64),
        ];

        let qs = &[
            E::cast_from(7.0871281941028743574e+05f64),
            E::cast_from(1.8194580422439972989e+06f64),
            E::cast_from(1.4194606696037208929e+06f64),
            E::cast_from(4.0029443582266975117e+05f64),
            E::cast_from(3.7890229745772202641e+04f64),
            E::cast_from(8.6383677696049909675e+02f64),
            E::cast_from(1.0f64),
        ];

        let w = x.abs();

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
            let r = (x * x).poly_rational_p::<P>(p1, q1);

            y04 = r * w * (w + x1) * (den.mul_adde(x11, w) - x12);
        }

        // 4 < w <= 8
        if P::POLICY.avoid_branching || be48.any() {
            let y = E::Vf::one() / (x * x);
            let r = y.poly_p::<P>(p2) / y.poly_p::<P>(q2);

            y48 = r * w * (w + x2) * (den.mul_adde(x21, w) - x22);
        }

        // 8 < w <= inf
        if P::POLICY.avoid_branching || !le8.all() {
            let y = E::Vf::splat_as(8.0) / w;
            let y2 = y * y;

            let factor = E::Vf::splat_as(FRAC_1_SQRT_PI) / w.sqrt();

            // y2 <= 1.0 given above division, no need for poly_rational
            let rc = y2.poly_p::<P>(pc) / y2.poly_p::<P>(qc);
            let rs = y2.poly_p::<P>(ps) / y2.poly_p::<P>(qs);

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

        // ln(x1, x2, x3)
        let lnx1 = E::Vf::splat_as(-0.112522807880794125038133980477252212668015528121886074367185547f64);
        let lnx2 = E::Vf::splat_as(1.3756575956013471336175786440565011276993580049899787950261170276f64);
        let lnx3 = E::Vf::splat_as(1.9581282122177381156900951130612350155885109149487567360554102593f64);

        let x11 = E::Vf::splat_as(2.280e+02f64);
        let x12 = E::Vf::splat_as(2.9519662791675215849e-03f64);
        let x21 = E::Vf::splat_as(1.0130e+03f64);
        let x22 = E::Vf::splat_as(6.4716931485786837568e-04f64);
        let x31 = E::Vf::splat_as(1.8140e+03f64);
        let x32 = E::Vf::splat_as(1.1356030177269762362e-04f64);
        let den = E::Vf::splat_as(-1.0 / 256.0);

        let p1 = &[
            E::cast_from(1.0723538782003176831e+11f64),
            E::cast_from(-8.3716255451260504098e+09f64),
            E::cast_from(2.0422274357376619816e+08f64),
            E::cast_from(-2.1287548474401797963e+06f64),
            E::cast_from(1.0102532948020907590e+04f64),
            E::cast_from(-1.8402381979244993524e+01f64),
        ];
        let q1 = &[
            E::cast_from(5.8873865738997033405e+11f64),
            E::cast_from(8.1617187777290363573e+09f64),
            E::cast_from(5.5662956624278251596e+07f64),
            E::cast_from(2.3889393209447253406e+05f64),
            E::cast_from(6.6475986689240190091e+02f64),
            E::cast_from(1.0f64),
        ];
        let p2 = &[
            E::cast_from(1.7427031242901594547e+01f64),
            E::cast_from(-1.4566865832663635920e+04f64),
            E::cast_from(4.6905288611678631510e+06f64),
            E::cast_from(-6.9590439394619619534e+08f64),
            E::cast_from(4.3600098638603061642e+10f64),
            E::cast_from(-5.5107435206722644429e+11f64),
            E::cast_from(-2.2213976967566192242e+13f64),
        ];
        let q2 = &[
            E::cast_from(1.0f64),
            E::cast_from(8.3030857612070288823e+02f64),
            E::cast_from(4.0669982352539552018e+05f64),
            E::cast_from(1.3960202770986831075e+08f64),
            E::cast_from(3.4015103849971240096e+10f64),
            E::cast_from(5.4266824419412347550e+12f64),
            E::cast_from(4.3386146580707264428e+14f64),
        ];
        let p3 = &[
            E::cast_from(-1.7439661319197499338e+01f64),
            E::cast_from(2.1363534169313901632e+04f64),
            E::cast_from(-1.0085539923498211426e+07f64),
            E::cast_from(2.1958827170518100757e+09f64),
            E::cast_from(-1.9363051266772083678e+11f64),
            E::cast_from(-1.2829912364088687306e+11f64),
            E::cast_from(6.7016641869173237784e+14f64),
            E::cast_from(-8.0728726905150210443e+15f64),
        ];
        let q3 = &[
            E::cast_from(1.0f64),
            E::cast_from(8.7903362168128450017e+02f64),
            E::cast_from(5.3924739209768057030e+05f64),
            E::cast_from(2.4727219475672302327e+08f64),
            E::cast_from(8.6926121104209825246e+10f64),
            E::cast_from(2.2598377924042897629e+13f64),
            E::cast_from(3.9272425569640309819e+15f64),
            E::cast_from(3.4563724628846457519e+17f64),
        ];
        let pc = &[
            E::cast_from(2.2779090197304684302e+04f64),
            E::cast_from(4.1345386639580765797e+04f64),
            E::cast_from(2.1170523380864944322e+04f64),
            E::cast_from(3.4806486443249270347e+03f64),
            E::cast_from(1.5376201909008354296e+02f64),
            E::cast_from(8.8961548424210455236e-01f64),
        ];
        let qc = &[
            E::cast_from(2.2779090197304684318e+04f64),
            E::cast_from(4.1370412495510416640e+04f64),
            E::cast_from(2.1215350561880115730e+04f64),
            E::cast_from(3.5028735138235608207e+03f64),
            E::cast_from(1.5711159858080893649e+02f64),
            E::cast_from(1.0f64),
        ];
        let ps = &[
            E::cast_from(-8.9226600200800094098e+01f64),
            E::cast_from(-1.8591953644342993800e+02f64),
            E::cast_from(-1.1183429920482737611e+02f64),
            E::cast_from(-2.2300261666214198472e+01f64),
            E::cast_from(-1.2441026745835638459e+00f64),
            E::cast_from(-8.8033303048680751817e-03f64),
        ];
        let qs = &[
            E::cast_from(5.7105024128512061905e+03f64),
            E::cast_from(1.1951131543434613647e+04f64),
            E::cast_from(7.2642780169211018836e+03f64),
            E::cast_from(1.4887231232283756582e+03f64),
            E::cast_from(9.0593769594993125859e+01f64),
            E::cast_from(1.0f64),
        ];

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

        let mut j0_2_frac_pi = unsafe { E::Vf::undefined() };
        let mut lnx_j0_2_frac_pi = unsafe { E::Vf::undefined() };
        let mut xx = unsafe { E::Vf::undefined() };
        let mut ixx = unsafe { E::Vf::undefined() };

        if P::POLICY.avoid_branching || le8.any() {
            xx = x * x;
            ixx = E::Vf::one() / xx; // setup division early to pipeline it
            j0_2_frac_pi = x.bessel_j_p::<P>(0) * E::Vf::splat_as(core::f64::consts::FRAC_2_PI);
            lnx_j0_2_frac_pi = x.ln_p::<P>() * j0_2_frac_pi;
        }

        if P::POLICY.avoid_branching || le3.any() {
            let r = xx.poly_rational_p::<P>(p1, q1);
            let f = (x + x1) * (den.mul_adde(x11, x) - x12);
            y03 = f.mul_adde(r, lnx1.nmul_adde(j0_2_frac_pi, lnx_j0_2_frac_pi));
        }

        if P::POLICY.avoid_branching || be355.any() {
            let r = ixx.poly_p::<P>(p2) / ixx.poly_p::<P>(q2);
            let f = (x + x2) * (den.mul_adde(x21, x) - x22);
            y355 = f.mul_adde(r, lnx2.nmul_adde(j0_2_frac_pi, lnx_j0_2_frac_pi));
        }

        if P::POLICY.avoid_branching || be558.any() {
            let r = ixx.poly_p::<P>(p3) / ixx.poly_p::<P>(q3);
            let f = (x + x3) * (den.mul_adde(x31, x) - x32);
            y558 = f.mul_adde(r, lnx3.nmul_adde(j0_2_frac_pi, lnx_j0_2_frac_pi));
        }

        if P::POLICY.avoid_branching || !le8.all() {
            let y = E::Vf::splat_as(8.0) / x;
            let y2 = y * y;

            let factor = E::Vf::splat_as(FRAC_1_SQRT_PI) / x.sqrt();

            // y2 <= 1.0 given above division, no need for poly_rational
            let rc = y2.poly_p::<P>(pc) / y2.poly_p::<P>(qc);
            let rs = y2.poly_p::<P>(ps) / y2.poly_p::<P>(qs);

            let (sx, cx) = x.sin_cos_p::<P>();

            y8i = factor * rc.mul_adde(sx - cx, y * rs * (cx + sx));
        }

        let mut y = y8i;
        y = le8.select(y558, y);
        y = le55.select(y355, y);
        y = le3.select(y03, y);
        y = x.eq(E::Vf::zero()).select(y00, y);
        y = x.lt(E::Vf::zero()).select(yi0, y);

        y
    }
}
