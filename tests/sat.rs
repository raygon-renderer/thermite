use thermite::*;

type Vi32 = <backends::AVX2 as Simd>::Vi32;
type Vu32 = <backends::AVX2 as Simd>::Vu32;

macro_rules! test_sat {
    (SUB: $ty:ty => $name:ident => $a:expr; $b:expr) => {
        #[test]
        fn $name() {
            let a = $a;
            let b = $b;
            let cv = <$ty>::splat(a).saturating_sub(<$ty>::splat(b));
            assert_eq!(cv.extract(0), a.saturating_sub(b));
        }
    };
    (ADD: $ty:ty => $name:ident => $a:expr; $b:expr) => {
        #[test]
        fn $name() {
            let a = $a;
            let b = $b;
            let cv = <$ty>::splat(a).saturating_add(<$ty>::splat(b));
            assert_eq!(cv.extract(0), a.saturating_add(b));
        }
    };
}

test_sat!(ADD: Vi32 => test_sat_sadd_overflowing => i32::MAX - 20; 210);
test_sat!(SUB: Vi32 => test_sat_ssub_overflowing => i32::MIN + 20; 210);
test_sat!(ADD: Vi32 => test_sat_sadd_regular => i32::MAX - 20; 2);
test_sat!(SUB: Vi32 => test_sat_ssub_regular => i32::MIN + 20; 2);
test_sat!(ADD: Vi32 => test_sat_sadd_regular_alt => -1; 2);
test_sat!(SUB: Vi32 => test_sat_ssub_regular_alt =>  1; 2);

test_sat!(ADD: Vu32 => test_sat_uadd_overflowing => u32::MAX - 20; 210);
test_sat!(ADD: Vu32 => test_sat_uadd_regular => 20; 210);
