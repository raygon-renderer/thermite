use thermite::*;

type Vi32 = <backends::AVX2 as Simd>::Vi32;

macro_rules! test_sat {
    (SUB $name:ident => $a:expr; $b:expr) => {
        #[test]
        fn $name() {
            let a = $a;
            let b = $b;
            let cv = Vi32::splat(a).saturating_sub(Vi32::splat(b));
            assert_eq!(cv.extract(0), a.saturating_sub(b));
        }
    };
    (ADD $name:ident => $a:expr; $b:expr) => {
        #[test]
        fn $name() {
            let a = $a;
            let b = $b;
            let cv = Vi32::splat(a).saturating_add(Vi32::splat(b));
            assert_eq!(cv.extract(0), a.saturating_add(b));
        }
    };
}

test_sat!(ADD test_sat_add_overflowing => i32::MAX - 20; 210);
test_sat!(SUB test_sat_sub_overflowing => i32::MIN + 20; 210);
test_sat!(ADD test_sat_add_regular => i32::MAX - 20; 2);
test_sat!(SUB test_sat_sub_regular => i32::MIN + 20; 2);
test_sat!(ADD test_sat_add_regular_alt => -1; 2);
test_sat!(SUB test_sat_sub_regular_alt =>  1; 2);
