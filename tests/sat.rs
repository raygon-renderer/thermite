use thermite::*;

type Vi32 = <backends::AVX2 as Simd>::Vi32;

#[test]
fn test_sat_add_overflowing() {
    let a = i32::MAX - 20;
    let b = 210;
    let cv = Vi32::splat(a).saturating_add(Vi32::splat(b));
    assert_eq!(cv.extract(0), a.saturating_add(b));
}

#[test]
fn test_sat_sub_overflowing() {
    let a = i32::MIN + 20;
    let b = 210;
    let cv = Vi32::splat(a).saturating_sub(Vi32::splat(b));
    assert_eq!(cv.extract(0), a.saturating_sub(b));
}

#[test]
fn test_sat_add_regular() {
    let a = i32::MAX - 20;
    let b = 2;
    let cv = Vi32::splat(a).saturating_add(Vi32::splat(b));
    assert_eq!(cv.extract(0), a.saturating_add(b));
}

#[test]
fn test_sat_sub_regular() {
    let a = i32::MIN + 20;
    let b = 2;
    let cv = Vi32::splat(a).saturating_sub(Vi32::splat(b));
    assert_eq!(cv.extract(0), a.saturating_sub(b));
}
