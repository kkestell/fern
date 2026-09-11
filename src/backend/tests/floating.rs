use super::*;
use crate::{backend::floating::literal, types::Float};

fn single(value: f32) -> Float {
    Float::Binary32(value.to_bits())
}

fn double(value: f64) -> Float {
    Float::Binary64(value.to_bits())
}

#[test]
fn a_literal_is_spelled_as_the_exact_value_its_bits_hold() {
    for (value, expected) in [
        (double(0.0), "d_0x0p+0"),
        (double(-0.0), "d_-0x0p+0"),
        (double(1.0), "d_0x1p+0"),
        (double(1.5), "d_0x1.8p+0"),
        (double(-1.5), "d_-0x1.8p+0"),
        (double(0.1), "d_0x1.999999999999ap-4"),
        (double(f64::MIN_POSITIVE), "d_0x1p-1022"),
        // A subnormal has no leading one and the smallest normal exponent.
        (Float::Binary64(1), "d_0x0.0000000000001p-1022"),
        (double(f64::INFINITY), "d_inf"),
        (double(f64::NEG_INFINITY), "d_-inf"),
        (double(f64::NAN), "d_nan"),
        (single(0.5), "s_0x1p-1"),
        (single(0.1), "s_0x1.99999ap-4"),
        (single(-0.0), "s_-0x0p+0"),
        (Float::Binary32(1), "s_0x0.000002p-126"),
        (single(f32::INFINITY), "s_inf"),
    ] {
        assert_eq!(literal(value), expected, "{value:?}");
    }
}

#[test]
fn emitted_floating_values_use_floating_classes_and_operations() {
    let qbe = emit(
        &lowered(
            "var scale: f64 = 1.5;
                 fn half(value: f32) -> f32 { return value / f32(2.0); }
                 fn main() -> void {
                     var narrow = half(f32(0.5));
                     var wide = scale * f64(narrow);
                     var negated = -wide;
                     if negated < wide { exit(0); }
                     exit(1);
                 }",
        ),
        None,
    );
    // The global holds its bits, so no decimal spelling can round it.
    assert!(
        qbe.contains(&format!("data $global0 = {{ l {} }}", 1.5f64.to_bits())),
        "{qbe}"
    );
    for expected in [
        "function s $fn0(s %param0)",
        "    %local0 =l alloc4 4",
        "    stores %param0, %local0",
        "    %v0 =s loads %local0",
        "    %v1 =s div %v0, s_0x1p+1",
        "    %v0 =s call $fn0(s s_0x1p-1)",
        "    %v1 =d loadd $global0",
        "    %v3 =d exts %v2",
        "    %v4 =d mul %v1, %v3",
        "    stored %v4, %local1",
        "    %v6 =d neg %v5",
        "    %v9 =w cltd %v7, %v8",
    ] {
        assert!(qbe.contains(expected), "{expected} missing from:\n{qbe}");
    }
    // Nothing reaches an integer word class or operation.
    assert!(!qbe.contains("=w mul"), "{qbe}");
    // Floating-point arithmetic and comparison never trap.
    assert!(!qbe.contains("$abort"), "{qbe}");
}

#[test]
fn native_floating_literals_globals_calls_and_returns_keep_their_values() {
    assert_eq!(
        native_status(
            "var narrow: f32 = 0.1;
                 if f64(narrow) == 0.1 { exit(1); }
                 var rounded: f32 = 0.5;
                 if f64(rounded) != 0.5 { exit(2); }
                 exit(42);"
        ),
        Some(42)
    );
    let program = lowered(
        "var scale: f64 = 2.5;
             var weights: [2]f64 = [0.5, 1.5];
             fn weigh(index: int) -> f64 { return weights[index] * scale; }
             fn doubled(row: [2]f32) -> [2]f32 { return [row[0] * 2.0, row[1] * 2.0]; }
             fn main() -> void {
                 if weigh(0) != 1.25 { exit(1); }
                 if weigh(1) != 3.75 { exit(2); }
                 var row: [2]f32 = [0.5, 1.5];
                 var expected: [2]f32 = [1.0, 3.0];
                 if doubled(row) != expected { exit(3); }
                 exit(42);
             }",
    );
    // An array of either format is an aggregate of its own scalar class.
    let qbe = emit(&program, None);
    assert!(qbe.contains("type :arrays2 = { s 2 }"), "{qbe}");
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&qbe, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
}

#[test]
fn native_floating_arithmetic_follows_ieee_754_without_trapping() {
    assert_eq!(
        native_status(
            "var one: f64 = 1.0; var zero: f64 = 0.0;
                 var negative_zero = -zero;
                 // A signed zero divides into a signed infinity.
                 if one / zero != one / negative_zero * -1.0 { exit(1); }
                 var nan = zero / zero;
                 if nan == nan { exit(2); }
                 if !(nan != nan) { exit(3); }
                 if nan < one || nan >= one { exit(4); }
                 // Zeroes of either sign compare equal.
                 if negative_zero != zero { exit(5); }
                 if -(one / zero) >= one { exit(6); }
                 var huge: f32 = 3.0e38;
                 // Overflow produces an infinity rather than trapping.
                 if huge * 2.0e0 <= huge { exit(7); }
                 exit(42);"
        ),
        Some(42)
    );
}

#[test]
fn native_floating_conversions_keep_exact_values_and_trap_on_the_rest() {
    assert_eq!(
        native_status(
            "var narrow: f32 = 0.5;
                 if f64(narrow) != 0.5 { exit(1); }
                 var wide: f64 = 0.5;
                 if f32(wide) != narrow { exit(2); }
                 // A signed zero and an infinity survive both directions.
                 var zero: f64 = 0.0;
                 var negative_zero = -zero;
                 if f64(f32(negative_zero)) != negative_zero { exit(3); }
                 var infinity = 1.0 / zero;
                 if f64(f32(infinity)) != infinity { exit(4); }
                 // A NaN narrows to a NaN, which equals nothing.
                 var nan = zero / zero;
                 var narrowed = f32(nan);
                 if narrowed == narrowed { exit(5); }
                 var whole: f64 = 2.0;
                 if i32(whole) != 2 { exit(6); }
                 if i32(-whole) != -2 { exit(7); }
                 if u8(negative_zero) != u8(0) { exit(8); }
                 var counted: i32 = 16777216;
                 if f64(f32(counted)) != 16777216.0 { exit(9); }
                 var large: u64 = 1024;
                 if f64(large) != 1024.0 { exit(10); }
                 exit(42);"
        ),
        Some(42)
    );
    for (body, expected) in [
        (
            "var wide: f64 = 0.1; var narrow = f32(wide); exit(0);",
            "checked conversion failed: `f64` to `f32`",
        ),
        (
            "var fraction: f64 = 1.5; var whole = i32(fraction); exit(0);",
            "checked conversion failed: `f64` to `i32`",
        ),
        (
            "var huge: f64 = 1.0e30; var whole = i32(huge); exit(0);",
            "checked conversion failed: `f64` to `i32`",
        ),
        (
            "var zero: f64 = 0.0; var nan = zero / zero; var whole = i32(nan); exit(0);",
            "checked conversion failed: `f64` to `i32`",
        ),
        (
            "var zero: f64 = 0.0; var infinity = 1.0 / zero; var whole = i64(infinity); exit(0);",
            "checked conversion failed: `f64` to `i64`",
        ),
        (
            "var negative: f64 = -1.0; var whole = u32(negative); exit(0);",
            "checked conversion failed: `f64` to `u32`",
        ),
        (
            "var counted: i32 = 16777217; var value = f32(counted); exit(0);",
            "checked conversion failed: `i32` to `f32`",
        ),
        (
            "var counted: i64 = 9223372036854775807; var value = f64(counted); exit(0);",
            "checked conversion failed: `i64` to `f64`",
        ),
        (
            "var counted: u64 = 18446744073709551615; var value = f64(counted); exit(0);",
            "checked conversion failed: `u64` to `f64`",
        ),
    ] {
        assert_native_failure(body, expected);
    }
}

#[test]
fn native_floating_array_equality_compares_element_by_element() {
    assert_eq!(
        native_status(
            "var zero: f64 = 0.0;
                 var negative_zero = -zero;
                 var left: [2]f64 = [0.5, 0.0];
                 var right: [2]f64 = [0.5, 0.0];
                 right[1] = negative_zero;
                 // The two zeroes are equal even though their bytes differ.
                 if left != right { exit(1); }
                 var nan = zero / zero;
                 right[1] = nan;
                 if left == right { exit(2); }
                 if !(left != right) { exit(3); }
                 // A NaN is not even equal to itself.
                 left[1] = nan;
                 if left == right { exit(4); }
                 var rows: [2][2]f32 = [[0.5, 1.5], [2.5, 3.5]];
                 var same: [2][2]f32 = [[0.5, 1.5], [2.5, 3.5]];
                 if rows != same { exit(5); }
                 same[1][1] = 4.5;
                 if rows == same { exit(6); }
                 exit(42);"
        ),
        Some(42)
    );
    let qbe = emit(
        &lowered(
            "fn main() -> void {
                 var left: [2]f64 = [0.5, 1.5];
                 var right: [2]f64 = [0.5, 1.5];
                 if left == right { exit(0); }
                 exit(1);
             }",
        ),
        None,
    );
    // Equal floating-point arrays need not hold identical bytes, so the
    // comparison loads each element rather than calling `memcmp`.
    assert!(!qbe.contains("memcmp"), "{qbe}");
    assert!(qbe.contains(" =w ceqd "), "{qbe}");
    assert!(qbe.contains("@aggregate0_loop"), "{qbe}");
}
