use std::{fs, path::Path, process::Command};
const EXIT: &str = "extern fn exit(status:i32)->void;";
fn run(source: &str) -> std::process::Output {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("main.fern");
    let output = dir.path().join("program");
    fs::write(&input, format!("{EXIT}\n{source}")).unwrap();
    fern::compile(&input, &output).unwrap_or_else(|error| panic!("{error}\n{source}"));
    let result = Command::new(output).output().unwrap();
    assert!(result.status.success(), "{result:?}\n{source}");
    result
}
#[test]
fn every_integer_width_wraps_and_converts_like_constants() {
    let mut source = String::from("fn main()->void {");
    for (ty, bits, signed) in [
        ("i8", 8, true),
        ("i16", 16, true),
        ("i32", 32, true),
        ("i64", 64, true),
        ("u8", 8, false),
        ("u16", 16, false),
        ("u32", 32, false),
        ("u64", 64, false),
        ("int", usize::BITS, true),
        ("uint", usize::BITS, false),
    ] {
        let max = (1u128 << (bits - u32::from(signed))) - 1;
        source.push_str(&format!(
            r#"{{
 var a:{ty}={max};var one:{ty}=1;var two:{ty}=2;
 const sum:{ty}={max}+1;const product:{ty}={max}*2;
 const negative:{ty}=-1;const shifted:{ty}={max}<<1;
 if a+one!=sum || a*two!=product || -one!=negative || a<<one!=shifted {{exit(1);}}
 a+=one;if a!=sum {{exit(2);}}
 var n:{ty}=7;var divisor:{ty}=3;
 if n/divisor!=2 || n%divisor!=1 {{exit(3);}}
 var small=i8(n);var wide=i64(small);var unsigned=u64(small);
 if small!=7 || wide!=7 || unsigned!=7 {{exit(4);}}
 if ^one!={ty}(^1) || (n&divisor)!=3 || (n|divisor)!=7 || (n^divisor)!=4 {{exit(5);}}
 }}"#
        ));
    }
    source.push_str("var n:i8=-1;if u64(n)!=18446744073709551615 || u8(n)!=255 {exit(6);}var x:i8=-128;if -x!=-128 {exit(7);} }");
    run(&source);
}
#[test]
fn ieee_arithmetic_and_conversions_match_constants() {
    run(r#"fn main()->void {
 var large:f32=16777216.0;var one:f32=1.0;const rounded:f32=16777216.0+1.0;
 if large+one!=rounded {exit(1);}
 var a:f32=1.25;var b:f32=2.5;
 if a+b!=3.75 || b-a!=1.25 || a*b!=3.125 || b/a!=2.0 {exit(2);}
 var zero:f64=0.0;var inf=1.0/zero;var nan=zero/zero;
 const cinf=1.0/0.0;const cnan=0.0/0.0;
 if inf!=cinf || nan==nan || cnan==cnan || nan<0.0 || nan>=0.0 {exit(3);}
 var minus_zero:f64=-0.0;
 if minus_zero!=zero || 1.0/minus_zero!=-cinf {exit(4);}
 var smallest:f32=1.401298464324817e-45;
 if smallest==0.0 || smallest+smallest!=2.802596928649634e-45 {exit(5);}
 var precise:f64=0.1;if f32(precise)!=0.1 || f64(f32(precise))==precise {exit(6);}
 var fraction:f64=-127.9;if i8(fraction)!=-127 || i64(fraction)!=-127 {exit(7);}
 var unsigned:f64=255.9;if u8(unsigned)!=255 {exit(8);}
 const cmaximum:u64=18446744073709551615;var maximum=cmaximum;
 if f32(maximum)!=f32(cmaximum) || f64(maximum)!=f64(cmaximum) {exit(9);}
 var boundary:f64=9223372036854775808.0;if u64(boundary)!=9223372036854775808 {exit(10);}
 var overflow:f32=1e39;if overflow!=1.0/f32(0.0) {exit(11);}
 }"#);
}
#[test]
fn contextual_f32_literals_round_directly() {
    run(r#"fn main()->void {
 // Above the f32 midpoint, but rounds to the midpoint in f64.
 var a:f32=1.0000000596046447753906250000000000000001;
 var b:f32=(1.0000000596046447753906250000000000000001);
 var c:f32=-1.0000000596046447753906250000000000000001;
 if a!=1.00000011920928955078125 || b!=a || c!=-a {exit(1);}
 }"#);
}
#[test]
fn pointer_arithmetic_casts_and_layout() {
    run(r#"type S struct {a:u8,b:i16,c:f32}
 fn main()->void {
 var a:[4]i16=[10,20,30,40];var p=&a[0];p[1]=25;*(p+2)=35;
 var end=p+4;var last=end-1;
 if last-p!=3 || (end-p)+2!=6 || *last!=40 || a[1]!=25 || a[2]!=35 {exit(1);}
 var q:*i16=null;if q!=null {exit(2);}q=p;
 var bytes=*u8(q);if *i16(bytes)!=q {exit(3);}
 var s=S {a=1,b=2,c=3.0};var sp=&s;(*sp).b=4;
 if s.b!=4 || sizeof(S)!=8 || sizeof([4]i16)!=8 || sizeof(bool)!=1 {exit(4);}
 var ap=&a;(*ap)[3]=50;if a[3]!=50 {exit(5);}
 }"#);
}
#[test]
fn parameters_are_addressable_mutable_copies() {
    run(r#"type S struct {a:[2]u8,x:f64}
 fn scalar(n:int)->int {var p=&n;*p+=1;return n;}
 fn aggregate(s:S)->S {s.a[0]=99;var p=&s;(*p).x=7.0;return s;}
 fn array(a:[2]u8)->[2]u8 {var p=&a;(*p)[1]=88;return a;}
 fn main()->void {
 var n=5;var s=S {a=[1,2],x=3.0};var copy=aggregate(s);var a=array(s.a);
 if scalar(n)!=6 || n!=5 || s.a[0]!=1 || s.x!=3.0 || copy.a[0]!=99 || copy.x!=7.0 || a[1]!=88 || s.a[1]!=2 {exit(1);}
 }"#);
}
#[test]
fn aggregate_equality_uses_values_and_ieee_rules() {
    run(r#"type S struct {byte:u8,value:f64}
 fn main()->void {
 var a=[S {byte=1,value=0.0},S {byte=2,value=3.0}];var b=a;
 b[0].value=-0.0;if a!=b {exit(1);}
 b[1].byte=3;if a==b {exit(2);}
 var zero:f64=0.0;b[1].value=zero/zero;if b==b {exit(3);}
 const c=[S {byte=1,value=0.0},S {byte=2,value=3.0}];if a!=c {exit(4);}
 }"#);
}
#[test]
fn evaluation_order_and_short_circuiting() {
    run(r#"var trace=0;var data=[10,20];
 fn step(n:int)->int {trace=trace*10+n;return n;}
 fn choose()->*int {step(1);return &data[0];}
 fn index()->int {step(2);return 1;}
 fn value()->int {step(3);return 7;}
 fn combine(a:int,b:int,c:int)->int {return a*100+b*10+c;}
 type S struct {a:int,b:int}
 fn main()->void {
 var n=step(1)+step(2)*step(3);if trace!=123 || n!=7 {exit(1);}
 trace=0;n=combine(step(1),step(2),step(3));if trace!=123 || n!=123 {exit(2);}
 trace=0;var s=S {b=step(2),a=step(1)};if trace!=21 || s.a!=1 || s.b!=2 {exit(3);}
 trace=0;choose()[index()]+=value();if trace!=123 || data[1]!=27 {exit(4);}
 trace=0;var b=false && step(1)==1;var c=true || step(2)==2;if b || !c || trace!=0 {exit(5);}
 }"#);
}
#[test]
fn aggregate_arguments_precede_later_side_effects() {
    run(r#"var a=[1,2];
 fn mutate()->int {a[0]=9;return 0;}
 fn first(value:[2]int,ignored:int)->int {return value[0];}
 fn main()->void {if first(a,mutate())!=1 || a[0]!=9 {exit(1);}}"#);
}
#[test]
fn loops_recursion_and_returns() {
    run(
        r#"fn fibonacci(n:int)->int {if n<2 {return n;}return fibonacci(n-1)+fibonacci(n-2);}
 fn forever()->int {for {return 42;}}
 fn main()->void {var n=0;var sum=0;for n<10 {n+=1;if n==3 {continue;}var m=0;for {m+=1;if m==2 {break;}}sum+=m;}if sum!=18 || fibonacci(10)!=55 || forever()!=42 {exit(1);}}"#,
    );
}
#[test]
fn external_libc_allocation_and_io() {
    run(
        r#"extern fn malloc(n:uint)->*u8;extern fn free(p:*u8)->void;extern fn putchar(c:i32)->i32;
 fn main()->void {var raw=malloc(sizeof([3]f64));if raw==null {exit(1);}var values=*f64(raw);values[0]=1.25;values[1]=2.5;values[2]=values[0]+values[1];if values[2]!=3.75 {exit(2);}free(raw);if putchar(65)!=65 {exit(3);}}"#,
    );
}
#[test]
fn invalid_programs_preserve_outputs() {
    for (source, expected) in [
        ("fn main()->void {const n=1/0;}", "undefined constant"),
        (
            "fn main()->void {const n=i8(-128)%i8(-1);}",
            "undefined constant",
        ),
        ("fn main()->void {const n=1<<64;}", "undefined constant"),
        (
            "fn main()->void {const n=u64(18446744073709551616.0);}",
            "undefined constant",
        ),
        ("fn main()->void {var n:i8=128;}", "out of range"),
        ("fn main()->void {var x:int;}", "initializer"),
        (
            "fn main()->void {var x=1;const y=x;}",
            "constant expression",
        ),
        ("fn main()->void {const x=1;var p=&x;}", "immutable"),
        (
            "extern fn x(n:f32)->void;fn main()->void {}",
            "integers and pointers",
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("main.fern");
        let output = dir.path().join("output");
        fs::write(&input, source).unwrap();
        fs::write(&output, b"keep").unwrap();
        let error = fern::compile(&input, &output).unwrap_err().to_string();
        assert!(error.contains(expected), "{error}");
        assert_eq!(fs::read(output).unwrap(), b"keep");
    }
}
#[test]
fn inconsistent_external_signatures_across_modules() {
    let dir = tempfile::tempdir().unwrap();
    for (path, source) in [
        ("lib/a.fern", "pub extern fn abs(x:i32)->i32;"),
        (
            "app/main.fern",
            "use lib;extern fn abs(x:i64)->i64;fn main()->void {lib::abs(1);}",
        ),
    ] {
        let path = dir.path().join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, source).unwrap();
    }
    let error = fern::compile(&dir.path().join("app"), &dir.path().join("output"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("inconsistent declarations"), "{error}");
}
#[cfg(unix)]
#[test]
fn c_abi_small_integers_and_struct_memory() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let c = dir.path().join("helper.c");
    let object = dir.path().join("helper.o");
    fs::write(
        &c,
        r#"
 #include <stdint.h>
 struct Record {uint8_t tag;int16_t count;float value;double other;};
 int8_t signed8(int8_t x){return x-1;}
 uint8_t unsigned8(uint8_t x){return x+1;}
 int16_t signed16(int16_t x){return x-1;}
 uint16_t unsigned16(uint16_t x){return x+1;}
 uintptr_t record_size(void){return sizeof(struct Record);}
 int32_t check_record(struct Record *r){
 if(r->tag!=255||r->count!=-123||r->value!=1.25f||r->other!=2.5)return 1;
 r->tag=7;r->count=-321;r->value=3.75f;return 0;
 }"#,
    )
    .unwrap();
    assert!(
        Command::new("cc")
            .arg("-c")
            .arg(&c)
            .arg("-o")
            .arg(&object)
            .status()
            .unwrap()
            .success()
    );
    let wrapper = dir.path().join("cc-wrapper");
    let quoted = object.to_string_lossy().replace('\'', "'\\''");
    fs::write(&wrapper, format!("#!/bin/sh\nexec cc \"$@\" '{quoted}'\n")).unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o755)).unwrap();
    let input = dir.path().join("main.fern");
    let output = dir.path().join("program");
    fs::write(&input,format!(r#"{EXIT}
 extern fn signed8(x:i8)->i8;extern fn unsigned8(x:u8)->u8;
 extern fn signed16(x:i16)->i16;extern fn unsigned16(x:u16)->u16;
 extern fn record_size()->uint;type Record struct {{tag:u8,count:i16,value:f32,other:f64}}
 extern fn check_record(record:*Record)->i32;
 fn main()->void {{
 var a:i8=-100;var b:u8=254;var c:i16=-30000;var d:u16=65534;
 if signed8(a)!=-101 || unsigned8(b)!=255 || signed16(c)!=-30001 || unsigned16(d)!=65535 {{exit(1);}}
 var r=Record {{tag=255,count=-123,value=1.25,other=2.5}};
 if sizeof(Record)!=record_size() || check_record(&r)!=0 || r.tag!=7 || r.count!=-321 || r.value!=3.75 {{exit(2);}}
 }}"#)).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_fern"))
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .env("CC", &wrapper)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(Command::new(output).status().unwrap().success());
}
#[test]
fn examples_execute() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = tempfile::tempdir().unwrap();
    for entry in fs::read_dir(root.join("examples")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|x| x == "fern") {
            let output = dir.path().join(path.file_stem().unwrap());
            fern::compile(&path, &output).unwrap();
            assert!(
                Command::new(output).status().unwrap().success(),
                "{}",
                path.display()
            );
        }
    }
}
#[test]
fn len_of_an_array_larger_than_the_stack_executes() {
    run(r#"extern fn malloc(bytes:uint)->*u8;
fn main()->void {
    var raw = malloc(sizeof([1200000]i64));
    if raw == null { exit(2); }
    var big = *[1200000]i64(raw);
    if len(*big) != 1200000 { exit(1); }
}"#);
}
#[test]
fn len_runs_the_effects_of_its_operand() {
    run(r#"var calls=0;
var grid:[2][3]i64=[[1,2,3],[4,5,6]];
fn make()->[5]i64 {calls+=1;return [1,2,3,4,5];}
fn one()->int {calls+=1;return 1;}
fn at()->*[3]i64 {calls+=1;return &grid[0];}
fn main()->void {
 if len(make())!=5 || len(grid[one()])!=3 || len(*(at()))!=3 {exit(1);}
 if calls!=3 {exit(2);}
}"#);
}
#[test]
fn string_literals_are_static_terminated_bytes() {
    run(r#"const shared="shared";
fn address()->*u8 {return shared;}
fn main()->void {
 var s="a\tb\x21\0c";
 if s[0]!='a' || s[1]!='\t' || s[2]!='b' || *(s+3)!='!' || s[4]!=0 || s[5]!='c' || s[6]!=0 {exit(1);}
 var e="é";
 if e[0]!=0xC3 || e[1]!=0xA9 || e[2]!=0 {exit(2);}
 if address()!=shared || address()!=address() {exit(3);}
 }"#);
}
#[test]
fn string_literals_reach_c() {
    let output = run(r#"extern fn puts(text:*u8)->i32;
var greeting:*u8="hello\tworld\x21";
fn main()->void {puts(greeting);puts("it's \"quoted\"");}"#);
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "hello\tworld!\nit's \"quoted\"\n"
    );
}
