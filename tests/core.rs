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
            "extern fn x(n:bool)->void;fn main()->void {}",
            "numeric, pointer, and struct",
        ),
        (
            "type S struct {b:bool} extern fn x(s:S)->void;fn main()->void {}",
            "cannot contain bool",
        ),
        (
            "extern fn x(n:i32,...)->void;fn main()->void {x(1,f32(1.0));}",
            "converted to f64",
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
fn compile_c(source: &Path, object: &Path) {
    assert!(
        Command::new("cc")
            .arg("-c")
            .arg(source)
            .arg("-o")
            .arg(object)
            .status()
            .unwrap()
            .success()
    );
}
fn fern_cli(input: &Path, output: &Path, links: &[&std::ffi::OsStr]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_fern"));
    command.arg(input).arg("-o").arg(output).args(links);
    command
}
fn succeeds(command: &mut Command) -> std::process::Output {
    let result = command.output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    result
}
#[test]
fn c_abi_small_integers_and_struct_memory() {
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
    compile_c(&c, &object);
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
    succeeds(&mut fern_cli(
        &input,
        &output,
        &["--link".as_ref(), object.as_os_str()],
    ));
    assert!(Command::new(output).status().unwrap().success());
}
#[test]
fn c_abi_floating_point() {
    let dir = tempfile::tempdir().unwrap();
    let c = dir.path().join("helper.c");
    let object = dir.path().join("helper.o");
    fs::write(
        &c,
        r#"
 #include <stdint.h>
 float halve(float x){return x/2;}
 double mix(int32_t a,float b,double c,int64_t d){return a+(double)b*c+d;}
 double sum10(double a,double b,double c,double d,double e,double f,double g,double h,double i,double j){
 return a+b*2+c*3+d*4+e*5+f*6+g*7+h*8+i*9+j*10;
 }"#,
    )
    .unwrap();
    compile_c(&c, &object);
    let input = dir.path().join("main.fern");
    let output = dir.path().join("program");
    fs::write(
        &input,
        format!(
            r#"{EXIT}
 extern fn halve(x:f32)->f32;extern fn mix(a:i32,b:f32,c:f64,d:i64)->f64;
 extern fn sum10(a:f64,b:f64,c:f64,d:f64,e:f64,f:f64,g:f64,h:f64,i:f64,j:f64)->f64;
 extern fn sqrt(x:f64)->f64;
 type Pair struct {{small:f32,large:f64}}
 fn main()->void {{
 if halve(3.0)!=1.5 {{exit(1);}}
 if mix(3,0.5,8.0,-10)!=-3.0 {{exit(2);}}
 if sum10(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.5)!=50.0 {{exit(3);}}
 var pair=Pair {{small=5.0,large=2.25}};
 if halve(pair.small)!=2.5 || sqrt(pair.large)!=1.5 {{exit(4);}}
 }}"#
        ),
    )
    .unwrap();
    succeeds(&mut fern_cli(
        &input,
        &output,
        &[
            "--link".as_ref(),
            object.as_os_str(),
            "-l".as_ref(),
            "m".as_ref(),
        ],
    ));
    let status = Command::new(output).status().unwrap();
    assert!(status.success(), "{status}");
}
#[test]
fn c_variadic_calls() {
    let dir = tempfile::tempdir().unwrap();
    let c = dir.path().join("helper.c");
    let object = dir.path().join("helper.o");
    fs::write(
        &c,
        r#"
 #include <stdarg.h>
 #include <stdint.h>
 #include <string.h>
 int64_t take(int32_t count,...){
 va_list ap;va_start(ap,count);
 int a=va_arg(ap,int);int64_t b=va_arg(ap,int64_t);double c=va_arg(ap,double);
 char *d=va_arg(ap,char *);int e=va_arg(ap,int);unsigned f=va_arg(ap,unsigned);
 intptr_t g=va_arg(ap,intptr_t);double h=va_arg(ap,double);int i=va_arg(ap,int);
 int64_t j=va_arg(ap,int64_t);int k=va_arg(ap,int);
 va_end(ap);
 if(count!=11)return 1;if(a!=7)return 2;if(b!=-9000000000LL)return 3;if(c!=0.25)return 4;
 if(strcmp(d,"ok"))return 5;if(e!=-3)return 6;if(f!=65535)return 7;if(g!=5)return 8;
 if(h!=1.5)return 9;if(i!='a')return 10;if(j!=123)return 11;if(k!=200)return 12;
 return 0;
 }
 int32_t none(int32_t x,...){return x+1;}"#,
    )
    .unwrap();
    compile_c(&c, &object);
    let input = dir.path().join("main.fern");
    let output = dir.path().join("program");
    fs::write(
        &input,
        format!(
            r#"{EXIT}
 extern fn take(count:i32,...)->i64;
 extern fn none(x:i32,...)->i32;
 extern fn snprintf(buffer:*u8,size:uint,format:*u8,...)->i32;
 extern fn strcmp(a:*u8,b:*u8)->i32;
 extern fn malloc(bytes:uint)->*u8;
 fn main()->void {{
 var small:i8=-3;var wide:u16=65535;var byte:u8=200;
 var result=take(11,i32(7),i64(-9000000000),0.25,"ok",small,wide,5,1.5,'a',i64(123),byte);
 if result!=0 {{exit(i32(result));}}
 if none(41)!=42 {{exit(20);}}
 var buffer=malloc(32);
 if snprintf(buffer,32,"%d %s %.1f",i32(42),"x",2.5)!=8 {{exit(21);}}
 if strcmp(buffer,"42 x 2.5")!=0 {{exit(22);}}
 }}"#
        ),
    )
    .unwrap();
    succeeds(&mut fern_cli(
        &input,
        &output,
        &["--link".as_ref(), object.as_os_str()],
    ));
    let status = Command::new(output).status().unwrap();
    assert!(status.success(), "{status}");
}
#[test]
fn variadic_is_part_of_an_external_signature() {
    let dir = tempfile::tempdir().unwrap();
    for (path, source) in [
        ("lib/a.fern", "pub extern fn printf(f:*u8,...)->i32;"),
        (
            "app/main.fern",
            "use lib;extern fn printf(f:*u8)->i32;fn main()->void {lib::printf(\"\");}",
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
#[test]
fn c_abi_structs() {
    let dir = tempfile::tempdir().unwrap();
    let c = dir.path().join("helper.c");
    let object = dir.path().join("helper.o");
    fs::write(
        &c,
        r#"
 #include <stddef.h>
 #include <stdint.h>
 typedef struct {float x,y;} Vec2;
 typedef struct {uint8_t tag;int16_t small;double value;} Mixed;
 typedef struct {int32_t values[10];void *next;} Big;
 typedef struct {Vec2 corners[2];uint8_t id;} Box;
 Vec2 vec2_scale(Vec2 v,float k){Vec2 r={v.x*k,v.y*k};return r;}
 Mixed mixed_make(uint8_t tag,int16_t small,double value){Mixed m={tag,small,value};return m;}
 int64_t mixed_sum(Mixed m){return m.tag+m.small+(int64_t)(m.value*4);}
 Big big_make(int32_t start){Big b;for(int i=0;i<10;i++)b.values[i]=start+i;b.next=NULL;return b;}
 int32_t big_clobber(Big b){int32_t s=0;for(int i=0;i<10;i++){s+=b.values[i];b.values[i]=-1;}return s;}
 Box box_flip(Box b){Box r={{b.corners[1],b.corners[0]},b.id+1};return r;}
 double spill(Vec2 a,Vec2 b,Vec2 c,Vec2 d,Vec2 e,Mixed m){
 return a.x+b.x*2+c.x*3+d.x*4+e.x*5+m.small+m.value*6;
 }"#,
    )
    .unwrap();
    compile_c(&c, &object);
    let input = dir.path().join("main.fern");
    let output = dir.path().join("program");
    fs::write(
        &input,
        format!(
            r#"{EXIT}
 type Vec2 struct {{x:f32,y:f32}}
 type Mixed struct {{tag:u8,small:i16,value:f64}}
 type Big struct {{values:[10]i32,next:*u8}}
 type Box struct {{corners:[2]Vec2,id:u8}}
 type Holder struct {{v:Vec2,m:Mixed}}
 extern fn vec2_scale(v:Vec2,k:f32)->Vec2;
 extern fn mixed_make(tag:u8,small:i16,value:f64)->Mixed;
 extern fn mixed_sum(m:Mixed)->i64;
 extern fn big_make(start:i32)->Big;
 extern fn big_clobber(b:Big)->i32;
 extern fn box_flip(b:Box)->Box;
 extern fn spill(a:Vec2,b:Vec2,c:Vec2,d:Vec2,e:Vec2,m:Mixed)->f64;
 fn scaled(v:Vec2)->Vec2 {{return vec2_scale(v,2.0);}}
 fn main()->void {{
 var v=Vec2 {{x=1.5,y=-2.0}};
 var s=vec2_scale(v,2.0);
 if s.x!=3.0 || s.y!=-4.0 {{exit(1);}}
 var h=Holder {{v=v,m=mixed_make(1,2,3.0)}};
 h.m=mixed_make(200,-300,0.5);
 if h.m.tag!=200 || h.m.small!=-300 || h.m.value!=0.5 {{exit(2);}}
 if mixed_sum(h.m)!=-98 {{exit(3);}}
 mixed_make(9,9,9.0);
 var b=big_make(5);
 if b.next!=null || b.values[0]!=5 || b.values[9]!=14 {{exit(4);}}
 var p=&b;
 if big_clobber(*p)!=95 || b.values[0]!=5 || b.values[9]!=14 {{exit(5);}}
 var box=Box {{corners=[v,s],id=7}};
 var f=box_flip(box);
 if f.corners[0].x!=3.0 || f.corners[1].y!=-2.0 || f.id!=8 || box.id!=7 {{exit(6);}}
 if spill(v,s,h.v,f.corners[0],scaled(v),h.m)!=-258.0 {{exit(7);}}
 if scaled(h.v).y!=-4.0 {{exit(8);}}
 }}"#
        ),
    )
    .unwrap();
    succeeds(&mut fern_cli(
        &input,
        &output,
        &["--link".as_ref(), object.as_os_str()],
    ));
    let status = Command::new(output).status().unwrap();
    assert!(status.success(), "{status}");
}
#[cfg(unix)]
#[test]
fn named_static_libraries_link_in_command_line_order() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let (helper, scale) = (dir.path().join("helper.c"), dir.path().join("scale.c"));
    fs::write(
        &helper,
        "int scale(int); int helper(int x){return scale(x)+2;}",
    )
    .unwrap();
    fs::write(&scale, "int scale(int x){return x*10;}").unwrap();
    let (helper_object, scale_object) = (dir.path().join("helper.o"), dir.path().join("scale.o"));
    compile_c(&helper, &helper_object);
    compile_c(&scale, &scale_object);
    let archive = dir.path().join("libhelper.a");
    assert!(
        Command::new("ar")
            .arg("rcs")
            .arg(&archive)
            .arg(&helper_object)
            .status()
            .unwrap()
            .success()
    );
    let log = dir.path().join("cc-arguments");
    let wrapper = dir.path().join("cc-wrapper");
    let quoted = log.to_string_lossy().replace('\'', "'\\''");
    fs::write(
        &wrapper,
        format!("#!/bin/sh\nprintf '%s\\n' \"$@\" > '{quoted}'\nexec cc \"$@\"\n"),
    )
    .unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o755)).unwrap();
    let input = dir.path().join("main.fern");
    let output = dir.path().join("program");
    fs::write(
        &input,
        format!("{EXIT}\nextern fn helper(x:i32)->i32;\nfn main()->void {{if helper(4)!=42 {{exit(1);}}}}"),
    )
    .unwrap();
    let links = [
        "-L".as_ref(),
        dir.path().as_os_str(),
        "-l".as_ref(),
        "helper".as_ref(),
        "--link".as_ref(),
        scale_object.as_os_str(),
    ];
    succeeds(fern_cli(&input, &output, &links).env("CC", &wrapper));
    assert!(Command::new(&output).status().unwrap().success());
    let arguments = fs::read_to_string(&log).unwrap();
    let arguments: Vec<_> = arguments.lines().collect();
    assert!(arguments[0].ends_with("program.s"), "{arguments:?}");
    let (search, scale) = (dir.path().to_str().unwrap(), scale_object.to_str().unwrap());
    assert_eq!(arguments[1..6], ["-L", search, "-l", "helper", scale]);
    assert_eq!(arguments[6], "-o");
}
#[test]
fn system_libraries_link_by_name() {
    // FE_TONEAREST is 0 on every supported host; FE_UPWARD differs by architecture.
    let upward = if cfg!(target_arch = "aarch64") {
        0x0040_0000
    } else {
        0x0800
    };
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("main.fern");
    let output = dir.path().join("program");
    fs::write(
        &input,
        format!(
            r#"{EXIT}
extern fn fegetround()->i32;extern fn fesetround(mode:i32)->i32;extern fn puts(text:*u8)->i32;
fn main()->void {{
if fegetround()!=0 {{exit(1);}}
if fesetround({upward})!=0 || fegetround()!={upward} {{exit(2);}}
if fesetround(0)!=0 || fegetround()!=0 {{exit(3);}}
if puts("rounding restored")<0 {{exit(4);}}
}}"#
        ),
    )
    .unwrap();
    succeeds(&mut fern_cli(
        &input,
        &output,
        &["-l".as_ref(), "m".as_ref()],
    ));
    let result = succeeds(&mut Command::new(&output));
    assert_eq!(result.stdout, b"rounding restored\n");
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
            // Capture the output: an example may print something large, such as
            // the ray traced image.
            let result = Command::new(output).output().unwrap();
            assert!(
                result.status.success(),
                "{}\n{}",
                path.display(),
                String::from_utf8_lossy(&result.stderr)
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
