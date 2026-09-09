use std::{env, path::Path, process::ExitCode};

fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 3
        || args[1] != "-o"
        || args[0].is_empty()
        || args[0].to_string_lossy().starts_with('-')
        || args[2].is_empty()
        || args[2].to_string_lossy().starts_with('-')
    {
        eprintln!("usage: fern <root> -o <output>");
        return ExitCode::FAILURE;
    }
    match fern::compile(Path::new(&args[0]), Path::new(&args[2])) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
