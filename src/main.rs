use fern::LinkInput;
use std::{
    env,
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    process::ExitCode,
};

const USAGE: &str =
    "usage: fern <root> -o <output> [--link <file> | -L <directory> | -l <name>]...";

fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let Some((root, output, links)) = parse(&args) else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };
    match fern::compile_with_links(Path::new(root), Path::new(output), &links) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn parse(args: &[OsString]) -> Option<(&OsStr, &OsStr, Vec<LinkInput>)> {
    let [root, flag, output, options @ ..] = args else {
        return None;
    };
    if flag != "-o" || !operand(root) || !operand(output) {
        return None;
    }
    let mut links = Vec::new();
    for pair in options.chunks(2) {
        let [option, value] = pair else {
            return None;
        };
        if !operand(value) {
            return None;
        }
        links.push(match option.to_str()? {
            "--link" => LinkInput::File(PathBuf::from(value)),
            "-L" => LinkInput::SearchPath(PathBuf::from(value)),
            "-l" => LinkInput::Library(value.clone()),
            _ => return None,
        });
    }
    Some((root, output, links))
}

fn operand(arg: &OsStr) -> bool {
    !arg.is_empty() && !arg.to_string_lossy().starts_with('-')
}
