use crate::{diagnostic::Diagnostic, frontend::Syntax};

#[derive(Debug)]
pub(crate) struct CheckedEntry {
    _private: (),
}

pub(crate) fn check(syntax: &Syntax) -> Result<CheckedEntry, Diagnostic> {
    let mut main = None;
    for (id, function) in syntax.functions.iter() {
        if syntax.names.resolve(&function.name) != "main" {
            return Err(Diagnostic::new(
                function.span.clone(),
                "only the `main` function is supported",
            ));
        }
        if main.replace(id).is_some() {
            return Err(Diagnostic::new(
                function.name_span.clone(),
                "duplicate `main` function",
            ));
        }
    }
    if main.is_none() {
        return Err(Diagnostic::new(0..0, "missing `main` function"));
    }
    Ok(CheckedEntry { _private: () })
}
