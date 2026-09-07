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
    let main = main.ok_or_else(|| Diagnostic::new(0..0, "missing `main` function"))?;
    if let Some(statement) = syntax.functions[main].body.first() {
        return Err(Diagnostic::new(
            syntax.statements[*statement].span.clone(),
            "nonempty bodies are not supported yet",
        ));
    }
    Ok(CheckedEntry { _private: () })
}
