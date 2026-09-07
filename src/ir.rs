use crate::{diagnostic::Diagnostic, semantic::CheckedEntry};

#[derive(Debug)]
pub(crate) struct Entry {
    pub status: i32,
}

pub(crate) fn lower(checked: CheckedEntry<'_>) -> Result<Entry, Diagnostic> {
    if let Some(statement) = checked.syntax.functions[checked.main].body.first() {
        return Err(Diagnostic::new(
            checked.syntax.statements[*statement].span.clone(),
            "lowering nonempty bodies is not supported yet",
        ));
    }
    Ok(Entry { status: 0 })
}
