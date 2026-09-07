use crate::{CompileError, semantic::CheckedEntry};

#[derive(Debug)]
pub(crate) struct Entry {
    status: i32,
}

#[derive(Debug)]
pub(crate) struct VerifiedEntry(Entry);

pub(crate) fn lower(_: CheckedEntry) -> Entry {
    Entry { status: 0 }
}

impl Entry {
    pub fn verify(self) -> Result<VerifiedEntry, CompileError> {
        if self.status != 0 {
            return Err(CompileError::new(
                "invalid IR: empty entry must complete with status zero",
            ));
        }
        Ok(VerifiedEntry(self))
    }
}

impl VerifiedEntry {
    pub fn status(&self) -> i32 {
        self.0.status
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_ir_cannot_reach_emission() {
        for status in [-1, 1, 256] {
            let result = Entry { status }
                .verify()
                .map(|verified| crate::backend::emit(&verified));
            assert!(result.is_err());
        }
    }
}
