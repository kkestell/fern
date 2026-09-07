use crate::CompileError;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub(crate) struct Source {
    pub path: PathBuf,
    pub text: String,
}

impl Source {
    pub fn load(path: &Path) -> Result<Self, CompileError> {
        let bytes = fs::read(path)
            .map_err(|e| CompileError::new(format!("cannot read {}: {e}", path.display())))?;
        let text = String::from_utf8(bytes).map_err(|e| {
            CompileError::new(format!(
                "{}: invalid UTF-8 at byte {}",
                path.display(),
                e.utf8_error().valid_up_to()
            ))
        })?;
        Ok(Self {
            path: path.to_owned(),
            text,
        })
    }
}
