use crate::CompileError;
use std::{
    fs,
    ops::Range,
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub(crate) struct Source {
    pub path: PathBuf,
    pub text: String,
    /// This file's start in the offset space its map shares across every file.
    pub base: usize,
}

impl Source {
    /// `span` in this file's own offsets, clamped to its text. A span never
    /// crosses the gap between two files, so clamping only trims a span
    /// pointing at the end of the file.
    pub fn local(&self, span: &Range<usize>) -> Range<usize> {
        let start = span.start.saturating_sub(self.base).min(self.text.len());
        let end = span.end.saturating_sub(self.base).min(self.text.len());
        start..end.max(start)
    }
}

/// The source files of one module, in load order, sharing a single offset
/// space. Every span in the compiler indexes that space, so a span identifies
/// both the file it points into and the position within it.
#[derive(Debug, Default)]
pub(crate) struct SourceMap {
    files: Vec<Source>,
}

impl SourceMap {
    /// Appends a file and returns its index. Its base leaves a one-byte gap
    /// after the previous text so an end-of-file span cannot collide with this
    /// file's first byte.
    pub fn push(&mut self, path: PathBuf, text: String) -> usize {
        let base = self
            .files
            .last()
            .map_or(0, |last| last.base + last.text.len() + 1);
        self.files.push(Source { path, text, base });
        self.files.len() - 1
    }

    pub fn files(&self) -> &[Source] {
        &self.files
    }

    /// The index of the file holding `offset`.
    pub fn index_at(&self, offset: usize) -> usize {
        // The first file's base is zero, so at least one file starts at or
        // before any offset.
        self.files.partition_point(|file| file.base <= offset) - 1
    }

    #[cfg(test)]
    pub fn from_text(text: &str) -> Self {
        Self::from_named_texts(&[("test.fern", text)])
    }

    #[cfg(test)]
    pub fn from_named_texts(files: &[(&str, &str)]) -> Self {
        let mut map = Self::default();
        for (path, text) in files {
            map.push(PathBuf::from(path), (*text).to_owned());
        }
        map
    }
}

/// A module directory's `.fern` files, sorted by file name, or an empty vector
/// when it holds none.
pub(crate) fn fern_files(directory: &Path) -> Result<Vec<PathBuf>, CompileError> {
    let entries = fs::read_dir(directory)
        .map_err(|e| CompileError::new(format!("cannot read {}: {e}", directory.display())))?;
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|e| CompileError::new(format!("cannot read {}: {e}", directory.display())))?
            .path();
        if path
            .extension()
            .is_some_and(|extension| extension == "fern")
            && path.is_file()
        {
            paths.push(path);
        }
    }
    // Directory order is not stable across platforms, and declaration order
    // reaches diagnostics and module-level initialization.
    paths.sort();
    Ok(paths)
}

pub(crate) fn read_text(path: &Path) -> Result<String, CompileError> {
    let bytes = fs::read(path)
        .map_err(|e| CompileError::new(format!("cannot read {}: {e}", path.display())))?;
    String::from_utf8(bytes).map_err(|e| {
        CompileError::new(format!(
            "{}: invalid UTF-8 at byte {}",
            path.display(),
            e.utf8_error().valid_up_to()
        ))
    })
}
