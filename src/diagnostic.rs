use crate::{CompileError, source::Source};
use ariadne::{Config, IndexType, Label, Report, ReportKind};
use std::ops::Range;

#[derive(Debug)]
pub(crate) struct Diagnostic {
    pub span: Range<usize>,
    pub message: String,
}

impl Diagnostic {
    pub fn new(span: Range<usize>, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
        }
    }

    pub fn render(self, source: &Source) -> CompileError {
        let path = source.path.display().to_string();
        let span = (path.as_str(), self.span);
        let mut rendered = Vec::new();
        Report::build(ReportKind::Error, span.clone())
            .with_config(
                Config::default()
                    .with_color(false)
                    .with_index_type(IndexType::Byte),
            )
            .with_message(&self.message)
            .with_label(Label::new(span).with_message(&self.message))
            .finish()
            .write(
                (path.as_str(), ariadne::Source::from(&source.text)),
                &mut rendered,
            )
            .expect("writing diagnostics to a Vec cannot fail");
        CompileError::new(String::from_utf8(rendered).expect("diagnostics are UTF-8"))
    }
}
