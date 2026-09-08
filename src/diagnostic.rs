use crate::source::Source;
use ariadne::{Config, IndexType, Label, Report, ReportKind};
use std::{fmt::Write as _, ops::Range};

const MAX_RENDERED_LINE_BYTES: usize = 240;
const DIAGNOSTIC_CONTEXT_BYTES: usize = 60;
const MAX_RENDERED_SPAN_BYTES: usize = 80;

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

    pub fn render(&self, source: &Source) -> String {
        DiagnosticRenderer::new(source).render(self)
    }
}

pub(crate) struct DiagnosticRenderer<'a> {
    path: String,
    source: ariadne::Source<&'a str>,
}

impl<'a> DiagnosticRenderer<'a> {
    pub(crate) fn new(source: &'a Source) -> Self {
        Self {
            path: source.path.display().to_string(),
            source: ariadne::Source::from(source.text.as_str()),
        }
    }

    pub(crate) fn render(&self, diagnostic: &Diagnostic) -> String {
        if let Some((line, line_number, column)) = self.source.get_byte_line(diagnostic.span.start)
            && let Some(line_text) = self.source.get_line_text(line)
            && self.span_touches_long_line(diagnostic, line_number)
        {
            return self.render_long_line(diagnostic, line_text, line_number, column);
        }

        let span = (self.path.as_str(), diagnostic.span.clone());
        let mut rendered = Vec::new();
        Report::build(ReportKind::Error, span.clone())
            .with_config(
                Config::default()
                    .with_color(false)
                    .with_index_type(IndexType::Byte),
            )
            .with_message(&diagnostic.message)
            .with_label(Label::new(span).with_message(&diagnostic.message))
            .finish()
            .write((self.path.as_str(), &self.source), &mut rendered)
            .expect("writing diagnostics to a Vec cannot fail");
        String::from_utf8(rendered).expect("diagnostics are UTF-8")
    }

    fn span_touches_long_line(&self, diagnostic: &Diagnostic, start_line: usize) -> bool {
        let last_byte = diagnostic
            .span
            .end
            .saturating_sub(1)
            .max(diagnostic.span.start)
            .min(self.source.text().len());
        let end_line = self
            .source
            .get_byte_line(last_byte)
            .map_or(start_line, |(_, line, _)| line);
        (start_line..=end_line).any(|line| {
            self.source
                .line(line)
                .and_then(|line| self.source.get_line_text(line))
                .is_some_and(|text| text.len() > MAX_RENDERED_LINE_BYTES)
        })
    }

    fn render_long_line(
        &self,
        diagnostic: &Diagnostic,
        line_text: &str,
        line_number: usize,
        column: usize,
    ) -> String {
        let content = line_text.trim_end_matches(['\r', '\n', '\u{b}', '\u{c}']);
        let focus_start = previous_boundary(content, column.min(content.len()));
        let focus_end = next_boundary(
            content,
            focus_start
                .saturating_add(diagnostic.span.len().clamp(1, MAX_RENDERED_SPAN_BYTES))
                .min(content.len()),
        );
        let excerpt_start = previous_boundary(
            content,
            focus_start.saturating_sub(DIAGNOSTIC_CONTEXT_BYTES),
        );
        let excerpt_end = next_boundary(
            content,
            focus_end
                .saturating_add(DIAGNOSTIC_CONTEXT_BYTES)
                .min(content.len()),
        );
        let leading = excerpt_start > 0;
        let trailing = excerpt_end < content.len();
        let excerpt = &content[excerpt_start..excerpt_end];
        let marker_column =
            usize::from(leading) + content[excerpt_start..focus_start].chars().count();
        let mut rendered = String::new();
        writeln!(rendered, "Error: {}", diagnostic.message).unwrap();
        writeln!(
            rendered,
            "  --> {}:{}:{} (bytes {}..{})",
            self.path,
            line_number + 1,
            column + 1,
            diagnostic.span.start,
            diagnostic.span.end
        )
        .unwrap();
        rendered.push_str("   |\n");
        writeln!(
            rendered,
            "{:>3} | {}{}{}",
            line_number + 1,
            if leading { "…" } else { "" },
            excerpt,
            if trailing { "…" } else { "" }
        )
        .unwrap();
        writeln!(
            rendered,
            "   | {}^ {}",
            " ".repeat(marker_column),
            diagnostic.message
        )
        .unwrap();
        rendered
    }
}

fn previous_boundary(text: &str, mut index: usize) -> usize {
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn next_boundary(text: &str, mut index: usize) -> usize {
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn long_source_lines_render_as_bounded_excerpts() {
        let prefix = "fn main() -> void { const value = ";
        let digits = "9".repeat(200_000);
        let text = format!("{prefix}{digits}; }}");
        let source = Source {
            path: PathBuf::from("long.fern"),
            text,
        };
        let diagnostic = Diagnostic::new(
            prefix.len()..prefix.len() + digits.len(),
            "integer literal exceeds compiler limit",
        );
        let rendered = diagnostic.render(&source);
        assert!(
            rendered.len() < 1_000,
            "diagnostic was {} bytes",
            rendered.len()
        );
        assert!(rendered.contains("long.fern:1:"));
        assert!(rendered.contains(&format!(
            "bytes {}..{}",
            prefix.len(),
            prefix.len() + digits.len()
        )));
        assert!(rendered.contains('…'));
    }

    #[test]
    fn multiline_spans_ending_on_long_lines_are_bounded() {
        let prefix = "fn main() -> void {\nvar x: int = (\ntrue";
        let text = format!("{prefix}{});\n}}", " ".repeat(200_000));
        let source = Source {
            path: PathBuf::from("long.fern"),
            text,
        };
        let start = source.text.rfind('(').unwrap();
        let end = source.text.find(");").unwrap() + 1;
        let rendered = Diagnostic::new(start..end, "type mismatch").render(&source);
        assert!(
            rendered.len() < 1_000,
            "diagnostic was {} bytes",
            rendered.len()
        );
        assert!(rendered.contains("long.fern:2:"));
        assert!(rendered.contains(&format!("bytes {start}..{end}")));
    }
}
