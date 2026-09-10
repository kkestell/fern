use super::syntax::*;
use crate::{diagnostic::Diagnostic, source::SourceMap, types::Scalar};
use la_arena::Idx;
use std::ops::Range;

/// Spells a qualified name as it appeared in source.
pub(super) fn spell(syntax: &Syntax, name: &QualifiedName) -> String {
    match &name.qualifier {
        Some(qualifier) => format!(
            "{}::{}",
            syntax.names.resolve(&qualifier.name),
            syntax.names.resolve(&name.name)
        ),
        None => syntax.names.resolve(&name.name).to_owned(),
    }
}

/// Trails a qualified name's qualifier span, so plain names project
/// unchanged.
pub(super) fn qualifier(name: &QualifiedName) -> String {
    match &name.qualifier {
        Some(qualifier) => format!(" qualifier={:?}", qualifier.name_span),
        None => String::new(),
    }
}

fn spell_path(syntax: &Syntax, path: &[PathComponent], separator: &str) -> String {
    path.iter()
        .map(|component| syntax.names.resolve(&component.name))
        .collect::<Vec<_>>()
        .join(separator)
}

fn path_spans(path: &[PathComponent]) -> Vec<Range<usize>> {
    path.iter()
        .map(|component| component.name_span.clone())
        .collect()
}

pub(super) fn parse(text: &str) -> Result<Syntax, Diagnostic> {
    super::parser::parse(&SourceMap::from_text(text))
}

pub(super) fn projected(text: &str) -> String {
    let sources = SourceMap::from_text(text);
    project(&sources, &super::parser::parse(&sources).unwrap())
}

/// Projects each file's imports and items. A path header separates the
/// files of a multi-file module.
pub(super) fn project(sources: &SourceMap, syntax: &Syntax) -> String {
    use std::fmt::Write;
    let mut output = String::new();
    for (source, file) in sources.files().iter().zip(&syntax.files) {
        if syntax.files.len() > 1 {
            writeln!(output, "// {}", source.path.display()).unwrap();
        }
        project_file(syntax, file, &mut output);
    }
    output
}

fn project_file(syntax: &Syntax, file: &FileSyntax, output: &mut String) {
    use std::fmt::Write;
    for import in &file.imports {
        let path = spell_path(syntax, &import.path, "::");
        match &import.selection {
            Some(selection) => writeln!(
                output,
                "use {path}::{{{}}} span={:?} path={:?} selection={:?}",
                spell_path(syntax, selection, ", "),
                import.span,
                path_spans(&import.path),
                path_spans(selection),
            ),
            None => writeln!(
                output,
                "use {path} span={:?} path={:?}",
                import.span,
                path_spans(&import.path),
            ),
        }
        .unwrap();
    }
    for item in &file.items {
        match item {
            TopLevelItem::Function {
                function: id,
                public,
            } => {
                let function = &syntax.functions[*id];
                writeln!(
                    output,
                    "{}fn {} name={:?} span={:?}",
                    if *public { "pub " } else { "" },
                    syntax.names.resolve(&function.name),
                    function.name_span,
                    function.span
                )
                .unwrap();
                for parameter in &function.parameters {
                    writeln!(
                        output,
                        "  parameter {} name={:?}",
                        syntax.names.resolve(&parameter.name),
                        parameter.name_span,
                    )
                    .unwrap();
                    project_annotation(syntax, parameter.annotation, 2, output);
                }
                match function.result {
                    FunctionResult::Void => writeln!(output, "  result void").unwrap(),
                    FunctionResult::Value(annotation) => {
                        writeln!(output, "  result").unwrap();
                        project_annotation(syntax, annotation, 2, output);
                    }
                }
                project_body(syntax, &function.body, 1, output);
            }
            TopLevelItem::Binding {
                binding: id,
                public,
            } => {
                if *public {
                    output.push_str("pub ");
                }
                project_body(syntax, std::slice::from_ref(id), 0, output);
            }
            TopLevelItem::Struct {
                declaration: id,
                public,
            } => {
                let declaration = &syntax.structs[*id];
                writeln!(
                    output,
                    "{}type {} struct name={:?} span={:?}",
                    if *public { "pub " } else { "" },
                    syntax.names.resolve(&declaration.name),
                    declaration.name_span,
                    declaration.span
                )
                .unwrap();
                for field in &declaration.fields {
                    writeln!(
                        output,
                        "  field {} name={:?}",
                        syntax.names.resolve(&field.name),
                        field.name_span,
                    )
                    .unwrap();
                    project_annotation(syntax, field.annotation, 2, output);
                }
            }
        }
    }
}

fn project_body(syntax: &Syntax, body: &[Idx<Statement>], depth: usize, output: &mut String) {
    use std::fmt::Write;
    let indent = "  ".repeat(depth);
    for &id in body {
        let statement = &syntax.statements[id];
        match &statement.kind {
            StatementKind::Binding {
                mutable,
                name,
                name_span,
                annotation,
                initializer,
            } => {
                writeln!(
                    output,
                    "{indent}{} {} name={name_span:?} span={:?}",
                    if *mutable { "var" } else { "const" },
                    syntax.names.resolve(name),
                    statement.span
                )
                .unwrap();
                if let Some(annotation) = annotation {
                    project_annotation(syntax, *annotation, depth + 1, output);
                }
                project_expression(syntax, *initializer, depth + 1, output);
            }
            StatementKind::Assignment { target, value } => {
                writeln!(
                    output,
                    "{indent}assign {} name={:?} span={:?}{}",
                    spell(syntax, &target.name),
                    target.name.name_span,
                    statement.span,
                    qualifier(&target.name)
                )
                .unwrap();
                project_target_steps(syntax, target, depth + 1, output);
                project_expression(syntax, *value, depth + 1, output);
            }
            StatementKind::CompoundAssignment {
                target,
                operator,
                operator_span,
                value,
            } => {
                writeln!(
                    output,
                    "{indent}compound assign {} {operator:?}= name={:?} operator={operator_span:?} span={:?}{}",
                    spell(syntax, &target.name),
                    target.name.name_span,
                    statement.span,
                    qualifier(&target.name)
                )
                .unwrap();
                project_target_steps(syntax, target, depth + 1, output);
                project_expression(syntax, *value, depth + 1, output);
            }
            StatementKind::Block { body } => {
                writeln!(output, "{indent}block span={:?}", statement.span).unwrap();
                project_body(syntax, body, depth + 1, output);
            }
            StatementKind::Exit { argument } => {
                writeln!(output, "{indent}exit span={:?}", statement.span).unwrap();
                project_expression(syntax, *argument, depth + 1, output);
            }
            StatementKind::Call { call } => {
                writeln!(
                    output,
                    "{indent}call {} target={:?} left_paren={:?} right_paren={:?} span={:?}{}",
                    spell(syntax, &call.target),
                    call.target.name_span,
                    call.left_paren_span,
                    call.right_paren_span,
                    statement.span,
                    qualifier(&call.target)
                )
                .unwrap();
                for &argument in &call.arguments {
                    project_expression(syntax, argument, depth + 1, output);
                }
            }
            StatementKind::Return { value } => {
                writeln!(output, "{indent}return span={:?}", statement.span).unwrap();
                if let Some(value) = value {
                    project_expression(syntax, *value, depth + 1, output);
                }
            }
            StatementKind::If {
                condition,
                then_body,
                else_branch,
            } => {
                writeln!(output, "{indent}if span={:?}", statement.span).unwrap();
                project_expression(syntax, *condition, depth + 1, output);
                writeln!(output, "{indent}then").unwrap();
                project_body(syntax, then_body, depth + 1, output);
                if let Some(branch) = else_branch {
                    writeln!(output, "{indent}else").unwrap();
                    project_body(syntax, std::slice::from_ref(branch), depth + 1, output);
                }
            }
            StatementKind::For {
                label,
                header,
                body,
            } => {
                let label_name = label
                    .as_ref()
                    .map(|label| syntax.names.resolve(&label.name))
                    .unwrap_or("-");
                writeln!(
                    output,
                    "{indent}for label={label_name} label_span={:?} span={:?}",
                    label.as_ref().map(|label| &label.name_span),
                    statement.span
                )
                .unwrap();
                match header {
                    ForHeader::Infinite => writeln!(output, "{indent}  infinite").unwrap(),
                    ForHeader::Condition(condition) => {
                        writeln!(output, "{indent}  condition").unwrap();
                        project_expression(syntax, *condition, depth + 2, output);
                    }
                    ForHeader::ThreeClause {
                        initializer,
                        condition,
                        post,
                    } => {
                        writeln!(output, "{indent}  initializer").unwrap();
                        project_body(syntax, std::slice::from_ref(initializer), depth + 2, output);
                        writeln!(output, "{indent}  condition").unwrap();
                        project_expression(syntax, *condition, depth + 2, output);
                        writeln!(output, "{indent}  post").unwrap();
                        project_body(syntax, std::slice::from_ref(post), depth + 2, output);
                    }
                    ForHeader::Iteration {
                        value,
                        index,
                        operand,
                    } => {
                        writeln!(
                            output,
                            "{indent}  iteration value={} index={} index_span={:?}",
                            syntax.names.resolve(value),
                            index
                                .as_ref()
                                .map(|(name, _)| syntax.names.resolve(name))
                                .unwrap_or("-"),
                            index.as_ref().map(|(_, span)| span),
                        )
                        .unwrap();
                        project_expression(syntax, *operand, depth + 2, output);
                    }
                }
                writeln!(output, "{indent}  body").unwrap();
                project_body(syntax, body, depth + 2, output);
            }
            StatementKind::Break { label } | StatementKind::Continue { label } => {
                let keyword = if matches!(&statement.kind, StatementKind::Break { .. }) {
                    "break"
                } else {
                    "continue"
                };
                writeln!(
                    output,
                    "{indent}{keyword} label={} label_span={:?} span={:?}",
                    label
                        .as_ref()
                        .map(|label| syntax.names.resolve(&label.name))
                        .unwrap_or("-"),
                    label.as_ref().map(|label| &label.name_span),
                    statement.span
                )
                .unwrap();
            }
        }
    }
}

/// Projects an assignment target's steps, which precede the assigned value
/// in evaluation order.
fn project_target_steps(
    syntax: &Syntax,
    target: &AssignmentTarget,
    depth: usize,
    output: &mut String,
) {
    use std::fmt::Write;
    let indent = "  ".repeat(depth);
    for step in &target.steps {
        match step {
            TargetStep::Index(index) => {
                writeln!(output, "{indent}index").unwrap();
                project_expression(syntax, *index, depth + 1, output);
            }
            TargetStep::Field { name, name_span } => writeln!(
                output,
                "{indent}field {} name={name_span:?}",
                syntax.names.resolve(name)
            )
            .unwrap(),
        }
    }
}

fn project_annotation(syntax: &Syntax, id: Idx<TypeAnnotation>, depth: usize, output: &mut String) {
    use std::fmt::Write;
    let indent = "  ".repeat(depth);
    let annotation = &syntax.annotations[id];
    match &annotation.kind {
        AnnotationKind::Scalar(ty) => {
            writeln!(
                output,
                "{indent}scalar {} span={:?}",
                ty.name(),
                annotation.span
            )
            .unwrap();
        }
        AnnotationKind::Named(name) => {
            writeln!(
                output,
                "{indent}named {} span={:?}{}",
                spell(syntax, name),
                annotation.span,
                qualifier(name)
            )
            .unwrap();
        }
        AnnotationKind::Array { length, element } => {
            writeln!(output, "{indent}array span={:?}", annotation.span).unwrap();
            match length {
                Some(length) => {
                    writeln!(output, "{indent}  length").unwrap();
                    project_expression(syntax, *length, depth + 2, output);
                }
                None => writeln!(output, "{indent}  inferred length").unwrap(),
            }
            writeln!(output, "{indent}  element").unwrap();
            project_annotation(syntax, *element, depth + 2, output);
        }
    }
}

fn project_expression(syntax: &Syntax, id: Idx<Expression>, depth: usize, output: &mut String) {
    use std::fmt::Write;
    let indent = "  ".repeat(depth);
    let expression = &syntax.expressions[id];
    match &expression.kind {
        ExpressionKind::Integer(spelling) => writeln!(
            output,
            "{indent}integer {spelling} span={:?}",
            expression.span
        )
        .unwrap(),
        ExpressionKind::Floating(spelling) => writeln!(
            output,
            "{indent}floating {spelling} span={:?}",
            expression.span
        )
        .unwrap(),
        ExpressionKind::Boolean(value) => {
            writeln!(output, "{indent}boolean {value} span={:?}", expression.span).unwrap()
        }
        ExpressionKind::Reference(name) => writeln!(
            output,
            "{indent}reference {} span={:?}{}",
            spell(syntax, name),
            expression.span,
            qualifier(name)
        )
        .unwrap(),
        ExpressionKind::Grouping { expression: inner } => {
            writeln!(output, "{indent}group span={:?}", expression.span).unwrap();
            project_expression(syntax, *inner, depth + 1, output);
        }
        ExpressionKind::Unary {
            operator,
            operator_span,
            operand,
        } => {
            writeln!(
                output,
                "{indent}unary {operator:?} operator={operator_span:?} span={:?}",
                expression.span
            )
            .unwrap();
            project_expression(syntax, *operand, depth + 1, output);
        }
        ExpressionKind::Binary {
            operator,
            operator_span,
            left,
            right,
        } => {
            writeln!(
                output,
                "{indent}binary {operator:?} operator={operator_span:?} span={:?}",
                expression.span
            )
            .unwrap();
            project_expression(syntax, *left, depth + 1, output);
            project_expression(syntax, *right, depth + 1, output);
        }
        ExpressionKind::Comparison {
            operator,
            operator_span,
            left,
            right,
        } => {
            writeln!(
                output,
                "{indent}comparison {operator:?} operator={operator_span:?} span={:?}",
                expression.span
            )
            .unwrap();
            project_expression(syntax, *left, depth + 1, output);
            project_expression(syntax, *right, depth + 1, output);
        }
        ExpressionKind::Logical {
            operator,
            operator_span,
            left,
            right,
        } => {
            writeln!(
                output,
                "{indent}logical {operator:?} operator={operator_span:?} span={:?}",
                expression.span
            )
            .unwrap();
            project_expression(syntax, *left, depth + 1, output);
            project_expression(syntax, *right, depth + 1, output);
        }
        ExpressionKind::LogicalNot {
            operator_span,
            operand,
        } => {
            writeln!(
                output,
                "{indent}logical Not operator={operator_span:?} span={:?}",
                expression.span
            )
            .unwrap();
            project_expression(syntax, *operand, depth + 1, output);
        }
        ExpressionKind::Conversion {
            destination,
            truncating,
            operand,
        } => {
            writeln!(
                output,
                "{indent}{} conversion {} span={:?}",
                if *truncating { "truncating" } else { "checked" },
                destination.name(),
                expression.span
            )
            .unwrap();
            project_expression(syntax, *operand, depth + 1, output);
        }
        ExpressionKind::Call(call) => {
            writeln!(
                output,
                "{indent}call {} target={:?} left_paren={:?} right_paren={:?} span={:?}{}",
                spell(syntax, &call.target),
                call.target.name_span,
                call.left_paren_span,
                call.right_paren_span,
                expression.span,
                qualifier(&call.target)
            )
            .unwrap();
            for &argument in &call.arguments {
                project_expression(syntax, argument, depth + 1, output);
            }
        }
        ExpressionKind::ArrayLiteral { elements, fill } => {
            writeln!(
                output,
                "{indent}array literal fill={fill:?} span={:?}",
                expression.span
            )
            .unwrap();
            for &element in elements {
                project_expression(syntax, element, depth + 1, output);
            }
        }
        ExpressionKind::StructLiteral { name, fields, fill } => {
            writeln!(
                output,
                "{indent}struct literal {} name={:?} fill={fill:?} span={:?}{}",
                spell(syntax, name),
                name.name_span,
                expression.span,
                qualifier(name)
            )
            .unwrap();
            for field in fields {
                writeln!(
                    output,
                    "{indent}  field {} name={:?}",
                    syntax.names.resolve(&field.name),
                    field.name_span
                )
                .unwrap();
                project_expression(syntax, field.value, depth + 2, output);
            }
        }
        ExpressionKind::Field {
            operand,
            name,
            name_span,
        } => {
            writeln!(
                output,
                "{indent}field {} name={name_span:?} span={:?}",
                syntax.names.resolve(name),
                expression.span
            )
            .unwrap();
            project_expression(syntax, *operand, depth + 1, output);
        }
        ExpressionKind::Index { operand, index } => {
            writeln!(output, "{indent}index span={:?}", expression.span).unwrap();
            project_expression(syntax, *operand, depth + 1, output);
            project_expression(syntax, *index, depth + 1, output);
        }
        ExpressionKind::Length { operand } => {
            writeln!(output, "{indent}len span={:?}", expression.span).unwrap();
            project_expression(syntax, *operand, depth + 1, output);
        }
    }
}

mod lexer;
mod parser;
mod syntax;
