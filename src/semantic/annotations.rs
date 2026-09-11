//! Annotation resolution, array lengths, and element-type rules.

use crate::{
    diagnostic::Diagnostic,
    frontend::syntax::{
        AnnotationKind, Expression, ExpressionKind, Syntax, TypeAnnotation, find_call,
    },
    types::{MAX_AGGREGATE_LAYOUT_BYTES, MAX_STRUCT_CONTAINMENT_DEPTH, Scalar, StructId, Type},
};

use la_arena::Idx;

use num_traits::ToPrimitive;

use super::model::*;

/// The length `[_]` takes from the declaration's initializer, which must be an
/// array literal that states its own length.
pub(super) fn inferred_length(
    syntax: &Syntax,
    span: &std::ops::Range<usize>,
    initializer: Option<Idx<Expression>>,
) -> Result<u64, Diagnostic> {
    let literal = initializer.map(|id| &syntax.expressions[id].kind);
    let Some(ExpressionKind::ArrayLiteral { elements, fill }) = literal else {
        return Err(Diagnostic::new(
            span.clone(),
            "`[_]` requires an array-literal initializer",
        ));
    };
    if fill.is_some() {
        return Err(Diagnostic::new(
            span.clone(),
            "`[_]` cannot take a length from a literal with a fill",
        ));
    }
    Ok(u64::try_from(elements.len()).expect("a source file holds fewer elements than u64::MAX"))
}
impl CheckedProgram<'_> {
    /// Resolves a written annotation to the type it names. `initializer` is the
    /// declaration's initializer, which is where a `[_]` length comes from.
    pub(super) fn resolve_annotation(
        &mut self,
        annotation: Idx<TypeAnnotation>,
        scopes: &ScopeStack<'_>,
        initializer: Option<Idx<Expression>>,
    ) -> Result<Type, Diagnostic> {
        let syntax = self.syntax;
        let written = &syntax.annotations[annotation];
        let ty = match &written.kind {
            AnnotationKind::Scalar(scalar) => Ok(Type::Scalar(*scalar)),
            AnnotationKind::Named(name) => {
                let id = self.resolve_type_name(name, scopes)?;
                self.resolve_struct_fields(id, &written.span, scopes)?;
                Ok(self.struct_type(id))
            }
            AnnotationKind::Pointer { constant, target } => Ok(Type::Pointer {
                constant: *constant,
                target: Box::new(self.resolve_pointer_target(*target, scopes)?),
            }),
            AnnotationKind::Array { length, element } => {
                let (length, element) = (*length, *element);
                let length = match length {
                    Some(length) => self.array_length(length, scopes)?,
                    None => inferred_length(syntax, &written.span, initializer)?,
                };
                // Only the declaration's own annotation has an initializer, so
                // a nested `[_]` has nothing to take a length from.
                let element = self.resolve_annotation(element, scopes, None)?;
                Ok(Type::Array {
                    length,
                    element: Box::new(element),
                })
            }
        }?;
        if !matches!(&written.kind, AnnotationKind::Pointer { .. }) {
            self.validate_aggregate_layout(&ty, &written.span)?;
        }
        Ok(ty)
    }

    /// Resolves the type a pointer reaches without following its inline layout.
    /// A pointer's representation is independent of that target, so a pointer
    /// field may be the edge that breaks a recursive struct declaration.
    fn resolve_pointer_target(
        &mut self,
        annotation: Idx<TypeAnnotation>,
        scopes: &ScopeStack<'_>,
    ) -> Result<Type, Diagnostic> {
        let syntax = self.syntax;
        let written = &syntax.annotations[annotation];
        match &written.kind {
            AnnotationKind::Scalar(scalar) => Ok(Type::Scalar(*scalar)),
            AnnotationKind::Named(name) => {
                let id = self.resolve_type_name(name, scopes)?;
                Ok(self.struct_type(id))
            }
            AnnotationKind::Pointer { constant, target } => Ok(Type::Pointer {
                constant: *constant,
                target: Box::new(self.resolve_pointer_target(*target, scopes)?),
            }),
            AnnotationKind::Array { length, element } => {
                let Some(length) = length else {
                    return Err(Diagnostic::new(
                        written.span.clone(),
                        "`[_]` requires an array-literal initializer",
                    ));
                };
                Ok(Type::Array {
                    length: self.array_length(*length, scopes)?,
                    element: Box::new(self.resolve_pointer_target(*element, scopes)?),
                })
            }
        }
    }

    /// Rejects layouts that native emission cannot address before they reach
    /// constant construction, IR lowering, or the backend.
    pub(super) fn validate_aggregate_layout(
        &self,
        ty: &Type,
        span: &std::ops::Range<usize>,
    ) -> Result<(), Diagnostic> {
        if self
            .layouts
            .size(self, ty)
            .is_none_or(|size| size > MAX_AGGREGATE_LAYOUT_BYTES)
        {
            return Err(Diagnostic::new(
                span.clone(),
                "aggregate layout exceeds compiler limit of 1 PiB",
            ));
        }
        Ok(())
    }

    /// Resolves one struct's field types, which happens the first time the
    /// program names the type. A struct reached again while it is resolving
    /// would contain itself and so would have no finite size.
    pub(super) fn resolve_struct_fields(
        &mut self,
        id: StructId,
        span: &std::ops::Range<usize>,
        scopes: &ScopeStack<'_>,
    ) -> Result<(), Diagnostic> {
        match self.structs[id.0].state {
            FieldState::Resolved => return Ok(()),
            FieldState::Resolving => {
                return Err(Diagnostic::new(
                    span.clone(),
                    format!("recursive struct type `{}`", self.struct_type(id)),
                ));
            }
            FieldState::Unresolved => {}
        }
        if self.struct_containment_depth == MAX_STRUCT_CONTAINMENT_DEPTH {
            return Err(Diagnostic::new(
                span.clone(),
                format!(
                    "struct containment exceeds compiler limit of {MAX_STRUCT_CONTAINMENT_DEPTH}"
                ),
            ));
        }
        self.structs[id.0].state = FieldState::Resolving;
        self.struct_containment_depth += 1;
        // Field annotations resolve against the imports of the file the
        // struct is declared in, which is not always the file being checked.
        let outer = std::mem::replace(&mut self.file, self.structs[id.0].file);
        let result = self.resolve_fields(id, scopes);
        // The checking state this walk swapped belongs to the caller, so it is
        // restored whether a field resolved or reported a diagnostic.
        self.struct_containment_depth -= 1;
        self.file = outer;
        if result.is_ok() {
            self.structs[id.0].state = FieldState::Resolved;
        } else {
            // A struct that failed to resolve holds only the fields the walk
            // reached, so it returns to the state it had before the attempt.
            self.structs[id.0].state = FieldState::Unresolved;
            self.structs[id.0].fields.clear();
            self.structs[id.0].ordinals.clear();
        }
        result
    }

    /// Resolves each declared field in order, rejecting a name the struct
    /// already holds.
    fn resolve_fields(&mut self, id: StructId, scopes: &ScopeStack<'_>) -> Result<(), Diagnostic> {
        let syntax = self.syntax;
        let declaration = self.structs[id.0].declaration;
        for field in &syntax.structs[declaration].fields {
            if self.structs[id.0].ordinals.contains_key(&field.name) {
                return Err(Diagnostic::new(
                    field.name_span.clone(),
                    format!(
                        "duplicate field name `{}`",
                        syntax.names.resolve(&field.name)
                    ),
                ));
            }
            let ty = self.resolve_annotation(field.annotation, scopes, None)?;
            let ordinal = self.structs[id.0].fields.len();
            self.structs[id.0].ordinals.insert(field.name, ordinal);
            self.structs[id.0].fields.push(CheckedField {
                name: field.name,
                ty,
            });
        }
        Ok(())
    }

    /// Evaluates a written array length, which is a constant `int` of at least
    /// one.
    fn array_length(
        &mut self,
        length: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<u64, Diagnostic> {
        let span = self.syntax.expressions[length].span.clone();
        // A length is resolved before the module's signatures are, so a call
        // here has no signature to check against. It is never constant anyway.
        let constant = |span: std::ops::Range<usize>| {
            Diagnostic::new(span, "array length must be a constant expression")
        };
        if let Some(call) = find_call(self.syntax, length) {
            return Err(constant(self.syntax.expressions[call].span.clone()));
        }
        let checked = self.check_expression(length, scopes, Some(Scalar::Int.into()))?;
        let Some(value) = checked.integer().cloned() else {
            return Err(constant(span));
        };
        self.expressions.insert(length, checked);
        value
            .to_u64()
            .filter(|length| *length >= 1)
            .ok_or_else(|| Diagnostic::new(span, "array length must be at least 1"))
    }
}

/// Checks a literal's element count against the array's length. A fill covers
/// the remaining elements, so it only bounds the count from above.
pub(super) fn check_element_count(
    span: std::ops::Range<usize>,
    ty: &Type,
    length: u64,
    count: usize,
    fill: bool,
) -> Result<(), Diagnostic> {
    let count = u64::try_from(count).expect("a source file holds fewer elements than u64::MAX");
    if if fill {
        count <= length
    } else {
        count == length
    } {
        return Ok(());
    }
    let bound = if fill { "at most " } else { "" };
    Err(Diagnostic::new(
        span,
        format!("expected {bound}{length} elements for `{ty}`, found {count}"),
    ))
}
