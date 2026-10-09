//! Annotation resolution, array lengths, and element-type rules.

use crate::{
    diagnostic::Diagnostic,
    frontend::syntax::{AnnotationKind, Expression, TypeAnnotation, find_call},
    types::{MAX_AGGREGATE_LAYOUT_BYTES, MAX_STRUCT_CONTAINMENT_DEPTH, Scalar, StructId, Type},
};

use la_arena::Idx;

use num_traits::ToPrimitive;

use super::model::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Placement {
    Value,
    PointerTarget,
}

impl CheckedProgram<'_> {
    /// Resolves a written annotation to its concrete type.
    pub(super) fn resolve_annotation(
        &mut self,
        annotation: Idx<TypeAnnotation>,
        scopes: &ScopeStack<'_>,
    ) -> Result<Type, Diagnostic> {
        self.resolve_written(annotation, scopes, Placement::Value)
    }

    fn resolve_written(
        &mut self,
        annotation: Idx<TypeAnnotation>,
        scopes: &ScopeStack<'_>,
        placement: Placement,
    ) -> Result<Type, Diagnostic> {
        let syntax = self.syntax;
        let written = &syntax.annotations[annotation];
        let ty = match &written.kind {
            AnnotationKind::Scalar(scalar) => Ok(Type::Scalar(*scalar)),
            AnnotationKind::Named(name) => {
                let id = self.resolve_type_name(name, scopes)?;
                if placement == Placement::Value {
                    self.resolve_struct_fields(id, &written.span, scopes)?;
                }
                Ok(self.struct_type(id))
            }
            AnnotationKind::Array { length, element } => {
                let (length, element) = (*length, *element);
                let length = self.array_length(length, scopes)?;
                let element = self.resolve_written(element, scopes, placement)?;
                Ok(Type::Array {
                    length,
                    element: Box::new(element),
                })
            }
            AnnotationKind::Pointer { target } => {
                let target = *target;
                let target_span = syntax.annotations[target].span.clone();
                let target = self.resolve_written(target, scopes, Placement::PointerTarget)?;
                // A target's layout is final once every struct it holds by
                // value has resolved; until then it waits for the module's
                // structs, so no partial struct layout is cached.
                if self.layout_final(&target) {
                    self.validate_aggregate_layout(&target, &target_span)?;
                } else {
                    self.pending_targets.push((target.clone(), target_span));
                }
                Ok(Type::Pointer {
                    target: Box::new(target),
                })
            }
        }?;
        if placement == Placement::Value {
            self.validate_aggregate_layout(&ty, &written.span)?;
        }
        Ok(ty)
    }

    /// Whether every struct this type holds by value has resolved fields, so
    /// that its layout no longer changes.
    fn layout_final(&self, ty: &Type) -> bool {
        match ty {
            Type::Scalar(_) | Type::Pointer { .. } => true,
            Type::Array { element, .. } => self.layout_final(element),
            Type::Struct(declared) => self.structs[declared.id.0].state == FieldState::Resolved,
        }
    }

    /// Validates the layouts of pointer targets that waited on the module's
    /// structs, which have all resolved by the time this runs.
    pub(super) fn validate_pending_targets(&mut self) -> Result<(), Diagnostic> {
        for (target, span) in std::mem::take(&mut self.pending_targets) {
            debug_assert!(self.layout_final(&target), "the module's structs resolved");
            self.validate_aggregate_layout(&target, &span)?;
        }
        Ok(())
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
    /// program uses the type by value. A struct reached again while it is
    /// resolving would contain itself and so would have no finite size.
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
            let ty = self.resolve_annotation(field.annotation, scopes)?;
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

/// Requires the literal to supply every element.
pub(super) fn check_element_count(
    span: std::ops::Range<usize>,
    ty: &Type,
    length: u64,
    count: usize,
) -> Result<(), Diagnostic> {
    if u64::try_from(count).ok() == Some(length) {
        Ok(())
    } else {
        Err(Diagnostic::new(
            span,
            format!("`{ty}` requires {length} elements, found {count}"),
        ))
    }
}
