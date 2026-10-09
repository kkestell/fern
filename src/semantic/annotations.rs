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

/// Rejects pointer syntax, which parses but is not yet checked.
pub(super) fn pointers_not_implemented(span: &std::ops::Range<usize>) -> Diagnostic {
    Diagnostic::new(span.clone(), "pointers are not yet implemented")
}

/// Rejects a program that writes a pointer type, takes an address, or calls
/// `new`, which Fern IR represents but native emission does not yet. Every
/// pointer value starts at one of these: `null` takes a nullable pointer type
/// from an annotation, and `*T(p)` writes one.
pub(crate) fn reject_pointer_syntax(syntax: &Syntax) -> Result<(), Diagnostic> {
    let annotation = syntax
        .annotations
        .values()
        .find(|annotation| matches!(annotation.kind, AnnotationKind::Pointer { .. }))
        .map(|annotation| &annotation.span);
    let address = || {
        syntax
            .expressions
            .values()
            .find(|expression| {
                matches!(
                    expression.kind,
                    ExpressionKind::AddressOf { .. } | ExpressionKind::New { .. }
                )
            })
            .map(|expression| &expression.span)
    };
    match annotation.or_else(address) {
        Some(span) => Err(pointers_not_implemented(span)),
        None => Ok(()),
    }
}

/// Where an annotation's type is used. A pointer's target is only referred
/// to, so resolving it neither resolves struct fields nor validates its
/// layout; a struct may therefore hold a pointer to itself. The enclosing
/// pointer validates the target's layout once it is final.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Placement {
    Value,
    PointerTarget,
}

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
        self.resolve_written(annotation, scopes, initializer, Placement::Value)
    }

    fn resolve_written(
        &mut self,
        annotation: Idx<TypeAnnotation>,
        scopes: &ScopeStack<'_>,
        initializer: Option<Idx<Expression>>,
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
                let length = match length {
                    Some(length) => self.array_length(length, scopes)?,
                    None => inferred_length(syntax, &written.span, initializer)?,
                };
                // Only the declaration's own annotation has an initializer, so
                // a nested `[_]` has nothing to take a length from.
                let element = self.resolve_written(element, scopes, None, placement)?;
                Ok(Type::Array {
                    length,
                    element: Box::new(element),
                })
            }
            AnnotationKind::Pointer { nullable, target } => {
                let (nullable, target) = (*nullable, *target);
                let target_span = syntax.annotations[target].span.clone();
                let target =
                    self.resolve_written(target, scopes, None, Placement::PointerTarget)?;
                // A target's layout is final once every struct it holds by
                // value has resolved; until then it waits for the module's
                // structs, so no partial struct layout is cached.
                if self.layout_final(&target) {
                    self.validate_aggregate_layout(&target, &target_span)?;
                } else {
                    self.pending_targets.push((target.clone(), target_span));
                }
                Ok(Type::Pointer {
                    nullable,
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
            self.structs[id.0].has_zero_value = self.structs[id.0]
                .fields
                .iter()
                .all(|field| self.has_zero_value(&field.ty));
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
