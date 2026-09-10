use crate::{
    diagnostic::Diagnostic,
    frontend::{
        AnnotationKind, AssignmentTarget, BinaryOperator, Call, ComparisonOperator, Expression,
        ExpressionKind, ForHeader, Function, FunctionResult, Import, Label, LogicalOperator,
        PathComponent, QualifiedName, Statement, StatementKind, Syntax, TopLevelItem,
        TypeAnnotation, UnaryOperator, integer_parts,
    },
    module::Module,
    types::{Scalar, Type},
};
use la_arena::{Arena, ArenaMap, Idx};
use lasso::Spur;
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use std::collections::{HashMap, HashSet};

/// The length `[_]` takes from the declaration's initializer, which must be an
/// array literal that states its own length.
fn inferred_length(
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

/// The value a constant expression folds to. An array literal folds when
/// every element does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Constant {
    Integer(BigInt),
    Array(Vec<Constant>),
}

impl Constant {
    /// The integer this constant folded to, or `None` when it folded to an
    /// array.
    pub(crate) fn integer(&self) -> Option<&BigInt> {
        match self {
            Self::Integer(value) => Some(value),
            Self::Array(_) => None,
        }
    }
}

impl From<BigInt> for Constant {
    fn from(value: BigInt) -> Self {
        Self::Integer(value)
    }
}

#[derive(Debug)]
pub(crate) struct Binding {
    pub ty: Type,
    pub mutable: bool,
    pub constant: Option<Constant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExpressionValue {
    Integer,
    Boolean,
    Reference(Idx<Binding>),
    Grouping {
        expression: Idx<Expression>,
    },
    Conversion {
        operand: Idx<Expression>,
        truncating: bool,
    },
    Unary {
        operator: UnaryOperator,
        operator_span: std::ops::Range<usize>,
        operand: Idx<Expression>,
    },
    Binary {
        operator: BinaryOperator,
        operator_span: std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    Comparison {
        operator: ComparisonOperator,
        operator_span: std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    Logical {
        operator: LogicalOperator,
        operator_span: std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    LogicalNot {
        operator_span: std::ops::Range<usize>,
        operand: Idx<Expression>,
    },
    Call {
        function: Idx<Function>,
    },
    /// An array literal, whose fill repeats the last element across the
    /// array's remaining elements.
    Array {
        elements: Vec<Idx<Expression>>,
        fill: bool,
    },
    Index {
        operand: Idx<Expression>,
        index: Idx<Expression>,
    },
    /// `len(a)`, which reads its length from the operand's type and still
    /// evaluates the operand.
    Length {
        operand: Idx<Expression>,
    },
}

#[derive(Debug)]
pub(crate) struct CheckedExpression {
    pub ty: Type,
    pub untyped: bool,
    pub value: ExpressionValue,
    pub constant: Option<Constant>,
}

impl CheckedExpression {
    /// The integer this expression folded to, or `None` when it did not fold
    /// or folded to an array.
    pub(crate) fn integer(&self) -> Option<&BigInt> {
        self.constant.as_ref().and_then(Constant::integer)
    }
}

struct CheckedBinaryOperand {
    id: Option<Idx<Expression>>,
    expression: CheckedExpression,
}

#[derive(Debug, Clone)]
pub(crate) struct FunctionSignature {
    pub parameters: Vec<Idx<Binding>>,
    pub result: Option<Type>,
}

/// Where an assignment stores: the binding its indices start from and the type
/// of the element they reach, which is the binding's own type when it has no
/// indices.
#[derive(Debug)]
pub(crate) struct CheckedTarget {
    pub binding: Idx<Binding>,
    pub ty: Type,
}

/// The bindings a `for … in` statement introduces. They belong to the
/// statement rather than to a declaration, so they are recorded on their own.
#[derive(Debug)]
pub(crate) struct IterationBindings {
    pub value: Idx<Binding>,
    pub index: Option<Idx<Binding>>,
}

#[derive(Debug)]
pub(crate) struct CheckedProgram<'a> {
    pub syntax: &'a Syntax,
    pub main: Idx<Function>,
    pub module_bindings: Vec<Idx<Statement>>,
    pub expressions: ArenaMap<Idx<Expression>, CheckedExpression>,
    pub declarations: ArenaMap<Idx<Statement>, Idx<Binding>>,
    pub bindings: Arena<Binding>,
    pub assignments: ArenaMap<Idx<Statement>, CheckedTarget>,
    pub iterations: ArenaMap<Idx<Statement>, IterationBindings>,
    pub functions: ArenaMap<Idx<Function>, FunctionSignature>,
    pub calls: ArenaMap<Idx<Statement>, Idx<Function>>,
    /// Checking state: the module-level functions of the module being checked,
    /// which an unqualified call resolves against. It is replaced per module,
    /// so it never describes the whole program.
    function_names: HashMap<Spur, Idx<Function>>,
    /// Checking state: each checked module's namespace, in module order, which
    /// a qualified name reaches through its file's imports.
    namespaces: Vec<Namespace>,
    /// Checking state: each source file's imported names, indexed by file.
    imports: Vec<FileImports>,
    /// Checking state: the file whose declarations are being checked, which
    /// selects the imports name resolution sees.
    file: usize,
}

/// A module-level declaration, as another file or module sees it.
#[derive(Debug, Clone, Copy)]
struct Declaration {
    public: bool,
    kind: DeclarationKind,
}

#[derive(Debug, Clone, Copy)]
enum DeclarationKind {
    Binding(Idx<Binding>),
    Function(Idx<Function>),
}

/// One module's declarations by name.
type Namespace = HashMap<Spur, Declaration>;

/// What a name introduced by a `use` declaration refers to. A whole-module
/// import holds the module's index into `CheckedProgram::namespaces` rather
/// than a copy of its namespace.
#[derive(Debug, Clone, Copy)]
enum Imported {
    Module(usize),
    Declaration(DeclarationKind),
}

/// What a name is being resolved as, which selects the wording of the
/// diagnostics an unqualified name shares between call and value position.
#[derive(Debug, Clone, Copy)]
enum Wanted {
    Function,
    Value,
}

impl Wanted {
    /// The tail of "module `m` is not ...".
    fn article_noun(self) -> &'static str {
        match self {
            Self::Function => "a function",
            Self::Value => "a value",
        }
    }

    /// The head of "... `n`", naming what was looked for and not found.
    fn unknown(self) -> &'static str {
        match self {
            Self::Function => "unknown function",
            Self::Value => "unknown binding",
        }
    }
}

/// One file's imported names. `introduced` keeps declaration order and spans,
/// so an unreferenced import reports the first one.
#[derive(Debug, Default)]
struct FileImports {
    names: HashMap<Spur, Imported>,
    introduced: Vec<(Spur, std::ops::Range<usize>)>,
    used: HashSet<Spur>,
}

/// The module-level bindings of one module, in source order.
#[derive(Default)]
struct ModuleBindings {
    /// The statement declaring each binding, by name.
    declarations: HashMap<Spur, Idx<Statement>>,
    statements: Vec<Idx<Statement>>,
    /// The file each binding is declared in.
    files: HashMap<Idx<Statement>, usize>,
}

/// Records a module-level name, rejecting a second declaration of it.
fn claim_module_name(
    names: &mut HashSet<Spur>,
    name: Spur,
    span: &std::ops::Range<usize>,
    syntax: &Syntax,
) -> Result<(), Diagnostic> {
    if names.insert(name) {
        return Ok(());
    }
    Err(Diagnostic::new(
        span.clone(),
        format!(
            "duplicate module-level name `{}`",
            syntax.names.resolve(&name)
        ),
    ))
}

fn check_parameter_names(function: &Function, syntax: &Syntax) -> Result<(), Diagnostic> {
    let mut names = HashSet::new();
    for parameter in &function.parameters {
        if !names.insert(parameter.name) {
            return Err(Diagnostic::new(
                parameter.name_span.clone(),
                format!(
                    "duplicate parameter name `{}`",
                    syntax.names.resolve(&parameter.name)
                ),
            ));
        }
    }
    Ok(())
}

/// Adds one name introduced by a `use` declaration to its file, rejecting a
/// name the file's module already declares or the file already imports.
fn introduce(
    file: &mut FileImports,
    namespace: &Namespace,
    component: &PathComponent,
    imported: Imported,
    syntax: &Syntax,
) -> Result<(), Diagnostic> {
    let name = syntax.names.resolve(&component.name);
    if namespace.contains_key(&component.name) {
        return Err(Diagnostic::new(
            component.name_span.clone(),
            format!("imported name `{name}` conflicts with a module-level declaration"),
        ));
    }
    if file.names.contains_key(&component.name) {
        return Err(Diagnostic::new(
            component.name_span.clone(),
            format!("duplicate imported name `{name}`"),
        ));
    }
    file.names.insert(component.name, imported);
    file.introduced
        .push((component.name, component.name_span.clone()));
    Ok(())
}

pub(crate) fn check<'a>(
    syntax: &'a Syntax,
    modules: &[Module],
    imports: &[Vec<usize>],
) -> Result<CheckedProgram<'a>, Diagnostic> {
    let root = modules.last().expect("a program has a root module");
    let mut checked = CheckedProgram {
        syntax,
        main: entry_point(syntax, root)?,
        module_bindings: Vec::new(),
        expressions: ArenaMap::default(),
        declarations: ArenaMap::default(),
        bindings: Arena::default(),
        assignments: ArenaMap::default(),
        iterations: ArenaMap::default(),
        functions: ArenaMap::default(),
        calls: ArenaMap::default(),
        function_names: HashMap::new(),
        namespaces: Vec::new(),
        imports: syntax
            .files
            .iter()
            .map(|_| FileImports::default())
            .collect(),
        file: 0,
    };
    // Dependencies come before dependents, so an imported module's namespace is
    // built and its bindings are ordered before any module that imports it.
    for module in modules {
        checked.check_module(module, imports)?;
    }
    Ok(checked)
}

/// Checks `syntax` as one root module with no imports.
#[cfg(test)]
pub(crate) fn check_root(syntax: &Syntax) -> Result<CheckedProgram<'_>, Diagnostic> {
    let (modules, imports) = crate::module::single(syntax);
    check(syntax, &modules, &imports)
}

/// The root module's sole `main`, with the entry-point signature. A `main` in a
/// dependency module is an ordinary function.
fn entry_point(syntax: &Syntax, root: &Module) -> Result<Idx<Function>, Diagnostic> {
    let mut main = None;
    for file in root.files.clone() {
        for item in &syntax.files[file].items {
            let TopLevelItem::Function { function: id, .. } = *item else {
                continue;
            };
            if syntax.names.resolve(&syntax.functions[id].name) != "main" {
                continue;
            }
            if main.is_some() {
                return Err(Diagnostic::new(
                    syntax.functions[id].name_span.clone(),
                    "duplicate `main` function",
                ));
            }
            check_entry_signature(syntax, &syntax.functions[id])?;
            main = Some(id);
        }
    }
    main.ok_or_else(|| Diagnostic::new(0..0, "missing `main` function"))
}

/// The entry point takes nothing and returns nothing. A program leaves through
/// `exit`, not through a value `main` returns.
fn check_entry_signature(syntax: &Syntax, main: &Function) -> Result<(), Diagnostic> {
    if let Some(parameter) = main.parameters.first() {
        return Err(Diagnostic::new(
            parameter.name_span.clone(),
            "`main` must not have parameters",
        ));
    }
    if let FunctionResult::Value(annotation) = main.result {
        return Err(Diagnostic::new(
            syntax.annotations[annotation].span.clone(),
            "`main` must return `void`",
        ));
    }
    Ok(())
}

impl CheckedProgram<'_> {
    /// Checks one module: its module-level names, each of its files' imports,
    /// its module-level initializers, its signatures, and its function bodies.
    /// Appends the namespace that later modules import from.
    ///
    /// Signatures are resolved after the module's initializers because an array
    /// length in a signature may name a module-level `const`, which only has a
    /// value once its own initializer is checked.
    fn check_module(&mut self, module: &Module, imports: &[Vec<usize>]) -> Result<(), Diagnostic> {
        let items = self.module_items(module);
        let bindings = self.declare_module_names(&items)?;
        let mut namespace = Namespace::new();
        let module_scope = self.declare_module_bindings(&items, &mut namespace)?;
        self.declare_function_names(&items, &mut namespace);
        self.resolve_imports(module, imports, &namespace)?;
        self.check_module_initializers(&bindings, &module_scope)?;
        self.resolve_signatures(&items, &module_scope)?;
        self.check_function_bodies(&items, &module_scope)?;
        self.check_imports_used(module)?;
        self.namespaces.push(namespace);
        Ok(())
    }

    /// Every top-level item of the module, paired with the file it comes from.
    fn module_items(&self, module: &Module) -> Vec<(usize, TopLevelItem)> {
        let syntax = self.syntax;
        module
            .files
            .clone()
            .flat_map(|file| {
                syntax.files[file]
                    .items
                    .iter()
                    .map(move |item| (file, *item))
            })
            .collect()
    }

    /// Checks that module-level names and each function's parameter names are
    /// unique, and records where the module's functions and bindings are
    /// declared.
    fn declare_module_names(
        &mut self,
        items: &[(usize, TopLevelItem)],
    ) -> Result<ModuleBindings, Diagnostic> {
        let syntax = self.syntax;
        let mut names = HashSet::new();
        let mut bindings = ModuleBindings::default();
        self.function_names = HashMap::new();
        for &(file, item) in items {
            match item {
                TopLevelItem::Function { function: id, .. } => {
                    let function = &syntax.functions[id];
                    claim_module_name(&mut names, function.name, &function.name_span, syntax)?;
                    check_parameter_names(function, syntax)?;
                    self.function_names.insert(function.name, id);
                }
                TopLevelItem::Binding { binding: id, .. } => {
                    let StatementKind::Binding {
                        name, name_span, ..
                    } = &syntax.statements[id].kind
                    else {
                        unreachable!("frontend only permits bindings at module level")
                    };
                    claim_module_name(&mut names, *name, name_span, syntax)?;
                    bindings.declarations.insert(*name, id);
                    bindings.statements.push(id);
                    bindings.files.insert(id, file);
                }
            }
        }
        Ok(bindings)
    }

    /// Allocates a binding for each module-level binding. The type stands in
    /// until the initializer is checked, which is what fills it in.
    fn declare_module_bindings(
        &mut self,
        items: &[(usize, TopLevelItem)],
        namespace: &mut Namespace,
    ) -> Result<HashMap<Spur, Idx<Binding>>, Diagnostic> {
        let syntax = self.syntax;
        let mut module_scope = HashMap::new();
        for &(_, item) in items {
            let TopLevelItem::Binding {
                binding: statement,
                public,
            } = item
            else {
                continue;
            };
            let StatementKind::Binding { name, mutable, .. } = &syntax.statements[statement].kind
            else {
                unreachable!("frontend only permits bindings at module level")
            };
            let binding = self.bindings.alloc(Binding {
                ty: Scalar::Int.into(),
                mutable: *mutable,
                constant: None,
            });
            self.declarations.insert(statement, binding);
            module_scope.insert(*name, binding);
            namespace.insert(
                *name,
                Declaration {
                    public,
                    kind: DeclarationKind::Binding(binding),
                },
            );
        }
        Ok(module_scope)
    }

    /// Puts every function in the module's namespace, so an import can name it
    /// before its signature is resolved.
    fn declare_function_names(
        &mut self,
        items: &[(usize, TopLevelItem)],
        namespace: &mut Namespace,
    ) {
        for &(_, item) in items {
            let TopLevelItem::Function { function, public } = item else {
                continue;
            };
            namespace.insert(
                self.syntax.functions[function].name,
                Declaration {
                    public,
                    kind: DeclarationKind::Function(function),
                },
            );
        }
    }

    /// Resolves every function's parameter and result types, so a call can be
    /// checked before the called function's body is.
    fn resolve_signatures(
        &mut self,
        items: &[(usize, TopLevelItem)],
        module_scope: &HashMap<Spur, Idx<Binding>>,
    ) -> Result<(), Diagnostic> {
        let syntax = self.syntax;
        let scopes = std::slice::from_ref(module_scope);
        for &(file, item) in items {
            let TopLevelItem::Function { function, .. } = item else {
                continue;
            };
            self.file = file;
            let function_syntax = &syntax.functions[function];
            let mut parameters = Vec::new();
            for parameter in &function_syntax.parameters {
                let ty = self.resolve_annotation(parameter.annotation, scopes, None)?;
                parameters.push(self.bindings.alloc(Binding {
                    ty,
                    mutable: false,
                    constant: None,
                }));
            }
            let result = match syntax.functions[function].result {
                FunctionResult::Void => None,
                FunctionResult::Value(annotation) => {
                    Some(self.resolve_annotation(annotation, scopes, None)?)
                }
            };
            self.functions
                .insert(function, FunctionSignature { parameters, result });
        }
        Ok(())
    }

    /// Resolves a written annotation to the type it names. `initializer` is the
    /// declaration's initializer, which is where a `[_]` length comes from.
    fn resolve_annotation(
        &mut self,
        annotation: Idx<TypeAnnotation>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
        initializer: Option<Idx<Expression>>,
    ) -> Result<Type, Diagnostic> {
        let syntax = self.syntax;
        let written = &syntax.annotations[annotation];
        match &written.kind {
            AnnotationKind::Named(scalar) => Ok(Type::Scalar(*scalar)),
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
        }
    }

    /// Evaluates a written array length, which is a constant `int` of at least
    /// one.
    fn array_length(
        &mut self,
        length: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
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

    fn resolve_imports(
        &mut self,
        module: &Module,
        imports: &[Vec<usize>],
        namespace: &Namespace,
    ) -> Result<(), Diagnostic> {
        let syntax = self.syntax;
        for file in module.files.clone() {
            let mut file_imports = FileImports::default();
            for (index, import) in syntax.files[file].imports.iter().enumerate() {
                self.resolve_import(import, imports[file][index], namespace, &mut file_imports)?;
            }
            self.imports[file] = file_imports;
        }
        Ok(())
    }

    /// Introduces the names one `use` declaration brings into its file, either
    /// the module itself or the declarations it selects.
    fn resolve_import(
        &self,
        import: &Import,
        resolved: usize,
        namespace: &Namespace,
        file_imports: &mut FileImports,
    ) -> Result<(), Diagnostic> {
        let syntax = self.syntax;
        let Some(selection) = &import.selection else {
            let last = import.path.last().expect("an import path has components");
            let imported = Imported::Module(resolved);
            return introduce(file_imports, namespace, last, imported, syntax);
        };
        let path: Vec<&str> = import
            .path
            .iter()
            .map(|component| syntax.names.resolve(&component.name))
            .collect();
        let path = path.join("::");
        let target = &self.namespaces[resolved];
        for component in selection {
            let name = syntax.names.resolve(&component.name);
            let Some(declaration) = target.get(&component.name).copied() else {
                return Err(Diagnostic::new(
                    component.name_span.clone(),
                    format!("module `{path}` has no declaration named `{name}`"),
                ));
            };
            if !declaration.public {
                return Err(Diagnostic::new(
                    component.name_span.clone(),
                    format!("declaration `{name}` is private to module `{path}`"),
                ));
            }
            let imported = Imported::Declaration(declaration.kind);
            introduce(file_imports, namespace, component, imported, syntax)?;
        }
        Ok(())
    }

    /// Checks module-level initializers in dependency order, so a binding's
    /// value is known before the bindings that reference it are checked.
    fn check_module_initializers(
        &mut self,
        bindings: &ModuleBindings,
        module_scope: &HashMap<Spur, Idx<Binding>>,
    ) -> Result<(), Diagnostic> {
        let dependencies = self.module_dependencies(bindings);
        let order = order_module_bindings(self.syntax, &bindings.statements, &dependencies)?;
        for &statement in &order {
            self.check_module_initializer(statement, bindings, module_scope)?;
        }
        self.module_bindings.extend(order);
        Ok(())
    }

    /// The module-level bindings each declaration references, in its annotation
    /// as well as its initializer, since an array length is an expression too.
    /// References to anything else are not part of the module's ordering.
    fn module_dependencies(
        &self,
        bindings: &ModuleBindings,
    ) -> HashMap<Idx<Statement>, Vec<ModuleDependency>> {
        let syntax = self.syntax;
        let mut dependencies = HashMap::new();
        for &statement in &bindings.statements {
            let StatementKind::Binding {
                annotation,
                initializer,
                ..
            } = &syntax.statements[statement].kind
            else {
                unreachable!("frontend only permits bindings at module level")
            };
            let mut references = Vec::new();
            if let Some(annotation) = annotation {
                collect_annotation_references(syntax, *annotation, &mut references);
            }
            collect_references(syntax, *initializer, &mut references);
            dependencies.insert(
                statement,
                references
                    .into_iter()
                    .filter_map(|(name, span)| {
                        bindings
                            .declarations
                            .get(&name)
                            .copied()
                            .map(|declaration| ModuleDependency {
                                declaration,
                                name,
                                span,
                            })
                    })
                    .collect::<Vec<_>>(),
            );
        }
        dependencies
    }

    fn check_module_initializer(
        &mut self,
        statement: Idx<Statement>,
        bindings: &ModuleBindings,
        module_scope: &HashMap<Spur, Idx<Binding>>,
    ) -> Result<(), Diagnostic> {
        let syntax = self.syntax;
        let StatementKind::Binding {
            mutable,
            annotation,
            initializer,
            ..
        } = &syntax.statements[statement].kind
        else {
            unreachable!("frontend only permits bindings at module level")
        };
        let (mutable, annotation, initializer) = (*mutable, *annotation, *initializer);
        self.file = bindings.files[&statement];
        // Signatures are resolved after this runs, so a call here has no
        // signature to check against. It is never constant, so report that.
        if let Some(call) = find_call(syntax, initializer) {
            return Err(Diagnostic::new(
                syntax.expressions[call].span.clone(),
                "module-level initializer must be a constant expression",
            ));
        }
        let scopes = std::slice::from_ref(module_scope);
        let destination = match annotation {
            Some(annotation) => {
                Some(self.resolve_annotation(annotation, scopes, Some(initializer))?)
            }
            None => None,
        };
        let expression = self.check_expression(initializer, scopes, destination)?;
        if expression.constant.is_none() {
            return Err(Diagnostic::new(
                syntax.expressions[initializer].span.clone(),
                "module-level initializer must be a constant expression",
            ));
        }
        let binding = self.declarations[statement];
        self.bindings[binding].ty = expression.ty.clone();
        self.bindings[binding].constant = if mutable {
            None
        } else {
            expression.constant.clone()
        };
        self.expressions.insert(initializer, expression);
        Ok(())
    }

    fn check_function_bodies(
        &mut self,
        items: &[(usize, TopLevelItem)],
        module_scope: &HashMap<Spur, Idx<Binding>>,
    ) -> Result<(), Diagnostic> {
        let syntax = self.syntax;
        for &(file, item) in items {
            let TopLevelItem::Function { function, .. } = item else {
                continue;
            };
            self.file = file;
            let parameter_bindings = self.functions[function].parameters.clone();
            let parameter_scope = syntax.functions[function]
                .parameters
                .iter()
                .zip(parameter_bindings)
                .map(|(parameter, binding)| (parameter.name, binding))
                .collect();
            let mut scopes = vec![module_scope.clone(), parameter_scope];
            let result = self.functions[function].result.clone();
            let body = &syntax.functions[function].body;
            self.check_body(body, result.as_ref(), &mut scopes, &mut Vec::new())?;
            if result.is_some() && !body_terminates(syntax, body) {
                let function = &syntax.functions[function];
                return Err(Diagnostic::new(
                    function.name_span.clone(),
                    format!(
                        "function `{}` can reach the end of its body without returning a value",
                        syntax.names.resolve(&function.name)
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Every name a file imports must be referenced by that file.
    fn check_imports_used(&self, module: &Module) -> Result<(), Diagnostic> {
        let syntax = self.syntax;
        for file in module.files.clone() {
            let file_imports = &self.imports[file];
            if let Some((name, span)) = file_imports
                .introduced
                .iter()
                .find(|(name, _)| !file_imports.used.contains(name))
            {
                return Err(Diagnostic::new(
                    span.clone(),
                    format!(
                        "imported name `{}` is never referenced",
                        syntax.names.resolve(name)
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// Reports whether every path through `body` ends in a `return` or `exit`, using
/// the structural rule under Function return in the specification.
fn body_terminates(syntax: &Syntax, body: &[Idx<Statement>]) -> bool {
    body.iter()
        .any(|&statement| statement_terminates(syntax, statement))
}

fn statement_terminates(syntax: &Syntax, statement: Idx<Statement>) -> bool {
    match &syntax.statements[statement].kind {
        StatementKind::Return { .. } | StatementKind::Exit { .. } => true,
        StatementKind::Block { body } => body_terminates(syntax, body),
        StatementKind::If {
            then_body,
            else_branch,
            ..
        } => {
            else_branch.is_some_and(|else_branch| statement_terminates(syntax, else_branch))
                && body_terminates(syntax, then_body)
        }
        StatementKind::For {
            label,
            header,
            body,
        } => {
            matches!(header, ForHeader::Infinite)
                && !body_breaks(syntax, body, label.as_ref().map(|label| label.name), false)
        }
        _ => false,
    }
}

/// Reports whether a reachable `break` in `body` targets the loop identified by
/// `label`. `nested` marks a body that lies inside a loop enclosed by that one,
/// where an unlabeled `break` targets the inner loop instead.
fn body_breaks(
    syntax: &Syntax,
    body: &[Idx<Statement>],
    label: Option<Spur>,
    nested: bool,
) -> bool {
    for &statement in body {
        if statement_breaks(syntax, statement, label, nested) {
            return true;
        }
        if statement_terminates(syntax, statement) {
            return false;
        }
    }
    false
}

fn statement_breaks(
    syntax: &Syntax,
    statement: Idx<Statement>,
    label: Option<Spur>,
    nested: bool,
) -> bool {
    match &syntax.statements[statement].kind {
        StatementKind::Break { label: target } => match target {
            Some(target) => label == Some(target.name),
            None => !nested,
        },
        StatementKind::Block { body } => body_breaks(syntax, body, label, nested),
        StatementKind::If {
            then_body,
            else_branch,
            ..
        } => {
            body_breaks(syntax, then_body, label, nested)
                || else_branch
                    .is_some_and(|else_branch| statement_breaks(syntax, else_branch, label, nested))
        }
        StatementKind::For { body, .. } => body_breaks(syntax, body, label, true),
        _ => false,
    }
}

#[derive(Clone)]
struct ModuleDependency {
    declaration: Idx<Statement>,
    name: Spur,
    span: std::ops::Range<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModuleVisit {
    Visiting,
    Complete,
}

fn order_module_bindings(
    syntax: &Syntax,
    declarations: &[Idx<Statement>],
    dependencies: &HashMap<Idx<Statement>, Vec<ModuleDependency>>,
) -> Result<Vec<Idx<Statement>>, Diagnostic> {
    let mut states = HashMap::new();
    let mut order = Vec::new();
    for &root in declarations {
        if states.get(&root) == Some(&ModuleVisit::Complete) {
            continue;
        }
        states.insert(root, ModuleVisit::Visiting);
        let mut stack = vec![(root, 0)];
        while let Some((declaration, next_dependency)) = stack.last_mut() {
            let declaration_dependencies = &dependencies[declaration];
            if *next_dependency == declaration_dependencies.len() {
                let declaration = *declaration;
                stack.pop();
                states.insert(declaration, ModuleVisit::Complete);
                order.push(declaration);
                continue;
            }

            let dependency = declaration_dependencies[*next_dependency].clone();
            *next_dependency += 1;
            match states.get(&dependency.declaration) {
                Some(ModuleVisit::Visiting) => {
                    return Err(Diagnostic::new(
                        dependency.span,
                        format!(
                            "module-level initializer cycle involving `{}`",
                            syntax.names.resolve(&dependency.name)
                        ),
                    ));
                }
                Some(ModuleVisit::Complete) => {}
                None => {
                    states.insert(dependency.declaration, ModuleVisit::Visiting);
                    stack.push((dependency.declaration, 0));
                }
            }
        }
    }
    Ok(order)
}

/// The module-level names an annotation's array lengths reference.
fn collect_annotation_references(
    syntax: &Syntax,
    annotation: Idx<TypeAnnotation>,
    references: &mut Vec<(Spur, std::ops::Range<usize>)>,
) {
    let AnnotationKind::Array { length, element } = syntax.annotations[annotation].kind else {
        return;
    };
    if let Some(length) = length {
        collect_references(syntax, length, references);
    }
    collect_annotation_references(syntax, element, references);
}

/// The unqualified module-level names an expression references. A call's
/// target names a function rather than a binding, so only its arguments count.
fn collect_references(
    syntax: &Syntax,
    expression: Idx<Expression>,
    references: &mut Vec<(Spur, std::ops::Range<usize>)>,
) {
    walk_expression(syntax, expression, &mut |id| {
        let expression = &syntax.expressions[id];
        if let ExpressionKind::Reference(name) = &expression.kind
            && name.qualifier.is_none()
        {
            references.push((name.name, expression.span.clone()));
        }
    });
}

/// The first call anywhere in `expression`, in source order.
fn find_call(syntax: &Syntax, expression: Idx<Expression>) -> Option<Idx<Expression>> {
    let mut call = None;
    walk_expression(syntax, expression, &mut |id| {
        if call.is_none() && matches!(syntax.expressions[id].kind, ExpressionKind::Call(_)) {
            call = Some(id);
        }
    });
    call
}

/// Visits `expression` and every sub-expression under it, in source order.
fn walk_expression(
    syntax: &Syntax,
    expression: Idx<Expression>,
    visit: &mut impl FnMut(Idx<Expression>),
) {
    visit(expression);
    match &syntax.expressions[expression].kind {
        ExpressionKind::Integer(_) | ExpressionKind::Boolean(_) | ExpressionKind::Reference(_) => {}
        ExpressionKind::Grouping { expression }
        | ExpressionKind::Unary {
            operand: expression,
            ..
        }
        | ExpressionKind::Conversion {
            operand: expression,
            ..
        }
        | ExpressionKind::LogicalNot {
            operand: expression,
            ..
        }
        | ExpressionKind::Length {
            operand: expression,
        } => walk_expression(syntax, *expression, visit),
        ExpressionKind::Binary { left, right, .. }
        | ExpressionKind::Comparison { left, right, .. }
        | ExpressionKind::Logical { left, right, .. }
        | ExpressionKind::Index {
            operand: left,
            index: right,
            ..
        } => {
            walk_expression(syntax, *left, visit);
            walk_expression(syntax, *right, visit);
        }
        ExpressionKind::Call(call) => {
            for &argument in &call.arguments {
                walk_expression(syntax, argument, visit);
            }
        }
        ExpressionKind::ArrayLiteral { elements, .. } => {
            for &element in elements {
                walk_expression(syntax, element, visit);
            }
        }
    }
}

impl CheckedProgram<'_> {
    fn check_body(
        &mut self,
        body: &[Idx<Statement>],
        result: Option<&Type>,
        scopes: &mut Vec<HashMap<Spur, Idx<Binding>>>,
        loops: &mut Vec<Option<Spur>>,
    ) -> Result<(), Diagnostic> {
        scopes.push(HashMap::new());
        for &statement in body {
            self.check_statement(statement, result, scopes, loops)?;
        }
        scopes.pop();
        Ok(())
    }

    fn check_statement(
        &mut self,
        statement: Idx<Statement>,
        result: Option<&Type>,
        scopes: &mut Vec<HashMap<Spur, Idx<Binding>>>,
        loops: &mut Vec<Option<Spur>>,
    ) -> Result<(), Diagnostic> {
        match &self.syntax.statements[statement].kind {
            StatementKind::Binding {
                name,
                mutable,
                annotation,
                initializer,
                ..
            } => self.check_binding(
                statement,
                *name,
                *mutable,
                *annotation,
                *initializer,
                scopes,
            ),
            StatementKind::Assignment { target, value } => {
                let target = self.assignment_target(target, scopes)?;
                let expression = self.check_expression(*value, scopes, Some(target.ty.clone()))?;
                self.expressions.insert(*value, expression);
                self.assignments.insert(statement, target);
                Ok(())
            }
            StatementKind::CompoundAssignment {
                target,
                operator,
                operator_span,
                value,
            } => self.check_compound_assignment(
                statement,
                target,
                *operator,
                operator_span,
                *value,
                scopes,
            ),
            StatementKind::Block { body } => self.check_body(body, result, scopes, loops),
            StatementKind::Exit { argument } => {
                let expression =
                    self.check_expression(*argument, scopes, Some(Scalar::Int.into()))?;
                self.expressions.insert(*argument, expression);
                Ok(())
            }
            StatementKind::If {
                condition,
                then_body,
                else_branch,
            } => {
                self.check_condition(*condition, scopes)?;
                self.check_body(then_body, result, scopes, loops)?;
                if let Some(else_branch) = else_branch {
                    self.check_statement(*else_branch, result, scopes, loops)?;
                }
                Ok(())
            }
            StatementKind::For { .. } => self.check_for(statement, result, scopes, loops),
            StatementKind::Break { label } => {
                self.check_loop_jump(statement, "break", label.as_ref(), loops)
            }
            StatementKind::Continue { label } => {
                self.check_loop_jump(statement, "continue", label.as_ref(), loops)
            }
            StatementKind::Call { call } => {
                let (function, _) = self.check_call(call, scopes, false)?;
                self.calls.insert(statement, function);
                Ok(())
            }
            StatementKind::Return { value } => self.check_return(statement, result, *value, scopes),
        }
    }

    fn check_binding(
        &mut self,
        statement: Idx<Statement>,
        name: Spur,
        mutable: bool,
        annotation: Option<Idx<TypeAnnotation>>,
        initializer: Idx<Expression>,
        scopes: &mut [HashMap<Spur, Idx<Binding>>],
    ) -> Result<(), Diagnostic> {
        let destination = match annotation {
            Some(annotation) => {
                Some(self.resolve_annotation(annotation, scopes, Some(initializer))?)
            }
            None => None,
        };
        let expression = self.check_expression(initializer, scopes, destination)?;
        let ty = expression.ty.clone();
        let constant = if mutable {
            None
        } else {
            expression.constant.clone()
        };
        self.expressions.insert(initializer, expression);
        let binding = self.bindings.alloc(Binding {
            ty,
            mutable,
            constant,
        });
        self.declarations.insert(statement, binding);
        scopes
            .last_mut()
            .expect("a statement is checked inside a scope")
            .insert(name, binding);
        Ok(())
    }

    /// Checks `target op= value` as the binary operation it stands for, with
    /// the target as the left operand.
    fn check_compound_assignment(
        &mut self,
        statement: Idx<Statement>,
        target: &AssignmentTarget,
        operator: BinaryOperator,
        operator_span: &std::ops::Range<usize>,
        value: Idx<Expression>,
        scopes: &mut [HashMap<Spur, Idx<Binding>>],
    ) -> Result<(), Diagnostic> {
        let target = self.assignment_target(target, scopes)?;
        let left = CheckedExpression {
            ty: target.ty.clone(),
            untyped: false,
            value: ExpressionValue::Reference(target.binding),
            constant: None,
        };
        let right = self.infer_expression(value, scopes)?;
        let (_, right, _, _, _) = self.check_integer_binary(
            operator,
            operator_span,
            &format!("{}=", operator.spelling()),
            CheckedBinaryOperand {
                id: None,
                expression: left,
            },
            CheckedBinaryOperand {
                id: Some(value),
                expression: right,
            },
        )?;
        self.expressions.insert(value, right.expression);
        self.assignments.insert(statement, target);
        Ok(())
    }

    fn check_for(
        &mut self,
        statement: Idx<Statement>,
        result: Option<&Type>,
        scopes: &mut Vec<HashMap<Spur, Idx<Binding>>>,
        loops: &mut Vec<Option<Spur>>,
    ) -> Result<(), Diagnostic> {
        let StatementKind::For {
            label,
            header,
            body,
        } = &self.syntax.statements[statement].kind
        else {
            unreachable!("a `for` statement is checked from its own syntax")
        };
        if let Some(label) = label
            && loops.iter().flatten().any(|name| *name == label.name)
        {
            return Err(Diagnostic::new(
                label.name_span.clone(),
                format!(
                    "duplicate enclosing loop label `{}`",
                    self.syntax.names.resolve(&label.name)
                ),
            ));
        }
        let has_header_scope = self.check_for_header(statement, header, result, scopes, loops)?;
        loops.push(label.as_ref().map(|label| label.name));
        self.check_body(body, result, scopes, loops)?;
        loops.pop();
        if has_header_scope {
            scopes.pop();
        }
        Ok(())
    }

    /// Checks a `for` header, reporting whether it opened a scope for the
    /// bindings its initializer declares.
    fn check_for_header(
        &mut self,
        statement: Idx<Statement>,
        header: &ForHeader,
        result: Option<&Type>,
        scopes: &mut Vec<HashMap<Spur, Idx<Binding>>>,
        loops: &mut Vec<Option<Spur>>,
    ) -> Result<bool, Diagnostic> {
        match header {
            ForHeader::Infinite => Ok(false),
            ForHeader::Condition(condition) => {
                self.check_condition(*condition, scopes)?;
                Ok(false)
            }
            ForHeader::ThreeClause {
                initializer,
                condition,
                post,
            } => {
                scopes.push(HashMap::new());
                self.check_statement(*initializer, result, scopes, loops)?;
                self.check_condition(*condition, scopes)?;
                self.check_statement(*post, result, scopes, loops)?;
                Ok(true)
            }
            ForHeader::Iteration {
                value,
                index,
                operand,
                ..
            } => {
                self.check_iteration_header(statement, *value, index.as_ref(), *operand, scopes)?;
                Ok(true)
            }
        }
    }

    /// Checks `for v in a` and `for v, i in a`, opening the scope that holds
    /// the loop's bindings. They are immutable and live only inside the loop.
    fn check_iteration_header(
        &mut self,
        statement: Idx<Statement>,
        value: Spur,
        index: Option<&(Spur, std::ops::Range<usize>)>,
        operand: Idx<Expression>,
        scopes: &mut Vec<HashMap<Spur, Idx<Binding>>>,
    ) -> Result<(), Diagnostic> {
        let checked = self.infer_expression(operand, scopes)?;
        let Type::Array { element, .. } = &checked.ty else {
            return Err(Diagnostic::new(
                self.syntax.expressions[operand].span.clone(),
                format!("`for … in` requires an array, found `{}`", checked.ty),
            ));
        };
        let element = (**element).clone();
        if let Some((name, span)) = index
            && *name == value
        {
            return Err(Diagnostic::new(
                span.clone(),
                "a `for` loop's value and index bindings must have different names",
            ));
        }
        self.expressions.insert(operand, checked);
        let mut loop_scope = HashMap::new();
        let mut bind = |program: &mut Self, name: Spur, ty: Type| {
            let binding = program.bindings.alloc(Binding {
                ty,
                mutable: false,
                constant: None,
            });
            loop_scope.insert(name, binding);
            binding
        };
        let value = bind(self, value, element);
        let index = index.map(|(name, _)| bind(self, *name, Scalar::Int.into()));
        self.iterations
            .insert(statement, IterationBindings { value, index });
        scopes.push(loop_scope);
        Ok(())
    }

    /// Checks that a `break` or `continue` names a loop it is inside.
    fn check_loop_jump(
        &self,
        statement: Idx<Statement>,
        keyword: &str,
        label: Option<&Label>,
        loops: &[Option<Spur>],
    ) -> Result<(), Diagnostic> {
        if loops.is_empty() {
            let start = self.syntax.statements[statement].span.start;
            return Err(Diagnostic::new(
                start..start + keyword.len(),
                format!("`{keyword}` is not inside a loop"),
            ));
        }
        if let Some(label) = label
            && !loops.iter().flatten().any(|name| *name == label.name)
        {
            return Err(Diagnostic::new(
                label.name_span.clone(),
                format!(
                    "unknown enclosing loop label `{}`",
                    self.syntax.names.resolve(&label.name)
                ),
            ));
        }
        Ok(())
    }

    fn check_return(
        &mut self,
        statement: Idx<Statement>,
        result: Option<&Type>,
        value: Option<Idx<Expression>>,
        scopes: &mut [HashMap<Spur, Idx<Binding>>],
    ) -> Result<(), Diagnostic> {
        match (result, value) {
            (None, None) => Ok(()),
            (None, Some(value)) => Err(Diagnostic::new(
                self.syntax.expressions[value].span.clone(),
                "cannot return a value from a `void` function",
            )),
            (Some(result), None) => Err(Diagnostic::new(
                self.syntax.statements[statement].span.clone(),
                format!("`return` must supply a value of type `{result}`"),
            )),
            (Some(result), Some(value)) => {
                let expression = self.check_expression(value, scopes, Some(result.clone()))?;
                self.expressions.insert(value, expression);
                Ok(())
            }
        }
    }

    /// The binding an assignment stores into, and the type of the element its
    /// indices reach. An element of a `const` binding is rejected with the
    /// binding itself, so the mutability check comes first.
    fn assignment_target(
        &mut self,
        target: &AssignmentTarget,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedTarget, Diagnostic> {
        let name = &target.name;
        let binding = self.resolve(name, scopes)?;
        if !self.bindings[binding].mutable {
            return Err(Diagnostic::new(
                name.span.clone(),
                format!(
                    "cannot assign to immutable binding `{}`",
                    self.syntax.names.resolve(&name.name)
                ),
            ));
        }
        // The indices are checked left to right, the order they are evaluated
        // in, and each one descends into the element type.
        let mut ty = self.bindings[binding].ty.clone();
        for &index in &target.indices {
            ty = self.check_index_step(&ty, &name.span, index, scopes)?;
        }
        Ok(CheckedTarget { binding, ty })
    }

    fn check_condition(
        &mut self,
        condition: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<(), Diagnostic> {
        let expression = self.check_expression(condition, scopes, Some(Scalar::Bool.into()))?;
        self.expressions.insert(condition, expression);
        Ok(())
    }

    fn check_call(
        &mut self,
        call: &Call,
        scopes: &[HashMap<Spur, Idx<Binding>>],
        value_context: bool,
    ) -> Result<(Idx<Function>, Option<Type>), Diagnostic> {
        let function = self.resolve_call(call, scopes)?;
        let signature = self.functions[function].clone();
        if call.arguments.len() != signature.parameters.len() {
            return Err(Diagnostic::new(
                call.target.span.clone(),
                format!(
                    "function `{}` expects {} argument{}, found {}",
                    self.syntax.names.resolve(&call.target.name),
                    signature.parameters.len(),
                    if signature.parameters.len() == 1 {
                        ""
                    } else {
                        "s"
                    },
                    call.arguments.len(),
                ),
            ));
        }
        for (&argument, &parameter) in call.arguments.iter().zip(&signature.parameters) {
            let expression =
                self.check_expression(argument, scopes, Some(self.bindings[parameter].ty.clone()))?;
            self.expressions.insert(argument, expression);
        }
        if value_context && signature.result.is_none() {
            return Err(Diagnostic::new(
                call.target.span.clone(),
                format!(
                    "void function `{}` cannot be used as a value",
                    self.syntax.names.resolve(&call.target.name)
                ),
            ));
        }
        Ok((function, signature.result))
    }

    fn resolve_call(
        &mut self,
        call: &Call,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<Idx<Function>, Diagnostic> {
        let target = &call.target;
        let declaration = if target.qualifier.is_some() {
            self.qualified(target, scopes)?
        } else {
            if scopes
                .iter()
                .rev()
                .any(|scope| scope.contains_key(&target.name))
            {
                return Err(Diagnostic::new(
                    target.span.clone(),
                    format!(
                        "cannot call non-function binding `{}`",
                        self.syntax.names.resolve(&target.name)
                    ),
                ));
            }
            if let Some(&function) = self.function_names.get(&target.name) {
                return Ok(function);
            }
            self.imported(target, Wanted::Function)?
        };
        match declaration {
            DeclarationKind::Function(function) => Ok(function),
            DeclarationKind::Binding(_) => Err(Diagnostic::new(
                target.span.clone(),
                format!(
                    "cannot call non-function declaration `{}`",
                    self.syntax.names.resolve(&target.name)
                ),
            )),
        }
    }

    fn resolve(
        &mut self,
        name: &QualifiedName,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<Idx<Binding>, Diagnostic> {
        let declaration = if name.qualifier.is_some() {
            self.qualified(name, scopes)?
        } else {
            if let Some(binding) = scopes.iter().rev().find_map(|scope| scope.get(&name.name)) {
                return Ok(*binding);
            }
            self.imported(name, Wanted::Value)?
        };
        match declaration {
            DeclarationKind::Binding(binding) => Ok(binding),
            DeclarationKind::Function(_) => Err(Diagnostic::new(
                name.span.clone(),
                format!(
                    "cannot use function `{}` as a value",
                    self.syntax.names.resolve(&name.name)
                ),
            )),
        }
    }

    /// The declaration an unqualified name reaches through its file's imports,
    /// marking that import referenced. `wanted` only chooses the wording of the
    /// diagnostics.
    fn imported(
        &mut self,
        name: &QualifiedName,
        wanted: Wanted,
    ) -> Result<DeclarationKind, Diagnostic> {
        let syntax = self.syntax;
        let spelling = syntax.names.resolve(&name.name);
        match self.imports[self.file].names.get(&name.name) {
            Some(&Imported::Declaration(kind)) => {
                self.imports[self.file].used.insert(name.name);
                Ok(kind)
            }
            Some(Imported::Module(_)) => Err(Diagnostic::new(
                name.span.clone(),
                format!("module `{spelling}` is not {}", wanted.article_noun()),
            )),
            None => Err(Diagnostic::new(
                name.span.clone(),
                format!("{} `{spelling}`", wanted.unknown()),
            )),
        }
    }

    /// Resolves a `module::name` against the current file's imports. A lexical
    /// binding shadows an imported module name, so a qualifier it holds cannot
    /// reach the module.
    fn qualified(
        &mut self,
        name: &QualifiedName,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<DeclarationKind, Diagnostic> {
        let qualifier = name
            .qualifier
            .as_ref()
            .expect("the caller checked the qualifier");
        if scopes
            .iter()
            .rev()
            .any(|scope| scope.contains_key(&qualifier.name))
        {
            return Err(Diagnostic::new(
                qualifier.name_span.clone(),
                format!(
                    "cannot use binding `{}` as a module",
                    self.syntax.names.resolve(&qualifier.name)
                ),
            ));
        }
        let module = match self.imports[self.file].names.get(&qualifier.name) {
            Some(&Imported::Module(module)) => module,
            Some(Imported::Declaration(_)) => {
                return Err(Diagnostic::new(
                    qualifier.name_span.clone(),
                    format!(
                        "`{}` is not a module",
                        self.syntax.names.resolve(&qualifier.name)
                    ),
                ));
            }
            None => {
                return Err(Diagnostic::new(
                    qualifier.name_span.clone(),
                    format!(
                        "unknown module `{}`",
                        self.syntax.names.resolve(&qualifier.name)
                    ),
                ));
            }
        };
        self.imports[self.file].used.insert(qualifier.name);
        let Some(declaration) = self.namespaces[module].get(&name.name).copied() else {
            return Err(Diagnostic::new(
                name.span.clone(),
                format!(
                    "module `{}` has no declaration named `{}`",
                    self.syntax.names.resolve(&qualifier.name),
                    self.syntax.names.resolve(&name.name)
                ),
            ));
        };
        if !declaration.public {
            return Err(Diagnostic::new(
                name.span.clone(),
                format!(
                    "declaration `{}` is private to module `{}`",
                    self.syntax.names.resolve(&name.name),
                    self.syntax.names.resolve(&qualifier.name)
                ),
            ));
        }
        Ok(declaration.kind)
    }

    fn check_expression(
        &mut self,
        id: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
        destination: Option<Type>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let syntax = self.syntax;
        // An array literal is the one expression whose type comes from its
        // destination rather than from concretizing an inferred type.
        if matches!(
            syntax.expressions[id].kind,
            ExpressionKind::ArrayLiteral { .. }
        ) {
            return self.check_array_literal(id, scopes, destination);
        }
        let mut checked = self.infer_expression(id, scopes)?;
        let mismatch = |checked: &CheckedExpression, destination: &Type| {
            Diagnostic::new(
                syntax.expressions[id].span.clone(),
                format!(
                    "cannot implicitly convert `{}` to `{destination}`",
                    checked.ty
                ),
            )
        };
        if checked.untyped {
            let destination = destination.unwrap_or_else(|| checked.ty.clone());
            let Some(scalar) = destination.scalar() else {
                return Err(mismatch(&checked, &destination));
            };
            self.concretize(id, &mut checked, scalar)?;
        } else if let Some(destination) = destination
            && checked.ty != destination
        {
            return Err(mismatch(&checked, &destination));
        }
        Ok(checked)
    }

    fn infer_expression(
        &mut self,
        id: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let expression = &self.syntax.expressions[id];
        match &expression.kind {
            ExpressionKind::Integer(spelling) => Ok(integer_literal(spelling)),
            ExpressionKind::Boolean(value) => Ok(CheckedExpression {
                ty: Scalar::Bool.into(),
                untyped: true,
                value: ExpressionValue::Boolean,
                constant: Some(BigInt::from(*value).into()),
            }),
            ExpressionKind::Reference(name) => {
                let binding = self.resolve(name, scopes)?;
                Ok(CheckedExpression {
                    ty: self.bindings[binding].ty.clone(),
                    untyped: false,
                    value: ExpressionValue::Reference(binding),
                    constant: self.bindings[binding].constant.clone(),
                })
            }
            ExpressionKind::Grouping { expression } => self.infer_grouping(*expression, scopes),
            ExpressionKind::Unary {
                operator,
                operator_span,
                operand,
            } => self.infer_unary(*operator, operator_span, *operand, scopes),
            ExpressionKind::Binary {
                operator,
                operator_span,
                left,
                right,
            } => self.infer_binary(*operator, operator_span, *left, *right, scopes),
            ExpressionKind::Comparison {
                operator,
                operator_span,
                left,
                right,
            } => self.infer_comparison(*operator, operator_span, *left, *right, scopes),
            ExpressionKind::Logical {
                operator,
                operator_span,
                left,
                right,
            } => self.infer_logical(*operator, operator_span, *left, *right, scopes),
            ExpressionKind::LogicalNot {
                operator_span,
                operand,
            } => self.infer_logical_not(operator_span, *operand, scopes),
            ExpressionKind::Conversion {
                destination,
                truncating,
                operand,
                ..
            } => self.infer_conversion(
                *destination,
                *truncating,
                *operand,
                &expression.span,
                scopes,
            ),
            ExpressionKind::ArrayLiteral { .. } => self.check_array_literal(id, scopes, None),
            ExpressionKind::Index { operand, index, .. } => {
                self.infer_index(*operand, *index, scopes)
            }
            ExpressionKind::Length { operand } => self.infer_length(*operand, scopes),
            ExpressionKind::Call(call) => {
                let (function, result) = self.check_call(call, scopes, true)?;
                Ok(CheckedExpression {
                    ty: result.expect("value-context call has a value result"),
                    untyped: false,
                    value: ExpressionValue::Call { function },
                    constant: None,
                })
            }
        }
    }

    /// Checks an array literal. Every element is checked against the array's
    /// element type, which the destination supplies when there is one and the
    /// elements themselves supply when there is not.
    fn check_array_literal(
        &mut self,
        id: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
        destination: Option<Type>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let span = self.syntax.expressions[id].span.clone();
        let ExpressionKind::ArrayLiteral { elements, fill } = &self.syntax.expressions[id].kind
        else {
            unreachable!("an array literal is checked from its own syntax")
        };
        let (elements, fill) = (elements.clone(), fill.clone());
        let (length, element_type) =
            self.array_literal_type(&span, scopes, destination, &elements, fill.as_ref())?;
        let ty = Type::Array {
            length,
            element: Box::new(element_type.clone()),
        };
        check_element_count(span, &ty, length, elements.len(), fill.is_some())?;
        let mut checked_elements = Vec::with_capacity(elements.len());
        for &element in &elements {
            checked_elements.push(self.check_expression(
                element,
                scopes,
                Some(element_type.clone()),
            )?);
        }
        let constant = fold_elements(&checked_elements, length);
        let folded = constant.is_some();
        for (&element, checked) in elements.iter().zip(checked_elements) {
            self.record_operand(element, checked, folded);
        }
        Ok(CheckedExpression {
            ty,
            untyped: false,
            value: ExpressionValue::Array {
                elements,
                fill: fill.is_some(),
            },
            constant,
        })
    }

    /// The length and element type an array literal is checked against.
    fn array_literal_type(
        &mut self,
        span: &std::ops::Range<usize>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
        destination: Option<Type>,
        elements: &[Idx<Expression>],
        fill: Option<&std::ops::Range<usize>>,
    ) -> Result<(u64, Type), Diagnostic> {
        match destination {
            Some(Type::Array { length, element }) => Ok((length, *element)),
            Some(destination) => Err(Diagnostic::new(
                span.clone(),
                format!("cannot implicitly convert an array literal to `{destination}`"),
            )),
            None => {
                if let Some(fill) = fill {
                    return Err(Diagnostic::new(
                        fill.clone(),
                        "a fill requires a length from context",
                    ));
                }
                let length = u64::try_from(elements.len())
                    .expect("a source file holds fewer elements than u64::MAX");
                Ok((length, self.common_element_type(elements, scopes)?))
            }
        }
    }

    /// The one type the elements of a literal with no context must share: the
    /// type of the first typed element, or the untyped default when every
    /// element is an untyped constant. The checked elements are discarded, so
    /// every element goes through the one checking path afterwards.
    fn common_element_type(
        &mut self,
        elements: &[Idx<Expression>],
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<Type, Diagnostic> {
        let mut default = None;
        for &element in elements {
            let checked = self.infer_expression(element, scopes)?;
            if !checked.untyped {
                return Ok(checked.ty);
            }
            default.get_or_insert(checked.ty);
        }
        Ok(default.expect("an array literal has at least one element"))
    }

    /// Checks one `[ … ]` step and gives the element type it reaches. The
    /// operand must be an array, and the index must be an `int` that is in
    /// range whenever it is constant. Index expressions and assignment targets
    /// share this, so both spell one rule.
    fn check_index_step(
        &mut self,
        operand: &Type,
        operand_span: &std::ops::Range<usize>,
        index: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<Type, Diagnostic> {
        let Type::Array { length, element } = operand else {
            return Err(Diagnostic::new(
                operand_span.clone(),
                format!("cannot index `{operand}`"),
            ));
        };
        let checked = self.check_expression(index, scopes, Some(Scalar::Int.into()))?;
        if let Some(value) = checked.integer()
            && value.to_u64().is_none_or(|value| value >= *length)
        {
            return Err(Diagnostic::new(
                self.syntax.expressions[index].span.clone(),
                format!("index {value} is out of range for `{operand}`"),
            ));
        }
        self.expressions.insert(index, checked);
        Ok((**element).clone())
    }

    fn infer_index(
        &mut self,
        operand: Idx<Expression>,
        index: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        let span = self.syntax.expressions[operand].span.clone();
        let element = self.check_index_step(&checked_operand.ty, &span, index, scopes)?;
        // `a[i]` is never a constant expression, so both sub-expressions keep
        // their own constants and are still evaluated.
        self.record_operand(operand, checked_operand, false);
        Ok(CheckedExpression {
            ty: element,
            untyped: false,
            value: ExpressionValue::Index { operand, index },
            constant: None,
        })
    }

    fn infer_length(
        &mut self,
        operand: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        let Type::Array { length, .. } = &checked_operand.ty else {
            return Err(Diagnostic::new(
                self.syntax.expressions[operand].span.clone(),
                format!(
                    "`len` requires an array operand, found `{}`",
                    checked_operand.ty
                ),
            ));
        };
        // The length comes from the operand's type, so it folds unless
        // reaching the type needs a call. The operand is evaluated either way.
        let constant = find_call(self.syntax, operand)
            .is_none()
            .then(|| Constant::Integer(BigInt::from(*length)));
        self.record_operand(operand, checked_operand, false);
        Ok(CheckedExpression {
            ty: Scalar::Int.into(),
            untyped: false,
            value: ExpressionValue::Length { operand },
            constant,
        })
    }

    /// Records a checked operand. A folded parent owns the constant value, so
    /// the operand it was folded from no longer carries one.
    fn record_operand(
        &mut self,
        id: Idx<Expression>,
        mut operand: CheckedExpression,
        folded: bool,
    ) {
        if folded {
            operand.constant = None;
        }
        self.expressions.insert(id, operand);
    }

    fn infer_grouping(
        &mut self,
        inner: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked = self.infer_expression(inner, scopes)?;
        let result = CheckedExpression {
            ty: checked.ty.clone(),
            untyped: checked.untyped,
            value: ExpressionValue::Grouping { expression: inner },
            constant: checked.constant.clone(),
        };
        self.record_operand(inner, checked, result.constant.is_some());
        Ok(result)
    }

    fn infer_unary(
        &mut self,
        operator: UnaryOperator,
        operator_span: &std::ops::Range<usize>,
        operand: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        let error = |message| Diagnostic::new(operator_span.clone(), message);
        let Some(operand_type) = integer_operand(&checked_operand) else {
            return Err(error(
                "integer unary operator requires an integer operand".to_string(),
            ));
        };
        if operator == UnaryOperator::Negate && !checked_operand.untyped && !operand_type.signed() {
            return Err(error(format!(
                "unary `-` is not permitted on `{operand_type}`"
            )));
        }
        if operator == UnaryOperator::WrappingNegate && checked_operand.untyped {
            return Err(error(
                "wrapping negation requires a typed operand".to_string(),
            ));
        }
        let constant = evaluate_unary(
            operator,
            operator_span.clone(),
            operand_type,
            &checked_operand,
        )?;
        let result = CheckedExpression {
            ty: checked_operand.ty.clone(),
            untyped: checked_operand.untyped,
            value: ExpressionValue::Unary {
                operator,
                operator_span: operator_span.clone(),
                operand,
            },
            constant: constant.map(Constant::Integer),
        };
        self.record_operand(operand, checked_operand, result.constant.is_some());
        Ok(result)
    }

    fn infer_binary(
        &mut self,
        operator: BinaryOperator,
        operator_span: &std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_left = self.infer_expression(left, scopes)?;
        let checked_right = self.infer_expression(right, scopes)?;
        let (checked_left, checked_right, ty, untyped, constant) = self.check_integer_binary(
            operator,
            operator_span,
            operator.spelling(),
            CheckedBinaryOperand {
                id: Some(left),
                expression: checked_left,
            },
            CheckedBinaryOperand {
                id: Some(right),
                expression: checked_right,
            },
        )?;
        let result = CheckedExpression {
            ty: ty.into(),
            untyped,
            value: ExpressionValue::Binary {
                operator,
                operator_span: operator_span.clone(),
                left,
                right,
            },
            constant: constant.map(Constant::Integer),
        };
        let folded = result.constant.is_some();
        self.record_operand(left, checked_left.expression, folded);
        self.record_operand(right, checked_right.expression, folded);
        Ok(result)
    }

    fn infer_comparison(
        &mut self,
        operator: ComparisonOperator,
        operator_span: &std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let mut checked_left = self.infer_expression(left, scopes)?;
        let mut checked_right = self.infer_expression(right, scopes)?;
        self.unify_comparison(
            operator,
            operator_span,
            (left, &mut checked_left),
            (right, &mut checked_right),
        )?;
        let constant = match (
            checked_left.constant.as_ref(),
            checked_right.constant.as_ref(),
        ) {
            (Some(left), Some(right)) => Some(compare_constants(operator, left, right)),
            _ => None,
        };
        let result = CheckedExpression {
            ty: Scalar::Bool.into(),
            untyped: constant.is_some(),
            value: ExpressionValue::Comparison {
                operator,
                operator_span: operator_span.clone(),
                left,
                right,
            },
            constant: constant.map(Constant::Integer),
        };
        let folded = result.constant.is_some();
        self.record_operand(left, checked_left, folded);
        self.record_operand(right, checked_right, folded);
        Ok(result)
    }

    /// Gives both comparison operands one type. An untyped operand takes the
    /// type of a typed one; otherwise the two types must already agree. Arrays
    /// are never untyped, so they only have to agree, and only `==` and `!=`
    /// reach them.
    fn unify_comparison(
        &mut self,
        operator: ComparisonOperator,
        operator_span: &std::ops::Range<usize>,
        left: (Idx<Expression>, &mut CheckedExpression),
        right: (Idx<Expression>, &mut CheckedExpression),
    ) -> Result<(), Diagnostic> {
        let (left_id, left) = left;
        let (right_id, right) = right;
        let unified = match (left.ty.scalar(), right.ty.scalar()) {
            (Some(left_type), Some(right_type)) => match (left.untyped, right.untyped) {
                (false, true) => return self.concretize(right_id, right, left_type),
                (true, false) => return self.concretize(left_id, left, right_type),
                _ => left_type == right_type,
            },
            _ => left.ty == right.ty,
        };
        if !unified {
            return Err(Diagnostic::new(
                operator_span.clone(),
                format!(
                    "comparison operands have different types `{}` and `{}`",
                    left.ty, right.ty
                ),
            ));
        }
        if left.ty.scalar().is_none()
            && !matches!(
                operator,
                ComparisonOperator::Equal | ComparisonOperator::NotEqual
            )
        {
            return Err(Diagnostic::new(
                operator_span.clone(),
                format!("only `==` and `!=` are defined on `{}`", left.ty),
            ));
        }
        Ok(())
    }

    fn infer_logical(
        &mut self,
        operator: LogicalOperator,
        operator_span: &std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let mut checked_left = self.infer_expression(left, scopes)?;
        let mut checked_right = self.infer_expression(right, scopes)?;
        require_boolean(self.syntax, left, &checked_left)?;
        require_boolean(self.syntax, right, &checked_right)?;
        match (checked_left.untyped, checked_right.untyped) {
            (true, false) => self.concretize(left, &mut checked_left, Scalar::Bool)?,
            (false, true) => self.concretize(right, &mut checked_right, Scalar::Bool)?,
            _ => {}
        }
        let constant = match (checked_left.integer(), checked_right.integer()) {
            (Some(left), Some(right)) => Some(BigInt::from(match operator {
                LogicalOperator::And => constant_boolean(left) && constant_boolean(right),
                LogicalOperator::Or => constant_boolean(left) || constant_boolean(right),
            })),
            _ => None,
        };
        let result = CheckedExpression {
            ty: Scalar::Bool.into(),
            untyped: checked_left.untyped && checked_right.untyped,
            value: ExpressionValue::Logical {
                operator,
                operator_span: operator_span.clone(),
                left,
                right,
            },
            constant: constant.map(Constant::Integer),
        };
        let folded = result.constant.is_some();
        self.record_operand(left, checked_left, folded);
        self.record_operand(right, checked_right, folded);
        Ok(result)
    }

    fn infer_logical_not(
        &mut self,
        operator_span: &std::ops::Range<usize>,
        operand: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        require_boolean(self.syntax, operand, &checked_operand)?;
        let constant = checked_operand
            .integer()
            .map(|value| BigInt::from(!constant_boolean(value)));
        let result = CheckedExpression {
            ty: Scalar::Bool.into(),
            untyped: checked_operand.untyped,
            value: ExpressionValue::LogicalNot {
                operator_span: operator_span.clone(),
                operand,
            },
            constant: constant.map(Constant::Integer),
        };
        self.record_operand(operand, checked_operand, result.constant.is_some());
        Ok(result)
    }

    fn infer_conversion(
        &mut self,
        destination: Scalar,
        truncating: bool,
        operand: Idx<Expression>,
        span: &std::ops::Range<usize>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let mut checked_operand = self.infer_expression(operand, scopes)?;
        if integer_operand(&checked_operand).is_none() {
            return Err(Diagnostic::new(
                span.clone(),
                format!("cannot convert `{}` to `{destination}`", checked_operand.ty),
            ));
        }
        if checked_operand.untyped && (!truncating || checked_operand.constant.is_none()) {
            let operand_type = if truncating { Scalar::Int } else { destination };
            self.concretize(operand, &mut checked_operand, operand_type)?;
        }
        let constant = self.convert_constant(&checked_operand, destination, truncating, span)?;
        let value = if constant.is_some() {
            ExpressionValue::Integer
        } else {
            ExpressionValue::Conversion {
                operand,
                truncating,
            }
        };
        self.record_operand(operand, checked_operand, constant.is_some());
        Ok(CheckedExpression {
            ty: destination.into(),
            untyped: false,
            value,
            constant: constant.map(Constant::Integer),
        })
    }

    /// Converts a constant operand at check time. A non-truncating conversion
    /// that would trap is an error rather than a trap at run time.
    fn convert_constant(
        &self,
        operand: &CheckedExpression,
        destination: Scalar,
        truncating: bool,
        span: &std::ops::Range<usize>,
    ) -> Result<Option<BigInt>, Diagnostic> {
        let Some(value) = operand.integer() else {
            return Ok(None);
        };
        if truncating {
            return Ok(Some(truncate_integer(value, destination)));
        }
        if !integer_fits(value, destination) {
            return Err(Diagnostic::new(
                span.clone(),
                format!("constant conversion to `{destination}` would trap"),
            ));
        }
        Ok(Some(value.clone()))
    }

    fn check_integer_binary(
        &mut self,
        operator: BinaryOperator,
        operator_span: &std::ops::Range<usize>,
        spelling: &str,
        mut left: CheckedBinaryOperand,
        mut right: CheckedBinaryOperand,
    ) -> Result<
        (
            CheckedBinaryOperand,
            CheckedBinaryOperand,
            Scalar,
            bool,
            Option<BigInt>,
        ),
        Diagnostic,
    > {
        let (Some(mut left_type), Some(right_type)) = (
            integer_operand(&left.expression),
            integer_operand(&right.expression),
        ) else {
            return Err(Diagnostic::new(
                operator_span.clone(),
                format!("integer `{spelling}` requires integer operands"),
            ));
        };
        let shift = matches!(
            operator,
            BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight
        );
        let wrapping = matches!(
            operator,
            BinaryOperator::WrappingAdd
                | BinaryOperator::WrappingSubtract
                | BinaryOperator::WrappingMultiply
        );
        if wrapping && left.expression.untyped && right.expression.untyped {
            return Err(Diagnostic::new(
                operator_span.clone(),
                "wrapping arithmetic requires a typed operand",
            ));
        }
        if !shift {
            match (left.expression.untyped, right.expression.untyped) {
                (false, false) if left_type != right_type => {
                    return Err(Diagnostic::new(
                        operator_span.clone(),
                        format!(
                            "binary operands have different types `{left_type}` and `{right_type}`"
                        ),
                    ));
                }
                (false, true) => self.concretize(
                    right
                        .id
                        .expect("an untyped right operand has an expression"),
                    &mut right.expression,
                    left_type,
                )?,
                (true, false) => {
                    self.concretize(
                        left.id.expect("an untyped left operand has an expression"),
                        &mut left.expression,
                        right_type,
                    )?;
                    left_type = right_type;
                }
                _ => {}
            }
        } else if (left.expression.constant.is_none() || right.expression.constant.is_none())
            && right.expression.untyped
        {
            self.concretize(
                right
                    .id
                    .expect("an untyped right operand has an expression"),
                &mut right.expression,
                Scalar::Int,
            )?;
        }
        let (ty, untyped) = if shift {
            (left_type, left.expression.untyped)
        } else if left.expression.untyped {
            (right_type, right.expression.untyped)
        } else {
            (left_type, false)
        };
        let constant = evaluate_binary(
            operator,
            operator_span.clone(),
            spelling,
            ty,
            untyped,
            (&left.expression, &right.expression),
        )?;
        Ok((left, right, ty, untyped, constant))
    }

    fn concretize(
        &mut self,
        id: Idx<Expression>,
        checked: &mut CheckedExpression,
        destination: Scalar,
    ) -> Result<(), Diagnostic> {
        debug_assert!(checked.untyped);
        let ty = checked
            .ty
            .scalar()
            .expect("an untyped expression has a scalar type");
        if ty.is_integer() != destination.is_integer() {
            return Err(Diagnostic::new(
                self.syntax.expressions[id].span.clone(),
                format!("cannot implicitly convert `{ty}` to `{destination}`"),
            ));
        }
        let constant = concretize_value(self.syntax, id, checked, destination)?
            .expect("caller provides an untyped expression");
        self.concretize_children(id, destination, constant)?;
        Ok(())
    }

    fn concretize_stored(
        &mut self,
        id: Idx<Expression>,
        destination: Scalar,
    ) -> Result<(), Diagnostic> {
        let Some(constant) =
            concretize_value(self.syntax, id, &mut self.expressions[id], destination)?
        else {
            return Ok(());
        };
        self.concretize_children(id, destination, constant)?;
        Ok(())
    }

    fn concretize_children(
        &mut self,
        id: Idx<Expression>,
        destination: Scalar,
        constant: bool,
    ) -> Result<(), Diagnostic> {
        let (first, second) = match &self.syntax.expressions[id].kind {
            ExpressionKind::Grouping { expression } => (Some(*expression), None),
            ExpressionKind::Unary {
                operator,
                operator_span,
                operand,
            } if !constant => {
                if *operator == UnaryOperator::Negate && !destination.signed() {
                    return Err(Diagnostic::new(
                        operator_span.clone(),
                        format!("unary `-` is not permitted on `{destination}`"),
                    ));
                }
                (Some(*operand), None)
            }
            ExpressionKind::Binary {
                operator: BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight,
                left,
                ..
            } if !constant => (Some(*left), None),
            ExpressionKind::Binary { left, right, .. } if !constant => (Some(*left), Some(*right)),
            _ => (None, None),
        };
        for child in first.into_iter().chain(second) {
            self.concretize_stored(child, destination)?;
        }
        Ok(())
    }
}

/// Checks a literal's element count against the array's length. A fill covers
/// the remaining elements, so it only bounds the count from above.
fn check_element_count(
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

/// The value an array literal folds to, which is one constant per element with
/// the fill repeating the last element. It folds only when every element does.
fn fold_elements(elements: &[CheckedExpression], length: u64) -> Option<Constant> {
    let mut values = elements
        .iter()
        .map(|element| element.constant.clone())
        .collect::<Option<Vec<_>>>()?;
    let last = values
        .last()
        .expect("an array literal has at least one element")
        .clone();
    values.resize(
        usize::try_from(length).expect("an array fits in the host's address space"),
        last,
    );
    Some(Constant::Array(values))
}

/// The scalar type an integer operator requires of its operand, or `None` when
/// the operand is a `bool` or an array.
fn integer_operand(checked: &CheckedExpression) -> Option<Scalar> {
    checked.ty.scalar().filter(|scalar| scalar.is_integer())
}

fn evaluate_unary(
    operator: UnaryOperator,
    operator_span: std::ops::Range<usize>,
    ty: Scalar,
    operand: &CheckedExpression,
) -> Result<Option<BigInt>, Diagnostic> {
    let Some(value) = operand.integer() else {
        return Ok(None);
    };
    let result = match operator {
        UnaryOperator::Negate => -value,
        UnaryOperator::WrappingNegate => truncate_integer(&-value, ty),
        UnaryOperator::Complement if operand.untyped => !value,
        UnaryOperator::Complement => truncate_integer(&!value, ty),
    };
    if operand.untyped && result.bits() > MAX_UNTYPED_INTEGER_BITS {
        return Err(Diagnostic::new(
            operator_span,
            "constant expression exceeds compiler resource limit",
        ));
    }
    if operator == UnaryOperator::Negate && !operand.untyped && !integer_fits(&result, ty) {
        return Err(Diagnostic::new(
            operator_span,
            format!("constant unary `-` on `{ty}` would trap"),
        ));
    }
    Ok(Some(result))
}

const MAX_UNTYPED_INTEGER_BITS: u64 = 2_000_000;

/// The largest shift count the compiler folds. An untyped shift has no width
/// to wrap against, so a large count is a resource limit rather than a result.
const MAX_CONSTANT_SHIFT: usize = 1_000_000;

/// Folds a shift of two constants. `right` is not negative: a negative shift
/// count is rejected before folding.
fn evaluate_shift(
    operator: BinaryOperator,
    operator_span: &std::ops::Range<usize>,
    ty: Scalar,
    untyped: bool,
    left: &BigInt,
    right: &BigInt,
) -> Result<BigInt, Diagnostic> {
    debug_assert!(right >= &BigInt::from(0u8));
    if untyped {
        return shift_untyped(operator, operator_span, left, right);
    }
    Ok(shift_typed(operator, ty, left, right))
}

/// Shifts an untyped constant, which has no width to shift bits out of.
fn shift_untyped(
    operator: BinaryOperator,
    operator_span: &std::ops::Range<usize>,
    left: &BigInt,
    right: &BigInt,
) -> Result<BigInt, Diagnostic> {
    let count = right.to_usize();
    if operator == BinaryOperator::ShiftRight && count.is_none() {
        return Ok(sign_fill(left));
    }
    let Some(count) = count else {
        return Err(Diagnostic::new(
            operator_span.clone(),
            "constant shift exceeds compiler resource limit",
        ));
    };
    if operator == BinaryOperator::ShiftLeft && shift_exceeds_limit(count, left) {
        return Err(Diagnostic::new(
            operator_span.clone(),
            "constant expression exceeds compiler resource limit",
        ));
    }
    Ok(if operator == BinaryOperator::ShiftLeft {
        left << count
    } else {
        left >> count
    })
}

/// Whether shifting `left` left by `count` would need more bits than the
/// compiler folds.
fn shift_exceeds_limit(count: usize, left: &BigInt) -> bool {
    count > MAX_CONSTANT_SHIFT
        || u64::try_from(count)
            .unwrap_or(u64::MAX)
            .saturating_add(left.bits())
            > MAX_UNTYPED_INTEGER_BITS
}

/// Shifts a typed constant. A count at or past the type's width shifts every
/// bit out.
fn shift_typed(operator: BinaryOperator, ty: Scalar, left: &BigInt, right: &BigInt) -> BigInt {
    if right >= &BigInt::from(ty.width()) {
        return if operator == BinaryOperator::ShiftRight && ty.signed() {
            sign_fill(left)
        } else {
            BigInt::from(0u8)
        };
    }
    let count = right
        .to_usize()
        .expect("count below every Fern integer width fits usize");
    if operator == BinaryOperator::ShiftLeft {
        truncate_integer(&(left << count), ty)
    } else {
        left >> count
    }
}

/// Folds a binary operation on two constants, reporting the operations the
/// program is not allowed to perform even when it never runs them.
fn evaluate_binary(
    operator: BinaryOperator,
    operator_span: std::ops::Range<usize>,
    spelling: &str,
    ty: Scalar,
    untyped: bool,
    operands: (&CheckedExpression, &CheckedExpression),
) -> Result<Option<BigInt>, Diagnostic> {
    let (left, right) = operands;
    let right_constant = right.integer();
    reject_constant_divisor(operator, &operator_span, spelling, right_constant)?;
    let (Some(left), Some(right)) = (left.integer(), right_constant) else {
        return Ok(None);
    };
    let result = fold_binary(operator, &operator_span, ty, untyped, left, right)?;
    check_constant_range(operator, &operator_span, ty, untyped, result).map(Some)
}

/// Rejects a right operand the operator cannot accept, whether or not the left
/// operand is constant.
fn reject_constant_divisor(
    operator: BinaryOperator,
    operator_span: &std::ops::Range<usize>,
    spelling: &str,
    right: Option<&BigInt>,
) -> Result<(), Diagnostic> {
    if matches!(operator, BinaryOperator::Divide | BinaryOperator::Remainder)
        && right == Some(&BigInt::from(0u8))
    {
        return Err(Diagnostic::new(
            operator_span.clone(),
            format!("constant `{spelling}` divisor is zero"),
        ));
    }
    if matches!(
        operator,
        BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight
    ) && right.is_some_and(|count| count < &BigInt::from(0u8))
    {
        let message = if spelling.ends_with('=') {
            format!("constant `{spelling}` shift count is negative")
        } else {
            "constant shift count is negative".to_owned()
        };
        return Err(Diagnostic::new(operator_span.clone(), message));
    }
    Ok(())
}

/// Applies the operator to two constants. The result is not range-checked
/// here, except where the operation itself would exceed what the compiler
/// folds.
fn fold_binary(
    operator: BinaryOperator,
    operator_span: &std::ops::Range<usize>,
    ty: Scalar,
    untyped: bool,
    left: &BigInt,
    right: &BigInt,
) -> Result<BigInt, Diagnostic> {
    Ok(match operator {
        BinaryOperator::Multiply => {
            if untyped
                && left.bits().saturating_add(right.bits()).saturating_sub(1)
                    > MAX_UNTYPED_INTEGER_BITS
            {
                return Err(Diagnostic::new(
                    operator_span.clone(),
                    "constant expression exceeds compiler resource limit",
                ));
            }
            left * right
        }
        BinaryOperator::Divide => {
            reject_division_trap("/", operator_span, ty, untyped, left, right)?;
            left / right
        }
        BinaryOperator::Remainder => {
            reject_division_trap("%", operator_span, ty, untyped, left, right)?;
            left % right
        }
        BinaryOperator::WrappingMultiply => truncate_integer(&(left * right), ty),
        BinaryOperator::Add => left + right,
        BinaryOperator::Subtract => left - right,
        BinaryOperator::WrappingAdd => truncate_integer(&(left + right), ty),
        BinaryOperator::WrappingSubtract => truncate_integer(&(left - right), ty),
        BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => {
            evaluate_shift(operator, operator_span, ty, untyped, left, right)?
        }
        BinaryOperator::And => left & right,
        BinaryOperator::Xor => left ^ right,
        BinaryOperator::Or => left | right,
    })
}

/// Dividing the most negative value of a type by `-1` traps, so a program that
/// spells it out is rejected at check time.
fn reject_division_trap(
    spelling: &str,
    operator_span: &std::ops::Range<usize>,
    ty: Scalar,
    untyped: bool,
    left: &BigInt,
    right: &BigInt,
) -> Result<(), Diagnostic> {
    if !untyped && is_minimum(left, ty) && right == &BigInt::from(-1) {
        return Err(Diagnostic::new(
            operator_span.clone(),
            format!("constant `{spelling}` on `{}` would trap", ty.name()),
        ));
    }
    Ok(())
}

/// Checks a folded result against the type it must fit, or against the
/// compiler's limit on how large an untyped constant may grow.
fn check_constant_range(
    operator: BinaryOperator,
    operator_span: &std::ops::Range<usize>,
    ty: Scalar,
    untyped: bool,
    result: BigInt,
) -> Result<BigInt, Diagnostic> {
    let checked_arithmetic = matches!(
        operator,
        BinaryOperator::Multiply | BinaryOperator::Add | BinaryOperator::Subtract
    );
    if checked_arithmetic && !untyped && !integer_fits(&result, ty) {
        return Err(Diagnostic::new(
            operator_span.clone(),
            format!(
                "constant `{}` on `{}` would overflow",
                operator.spelling(),
                ty.name()
            ),
        ));
    }
    if untyped && result.bits() > MAX_UNTYPED_INTEGER_BITS {
        return Err(Diagnostic::new(
            operator_span.clone(),
            "constant expression exceeds compiler resource limit",
        ));
    }
    Ok(result)
}

/// What shifting every bit out of a value leaves behind: its sign.
fn sign_fill(left: &BigInt) -> BigInt {
    if left < &BigInt::from(0u8) {
        BigInt::from(-1)
    } else {
        BigInt::from(0u8)
    }
}

fn concretize_value(
    syntax: &Syntax,
    id: Idx<Expression>,
    checked: &mut CheckedExpression,
    destination: Scalar,
) -> Result<Option<bool>, Diagnostic> {
    if !checked.untyped {
        return Ok(None);
    }
    let ty = checked
        .ty
        .scalar()
        .expect("an untyped expression has a scalar type");
    if ty.is_integer()
        && checked
            .integer()
            .is_some_and(|value| !integer_fits(value, destination))
    {
        return Err(out_of_range(syntax, id, destination));
    }
    let constant = checked.constant.is_some();
    checked.ty = destination.into();
    checked.untyped = false;
    Ok(Some(constant))
}

/// The checked form of an integer literal. A literal starts untyped, so its
/// value is kept exactly until the expression's type is known.
fn integer_literal(spelling: &str) -> CheckedExpression {
    let (base, digits, suffix) = integer_parts(spelling);
    debug_assert!(suffix.is_empty(), "frontend rejects literal suffixes");
    let value =
        BigUint::parse_bytes(digits.as_bytes(), base).expect("frontend validated integer digits");
    CheckedExpression {
        ty: Scalar::Int.into(),
        untyped: true,
        value: ExpressionValue::Integer,
        constant: Some(BigInt::from(value).into()),
    }
}

/// Compares two constant operands. Equality reaches every value type, so an
/// array folds elementwise; ordering reaches only integers, which is all
/// checking lets through.
fn compare_constants(operator: ComparisonOperator, left: &Constant, right: &Constant) -> BigInt {
    let ordering = || {
        let message = "ordering compares integer constants";
        left.integer()
            .expect(message)
            .cmp(right.integer().expect(message))
    };
    BigInt::from(match operator {
        ComparisonOperator::Equal => left == right,
        ComparisonOperator::NotEqual => left != right,
        ComparisonOperator::Less => ordering().is_lt(),
        ComparisonOperator::LessEqual => ordering().is_le(),
        ComparisonOperator::Greater => ordering().is_gt(),
        ComparisonOperator::GreaterEqual => ordering().is_ge(),
    })
}

fn require_boolean(
    syntax: &Syntax,
    id: Idx<Expression>,
    checked: &CheckedExpression,
) -> Result<(), Diagnostic> {
    if checked.ty == Scalar::Bool.into() {
        Ok(())
    } else {
        Err(Diagnostic::new(
            syntax.expressions[id].span.clone(),
            format!("logical operand has type `{}`, expected `bool`", checked.ty),
        ))
    }
}

fn constant_boolean(value: &BigInt) -> bool {
    debug_assert!(value == &BigInt::from(0u8) || value == &BigInt::from(1u8));
    value == &BigInt::from(1u8)
}

fn out_of_range(syntax: &Syntax, id: Idx<Expression>, destination: Scalar) -> Diagnostic {
    let literal = matches!(syntax.expressions[id].kind, ExpressionKind::Integer(_));
    Diagnostic::new(
        syntax.expressions[id].span.clone(),
        format!(
            "integer {} out of range for `{}`",
            if literal { "literal" } else { "value" },
            destination
        ),
    )
}

fn is_minimum(value: &BigInt, ty: Scalar) -> bool {
    value == &BigInt::from(ty.min())
}

fn integer_fits(value: &BigInt, ty: Scalar) -> bool {
    value >= &BigInt::from(ty.min()) && value <= &BigInt::from(ty.max())
}

fn integer_from_bits(bits: BigInt, ty: Scalar) -> BigInt {
    if ty.signed() && bits >= (BigInt::from(1u8) << (ty.width() - 1)) {
        bits - (BigInt::from(1u8) << ty.width())
    } else {
        bits
    }
}

fn truncate_integer(value: &BigInt, ty: Scalar) -> BigInt {
    let modulus = BigInt::from(1u8) << ty.width();
    let bits = ((value % &modulus) + &modulus) % &modulus;
    integer_from_bits(bits, ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SourceMap;

    fn parse(text: &str) -> Result<Syntax, Diagnostic> {
        crate::frontend::parse(&SourceMap::from_text(text))
    }

    fn literal(value: u128, base: u32) -> String {
        match base {
            2 => format!("0b{value:b}"),
            8 => format!("0o{value:o}"),
            10 => value.to_string(),
            16 => format!("0x{value:X}"),
            _ => unreachable!(),
        }
    }

    fn big(value: i128) -> BigInt {
        BigInt::from(value)
    }

    /// The recorded constant an integer-valued expression or binding folds to.
    fn folded(value: i128) -> Option<Constant> {
        Some(Constant::Integer(big(value)))
    }

    /// The recorded constant an array of integers folds to.
    fn folded_array(values: &[i128]) -> Option<Constant> {
        Some(Constant::Array(
            values
                .iter()
                .map(|&value| Constant::Integer(big(value)))
                .collect(),
        ))
    }

    /// The type and recorded constant of every binding of a checked program,
    /// in the order the bindings were declared.
    fn checked_bindings(text: &str) -> Vec<(Type, Option<Constant>)> {
        let syntax = parse(text).unwrap();
        let checked = check_root(&syntax).unwrap();
        checked
            .bindings
            .iter()
            .map(|(_, binding)| (binding.ty.clone(), binding.constant.clone()))
            .collect()
    }

    /// The value type a scalar spelling names, which is what checking records.
    fn value_type(scalar: Scalar) -> Type {
        Type::Scalar(scalar)
    }

    /// An array type, innermost element type first.
    fn array_type(length: u64, element: Type) -> Type {
        Type::Array {
            length,
            element: Box::new(element),
        }
    }

    /// A `counter` module beside the `app` root module, with one public
    /// function, one public `var`, one public `const`, and one private `const`.
    const COUNTER: &str = "pub var value = 0;
pub const step = 2;
const origin = 10;
pub fn bump(amount: int) -> int {
    value = value + amount;
    return value;
}
";

    /// Loads a tree of `(relative path, source)` files rooted at `app` and
    /// checks it, so multi-module tests run through real import resolution.
    fn load_tree<'a>(
        files: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> (tempfile::TempDir, crate::module::Program) {
        let dir = crate::module::tree(files);
        let program = crate::module::load(&dir.path().join("app"), &[dir.path().to_owned()])
            .unwrap_or_else(|error| panic!("{}", error.into_compile_error()));
        (dir, program)
    }

    fn accepts_tree<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>) {
        let (_dir, program) = load_tree(files);
        check(&program.syntax, &program.modules, &program.imports).unwrap();
    }

    fn tree_error<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>) -> Diagnostic {
        let (_dir, program) = load_tree(files);
        check(&program.syntax, &program.modules, &program.imports).unwrap_err()
    }

    fn rejects_tree<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>, message: &str) {
        assert_eq!(tree_error(files).message, message);
    }

    /// Rejects a tree whose root module is one `app/main.fern` file, checking
    /// the diagnostic's span against `marked`, where `«»` bracket it.
    fn rejects_root(marked: &str, message: &str) {
        let start = marked.find('«').unwrap();
        let end = marked.find('»').unwrap() - '«'.len_utf8();
        let main = marked.replace(['«', '»'], "");
        let error = tree_error([
            ("app/main.fern", main.as_str()),
            ("counter/counter.fern", COUNTER),
        ]);
        assert_eq!(error.message, message, "{marked}");
        assert_eq!(error.span, start..end, "{marked}");
    }

    #[test]
    fn annotations_resolve_to_array_types() {
        // An array annotation only resolves while its value is still rejected,
        // so each case reads the type back from a rejected initializer.
        for (source, expected) in [
            (
                "var a: [3]int = 1; fn main() -> void {}",
                array_type(3, value_type(Scalar::Int)),
            ),
            (
                "var a: [2][3]int = 1; fn main() -> void {}",
                array_type(2, array_type(3, value_type(Scalar::Int))),
            ),
            (
                "const n = 4; var a: [n]u8 = 1; fn main() -> void {}",
                array_type(4, value_type(Scalar::U8)),
            ),
            (
                "var a: [n]u8 = 1; const n = 4; fn main() -> void {}",
                array_type(4, value_type(Scalar::U8)),
            ),
            (
                "var a: [1 + 1]bool = 1; fn main() -> void {}",
                array_type(2, value_type(Scalar::Bool)),
            ),
        ] {
            let error = check_root(&parse(source).unwrap()).unwrap_err();
            assert_eq!(
                error.message,
                format!("cannot implicitly convert `int` to `{expected}`"),
                "{source}"
            );
        }
    }

    #[test]
    fn an_underscore_length_comes_from_an_array_literal_initializer() {
        assert_eq!(
            checked_bindings("var a: [_]int = [1, 2, 3]; fn main() -> void {}")[0].0,
            array_type(3, value_type(Scalar::Int))
        );

        for marked in [
            "const a: «[_]int» = 1; fn main() -> void {}",
            "fn main() -> void { var a: «[_]int» = 1; }",
        ] {
            rejects_root(marked, "`[_]` requires an array-literal initializer");
        }
        rejects_root(
            "const a: «[_]int» = [0...]; fn main() -> void {}",
            "`[_]` cannot take a length from a literal with a fill",
        );
        rejects_root(
            "const a: [_]«[_]int» = [[1, 2]]; fn main() -> void {}",
            "`[_]` requires an array-literal initializer",
        );
        for marked in [
            "fn f(a: «[_]int») -> void {} fn main() -> void {}",
            "fn f() -> «[_]int» { return 1; } fn main() -> void {}",
        ] {
            rejects_root(marked, "`[_]` requires an array-literal initializer");
        }
    }

    #[test]
    fn an_array_length_is_a_constant_int_of_at_least_one() {
        for (marked, message) in [
            (
                "var a: [«0»]int = 1; fn main() -> void {}",
                "array length must be at least 1",
            ),
            (
                "var a: [«-1»]int = 1; fn main() -> void {}",
                "array length must be at least 1",
            ),
            (
                "var n = 3; var a: [«n»]int = 1; fn main() -> void {}",
                "array length must be a constant expression",
            ),
            (
                "var a: [«true»]int = 1; fn main() -> void {}",
                "cannot implicitly convert `bool` to `int`",
            ),
            (
                "const n: [«n»]int = 1; fn main() -> void {}",
                "module-level initializer cycle involving `n`",
            ),
        ] {
            rejects_root(marked, message);
        }
        rejects_root(
            "fn main() -> void { var i = 0; var a: [«i»]int = 1; }",
            "array length must be a constant expression",
        );
        // A call is rejected before signatures are resolved, at module level
        // and in a function body alike.
        for marked in [
            "var a: [«size()»]int = 1; fn size() -> int { return 3; } fn main() -> void {}",
            "fn size() -> int { return 3; } fn main() -> void { var a: [«size()»]int = 1; }",
        ] {
            rejects_root(marked, "array length must be a constant expression");
        }
    }

    #[test]
    fn an_underscore_takes_its_length_from_the_literal_element_count() {
        for (source, expected) in [
            ("var a: [_]int = [1, 2, 3]; fn main() -> void {}", 3),
            (
                "var a: [_][2]int = [[1, 2], [3, 4]]; fn main() -> void {}",
                2,
            ),
            ("var a: [_]int = [7]; fn main() -> void {}", 1),
        ] {
            let syntax = parse(source).unwrap();
            let StatementKind::Binding { initializer, .. } =
                &syntax.statements[match syntax.files[0].items[0] {
                    TopLevelItem::Binding { binding, .. } => binding,
                    TopLevelItem::Function { .. } => unreachable!("the first item is a binding"),
                }]
                .kind
            else {
                unreachable!("the first item is a binding")
            };
            let length = inferred_length(&syntax, &(0..0), Some(*initializer)).unwrap();
            assert_eq!(length, expected, "{source}");
        }
    }

    #[test]
    fn an_array_literal_takes_its_type_from_context() {
        for (source, expected) in [
            (
                "fn main() -> void { var a: [3]int = [1, 2, 3]; }",
                array_type(3, value_type(Scalar::Int)),
            ),
            (
                "fn main() -> void { var a: [2]u8 = [0, 255]; }",
                array_type(2, value_type(Scalar::U8)),
            ),
            (
                "fn main() -> void { var a: [2]bool = [true, false]; }",
                array_type(2, value_type(Scalar::Bool)),
            ),
            (
                "fn main() -> void { var a: [2][3]int = [[1, 2, 3], [4, 5, 6]]; }",
                array_type(2, array_type(3, value_type(Scalar::Int))),
            ),
        ] {
            assert_eq!(checked_bindings(source)[0].0, expected, "{source}");
        }
        // A parameter, a result, and an assignment target give a literal its
        // type the same way an annotation does.
        accepts_source("fn take(a: [2]u8) -> void {} fn main() -> void { take([0, 255]); }");
        accepts_source("fn make() -> [2]u8 { return [0, 255]; } fn main() -> void {}");
        accepts_source("fn main() -> void { var a: [2]u8 = [0, 0]; a = [1, 255]; }");
        // Context reaches the literal itself and no further, so a grouped
        // literal is checked with none.
        rejects_root(
            "fn main() -> void { var a: [2]u8 = «([0, 255])»; }",
            "cannot implicitly convert `[2]int` to `[2]u8`",
        );
        rejects_root(
            "fn main() -> void { var a: int = «[1, 2]»; }",
            "cannot implicitly convert an array literal to `int`",
        );
        rejects_root(
            "fn main() -> void { var a: [2]u8 = [0, «256»]; }",
            "integer literal out of range for `u8`",
        );
    }

    #[test]
    fn a_literal_without_context_takes_one_common_element_type() {
        for (source, expected) in [
            (
                "fn main() -> void { var a = [1, 2, 3]; }",
                array_type(3, value_type(Scalar::Int)),
            ),
            (
                "fn main() -> void { var a = [true, false]; }",
                array_type(2, value_type(Scalar::Bool)),
            ),
            (
                "fn main() -> void { var x: u8 = 1; var a = [1, x]; }",
                array_type(2, value_type(Scalar::U8)),
            ),
            (
                "fn main() -> void { var x: u8 = 1; var a = [x, 1]; }",
                array_type(2, value_type(Scalar::U8)),
            ),
            (
                "fn main() -> void { var a = [[1, 2], [3, 4]]; }",
                array_type(2, array_type(2, value_type(Scalar::Int))),
            ),
        ] {
            assert_eq!(
                checked_bindings(source).last().unwrap().0,
                expected,
                "{source}"
            );
        }
        rejects_root(
            "fn main() -> void { var x: u8 = 1; var y: int = 1; var a = [x, «y»]; }",
            "cannot implicitly convert `int` to `u8`",
        );
    }

    #[test]
    fn an_array_literal_has_one_element_per_array_element() {
        rejects_root(
            "fn main() -> void { var a: [3]int = «[1, 2]»; }",
            "expected 3 elements for `[3]int`, found 2",
        );
        rejects_root(
            "fn main() -> void { var a: [2]int = «[1, 2, 3]»; }",
            "expected 2 elements for `[2]int`, found 3",
        );
        rejects_root(
            "fn take(a: [3]int) -> void {} fn main() -> void { take(«[1, 2]»); }",
            "expected 3 elements for `[3]int`, found 2",
        );
        rejects_root(
            "fn make() -> [2]int { return «[1, 2, 3]»; } fn main() -> void {}",
            "expected 2 elements for `[2]int`, found 3",
        );
    }

    #[test]
    fn a_fill_repeats_the_last_element_across_the_remaining_elements() {
        for (source, expected) in [
            ("const a: [2]int = [7...]; fn main() -> void {}", vec![7, 7]),
            (
                "const a: [8]int = [1, 2, 3, 0...]; fn main() -> void {}",
                vec![1, 2, 3, 0, 0, 0, 0, 0],
            ),
            (
                "const a: [3]int = [1, 2, 3...]; fn main() -> void {}",
                vec![1, 2, 3],
            ),
        ] {
            assert_eq!(
                checked_bindings(source)[0].1,
                folded_array(&expected),
                "{source}"
            );
        }
        // A nested fill takes its length from the element type.
        let row = Constant::Array(vec![Constant::Integer(big(1)); 3]);
        assert_eq!(
            checked_bindings("const a: [2][3]u8 = [[1...]...]; fn main() -> void {}")[0].1,
            Some(Constant::Array(vec![row.clone(), row]))
        );
        rejects_root(
            "fn main() -> void { var a: [2]int = «[1, 2, 3...]»; }",
            "expected at most 2 elements for `[2]int`, found 3",
        );
        rejects_root(
            "fn main() -> void { var bad = [0«...»]; }",
            "a fill requires a length from context",
        );
    }

    #[test]
    fn a_module_level_declaration_holds_a_constant_array() {
        let source = "var counts: [3]int = [1, 2, 3];
const limits: [2]u8 = [0, 255];
fn main() -> void {}";
        assert_eq!(
            checked_bindings(source),
            [
                (array_type(3, value_type(Scalar::Int)), None),
                (
                    array_type(2, value_type(Scalar::U8)),
                    folded_array(&[0, 255])
                ),
            ]
        );
        // A module-level `var` records no constant on its binding, so the
        // folded array is read back from the initializer instead.
        let syntax = parse(source).unwrap();
        let checked = check_root(&syntax).unwrap();
        let initializers: Vec<_> = checked
            .module_bindings
            .iter()
            .map(|&statement| match syntax.statements[statement].kind {
                StatementKind::Binding { initializer, .. } => {
                    checked.expressions[initializer].constant.clone()
                }
                _ => unreachable!("module bindings are binding statements"),
            })
            .collect();
        assert_eq!(
            initializers,
            [folded_array(&[1, 2, 3]), folded_array(&[0, 255])]
        );
        rejects_root(
            "var seed = 1; var a: [2]int = «[seed, 2]»; fn main() -> void {}",
            "module-level initializer must be a constant expression",
        );
    }

    #[test]
    fn a_whole_array_value_requires_an_identical_type() {
        let bindings = checked_bindings("fn main() -> void { var a: [2]int = [1, 2]; var b = a; }");
        assert_eq!(bindings[0].0, bindings[1].0);
        accepts_source(
            "fn take(a: [2]int) -> [2]int { return a; }
             fn main() -> void { var a: [2]int = [1, 2]; var b: [2]int = [3, 4]; a = take(b); }",
        );
        rejects_root(
            "fn main() -> void { var a: [3]int = [1, 2, 3]; var b: [2]int = [1, 2]; a = «b»; }",
            "cannot implicitly convert `[2]int` to `[3]int`",
        );
        rejects_root(
            "fn take(a: [3]int) -> void {}
fn main() -> void { var b: [2]int = [1, 2]; take(«b»); }",
            "cannot implicitly convert `[2]int` to `[3]int`",
        );
        rejects_root(
            "fn make() -> [3]int { var b: [2]int = [1, 2]; return «b»; }
fn main() -> void {}",
            "cannot implicitly convert `[2]int` to `[3]int`",
        );
        rejects_root(
            "fn main() -> void { var a: [2]int = [1, 2]; var b = a «+» 1; }",
            "integer `+` requires integer operands",
        );
    }

    #[test]
    fn indexing_an_array_yields_its_element_type() {
        for (source, expected) in [
            (
                "fn main() -> void { var a: [3]int = [1, 2, 3]; var e = a[0]; }",
                value_type(Scalar::Int),
            ),
            (
                "fn main() -> void { var g: [2][3]u8 = [[1...]...]; var e = g[1]; }",
                array_type(3, value_type(Scalar::U8)),
            ),
            (
                "fn main() -> void { var g: [2][3]u8 = [[1...]...]; var e = g[1][2]; }",
                value_type(Scalar::U8),
            ),
        ] {
            assert_eq!(
                checked_bindings(source).last().unwrap().0,
                expected,
                "{source}"
            );
        }
        accepts_source("fn main() -> void { var a: [3]int = [1, 2, 3]; var i = 2; exit(a[i]); }");
        rejects_root(
            "fn main() -> void { var x = 1; var e = «x»[0]; }",
            "cannot index `int`",
        );
        rejects_root(
            "fn main() -> void { var a: [3]int = [1, 2, 3]; var i: u8 = 1; var e = a[«i»]; }",
            "cannot implicitly convert `u8` to `int`",
        );
        rejects_root(
            "fn main() -> void { var a: [3]int = [1, 2, 3]; var e = a[«true»]; }",
            "cannot implicitly convert `bool` to `int`",
        );
    }

    #[test]
    fn a_constant_index_must_be_in_range_and_never_folds() {
        accepts_source("fn main() -> void { const a: [3]int = [1, 2, 3]; exit(a[2]); }");
        rejects_root(
            "fn main() -> void { const a: [3]int = [1, 2, 3]; exit(a[«3»]); }",
            "index 3 is out of range for `[3]int`",
        );
        rejects_root(
            "fn main() -> void { const a: [3]int = [1, 2, 3]; exit(a[«-1»]); }",
            "index -1 is out of range for `[3]int`",
        );
        // `a[i]` is never a constant expression, even when the array and the
        // index both are.
        rejects_root(
            "const a: [3]int = [1, 2, 3]; const first = «a[0]»; fn main() -> void {}",
            "module-level initializer must be a constant expression",
        );
    }

    #[test]
    fn an_element_of_a_mutable_array_may_be_assigned() {
        accepts_source("fn main() -> void { var a: [3]int = [1, 2, 3]; a[0] = 9; a[1] += 1; }");
        accepts_source(
            "fn main() -> void { var g: [2][3]int = [[1...]...]; g[0][1] = 9; g[1] = [4, 5, 6]; }",
        );
        rejects_root(
            "fn main() -> void { const a: [3]int = [1, 2, 3]; «a»[0] = 9; }",
            "cannot assign to immutable binding `a`",
        );
        rejects_root(
            "fn main() -> void { var a: [3]int = [1, 2, 3]; a[0] = «true»; }",
            "cannot implicitly convert `bool` to `int`",
        );
        rejects_root(
            "fn main() -> void { var a: [3]int = [1, 2, 3]; «a»[0][0] = 9; }",
            "cannot index `int`",
        );
        rejects_root(
            "fn main() -> void { var a: [3]int = [1, 2, 3]; a[«3»] = 9; }",
            "index 3 is out of range for `[3]int`",
        );
        rejects_root(
            "fn main() -> void { var g: [2][3]int = [[1...]...]; g[0] «+=» 1; }",
            "integer `+=` requires integer operands",
        );
    }

    #[test]
    fn len_reads_the_length_from_the_operands_type() {
        for (source, expected) in [
            (
                "fn main() -> void { const a: [3]int = [1, 2, 3]; const n = len(a); }",
                3,
            ),
            (
                "fn main() -> void { var a: [4]u8 = [0...]; const n = len(a); }",
                4,
            ),
            (
                "fn main() -> void { var g: [2][3]int = [[1...]...]; const n = len(g[0]); }",
                3,
            ),
        ] {
            assert_eq!(
                checked_bindings(source).last().unwrap(),
                &(value_type(Scalar::Int), folded(expected)),
                "{source}"
            );
        }
        // A folded length is usable wherever a constant expression is.
        assert_eq!(
            checked_bindings(
                "const a: [3]int = [1, 2, 3];
                 fn main() -> void { var b: [len(a)]u8 = [0...]; }"
            )[1]
            .0,
            array_type(3, value_type(Scalar::U8))
        );
        rejects_root(
            "fn main() -> void { var x = 1; const n = len(«x»); }",
            "`len` requires an array operand, found `int`",
        );
        // Reaching the operand's type through a call means the length does not
        // fold, because the call is still evaluated.
        rejects_root(
            "fn make() -> [3]int { return [1, 2, 3]; }
const n = len(«make()»);
fn main() -> void {}",
            "module-level initializer must be a constant expression",
        );
    }

    #[test]
    fn arrays_compare_for_equality_only() {
        accepts_source(
            "fn main() -> void {
                 var a: [3]int = [1, 2, 3];
                 var b: [3]int = [1, 2, 3];
                 if a == b { exit(1); }
                 if a != b { exit(2); }
             }",
        );
        for (source, expected) in [
            (
                "const a: [3]int = [1, 2, 3]; const same = a == [1, 2, 3]; fn main() -> void {}",
                folded(1),
            ),
            (
                "const a: [3]int = [1, 2, 3]; const same = a == [1, 2, 4]; fn main() -> void {}",
                folded(0),
            ),
            (
                "const a: [3]int = [1, 2, 3]; const same = a != [1, 2, 3]; fn main() -> void {}",
                folded(0),
            ),
        ] {
            assert_eq!(checked_bindings(source)[1].1, expected, "{source}");
        }
        rejects_root(
            "fn main() -> void { var a: [3]int = [1, 2, 3]; var b: [3]int = [1, 2, 3];
             if a «<» b { exit(1); } }",
            "only `==` and `!=` are defined on `[3]int`",
        );
        rejects_root(
            "fn main() -> void { var a: [3]int = [1, 2, 3]; var b: [2]int = [1, 2];
             if a «==» b { exit(1); } }",
            "comparison operands have different types `[3]int` and `[2]int`",
        );
        rejects_root(
            "fn main() -> void { var a: [3]int = [1, 2, 3]; var x = 1;
             if a «==» x { exit(1); } }",
            "comparison operands have different types `[3]int` and `int`",
        );
    }

    #[test]
    fn for_in_binds_an_element_and_an_optional_index() {
        for (source, value, index) in [
            (
                "fn main() -> void { var a: [2]u8 = [0...]; for v in a { exit(int(v)); } }",
                value_type(Scalar::U8),
                None,
            ),
            (
                "fn main() -> void { var g: [2][3]int = [[1...]...]; for row, i in g { exit(i); } }",
                array_type(3, value_type(Scalar::Int)),
                Some(value_type(Scalar::Int)),
            ),
        ] {
            let syntax = parse(source).unwrap();
            let checked = check_root(&syntax).unwrap();
            let (_, bindings) = checked.iterations.iter().next().unwrap();
            assert_eq!(checked.bindings[bindings.value].ty, value, "{source}");
            assert!(!checked.bindings[bindings.value].mutable, "{source}");
            assert_eq!(
                bindings
                    .index
                    .map(|binding| checked.bindings[binding].ty.clone()),
                index,
                "{source}"
            );
        }
        accepts_source(
            "fn main() -> void { var g: [2][3]int = [[1...]...];
             for row in g { for v in row { exit(v); } } }",
        );
        // The body may shadow the bindings, and neither outlives the loop.
        accepts_source(
            "fn main() -> void { var a: [2]int = [1, 2]; for v in a { const v = 9; exit(v); } }",
        );
        rejects_root(
            "fn main() -> void { var a: [2]int = [1, 2]; for v in a {} exit(«v»); }",
            "unknown binding `v`",
        );
        rejects_root(
            "fn main() -> void { var a: [2]int = [1, 2]; for v in a { «v» = 9; } }",
            "cannot assign to immutable binding `v`",
        );
        rejects_root(
            "fn main() -> void { var a: [2]int = [1, 2]; for v, «v» in a {} }",
            "a `for` loop's value and index bindings must have different names",
        );
        rejects_root(
            "fn main() -> void { var x = 1; for v in «x» {} }",
            "`for … in` requires an array, found `int`",
        );
    }

    #[test]
    fn public_declarations_are_reachable_through_both_import_forms() {
        accepts_tree([
            (
                "app/main.fern",
                "use counter;
                 fn main() -> void {
                     counter::value = counter::bump(counter::step);
                     exit(counter::value + doubled());
                 }",
            ),
            (
                "app/totals.fern",
                "use counter::{step, bump};
                 fn doubled() -> int { return bump(step) * step; }",
            ),
            ("counter/counter.fern", COUNTER),
        ]);
    }

    #[test]
    fn a_private_declaration_is_module_wide_but_not_visible_to_importers() {
        // Another file of the same module reads the private `origin`.
        accepts_tree([
            (
                "app/main.fern",
                "use counter::{doubled_origin};
                 fn main() -> void { exit(doubled_origin); }",
            ),
            ("counter/counter.fern", COUNTER),
            (
                "counter/extra.fern",
                "pub const doubled_origin = origin * 2;",
            ),
        ]);
        rejects_root(
            "use counter; fn main() -> void { exit(«counter::origin»); }",
            "declaration `origin` is private to module `counter`",
        );
        rejects_root(
            "use counter::{«origin»}; fn main() -> void { exit(origin); }",
            "declaration `origin` is private to module `counter`",
        );
        rejects_root(
            "use counter; fn main() -> void { exit(«counter::missing»); }",
            "module `counter` has no declaration named `missing`",
        );
        rejects_root(
            "use counter::{«missing»}; fn main() -> void { exit(missing); }",
            "module `counter` has no declaration named `missing`",
        );
    }

    #[test]
    fn imported_bindings_keep_their_mutability_across_modules() {
        accepts_tree([
            (
                "app/main.fern",
                "use counter::{value};
                 fn main() -> void { value = 7; exit(value); }",
            ),
            ("counter/counter.fern", COUNTER),
        ]);
        rejects_root(
            "use counter; fn main() -> void { «counter::step» = 1; exit(0); }",
            "cannot assign to immutable binding `step`",
        );
        rejects_root(
            "use counter::{step}; fn main() -> void { «step» = 1; exit(0); }",
            "cannot assign to immutable binding `step`",
        );
    }

    #[test]
    fn a_module_level_initializer_reads_an_imported_constant() {
        accepts_tree([
            (
                "app/main.fern",
                "use counter::{step};
                 const total = step * 3;
                 fn main() -> void { exit(total); }",
            ),
            ("counter/counter.fern", COUNTER),
        ]);
        accepts_tree([
            (
                "app/main.fern",
                "use counter;
                 const total = counter::step * 3;
                 fn main() -> void { exit(total); }",
            ),
            ("counter/counter.fern", COUNTER),
        ]);
    }

    #[test]
    fn an_introduced_name_must_not_collide_in_its_file() {
        for (marked, name) in [
            (
                "use counter::{«step»}; const step = 1; fn main() -> void { exit(step); }",
                "step",
            ),
            (
                "use counter::{«bump»}; fn bump() -> void {} fn main() -> void { bump(); }",
                "bump",
            ),
        ] {
            rejects_root(
                marked,
                &format!("imported name `{name}` conflicts with a module-level declaration"),
            );
        }
        rejects_root(
            "use counter::{step}; use counter::{«step»}; fn main() -> void { exit(step); }",
            "duplicate imported name `step`",
        );
        rejects_root(
            "use counter; use «counter»; fn main() -> void { exit(counter::step); }",
            "duplicate imported name `counter`",
        );
    }

    #[test]
    fn an_introduced_name_must_be_referenced_in_its_own_file() {
        rejects_root(
            "use «counter»; fn main() -> void { exit(0); }",
            "imported name `counter` is never referenced",
        );
        rejects_root(
            "use counter::{«step»}; fn main() -> void { exit(0); }",
            "imported name `step` is never referenced",
        );
        // An import is file-local, so the other file neither sees it nor keeps
        // it referenced.
        rejects_tree(
            [
                (
                    "app/main.fern",
                    "use counter::{step}; fn main() -> void { exit(0); }",
                ),
                (
                    "app/totals.fern",
                    "fn tripled() -> int { return step * 3; }",
                ),
                ("counter/counter.fern", COUNTER),
            ],
            "unknown binding `step`",
        );
        rejects_tree(
            [
                (
                    "app/main.fern",
                    "use counter::{step}; fn main() -> void { exit(0); }",
                ),
                (
                    "app/totals.fern",
                    "use counter::{step}; fn tripled() -> int { return step * 3; }",
                ),
                ("counter/counter.fern", COUNTER),
            ],
            "imported name `step` is never referenced",
        );
    }

    #[test]
    fn a_local_binding_shadows_an_imported_name() {
        accepts_tree([
            (
                "app/main.fern",
                "use counter::{step};
                 fn main() -> void {
                     const outer = step;
                     { var step = 5; step = 7; }
                     exit(outer);
                 }",
            ),
            ("counter/counter.fern", COUNTER),
        ]);
        rejects_root(
            "use counter; fn main() -> void { const counter = 1; exit(«counter»::step); }",
            "cannot use binding `counter` as a module",
        );
    }

    #[test]
    fn a_qualified_name_resolves_only_through_an_imported_module() {
        rejects_root(
            "use counter; fn main() -> void { exit(counter::step + «text»::width); }",
            "unknown module `text`",
        );
        rejects_root(
            "use counter::{step}; fn main() -> void { exit(«step»::inner); }",
            "`step` is not a module",
        );
        rejects_root(
            "use counter; fn main() -> void { exit(«counter»); }",
            "module `counter` is not a value",
        );
        rejects_root(
            "use counter; fn main() -> void { «counter»(); }",
            "module `counter` is not a function",
        );
        rejects_root(
            "use counter; fn main() -> void { const f = «counter::bump»; exit(f); }",
            "cannot use function `bump` as a value",
        );
        rejects_root(
            "use counter; fn main() -> void { «counter::value»(); }",
            "cannot call non-function declaration `value`",
        );
        rejects_root(
            "use counter::{value}; fn main() -> void { «value»(); }",
            "cannot call non-function declaration `value`",
        );
    }

    #[test]
    fn only_the_root_modules_main_is_the_entry_point() {
        let (_dir, program) = load_tree([
            (
                "app/main.fern",
                "use counter; fn main() -> void { exit(counter::bump(1)); }",
            ),
            (
                "counter/counter.fern",
                "pub var value = 0;
                 pub fn bump(amount: int) -> int { value = value + amount; return value; }
                 fn main(flag: int) -> int { return flag; }",
            ),
        ]);
        let checked = check(&program.syntax, &program.modules, &program.imports).unwrap();
        let root = program.modules.last().unwrap();
        let entry = program.syntax.files[root.files.start].items[0];
        assert!(matches!(
            entry,
            TopLevelItem::Function { function, .. } if function == checked.main
        ));

        rejects_tree(
            [
                (
                    "app/main.fern",
                    "use counter; fn helper() -> int { return counter::value; }",
                ),
                (
                    "counter/counter.fern",
                    "pub var value = 1; fn main() -> void {}",
                ),
            ],
            "missing `main` function",
        );
    }

    #[test]
    fn bindings_have_concrete_types_and_distinct_identities() {
        let text =
            "fn main() -> void { const x = 1; var x: int = x; const x = x; exit(x); exit(0); }";
        let syntax = parse(text).unwrap();
        let checked = check_root(&syntax).unwrap();
        assert!(std::ptr::eq(checked.syntax, &syntax));
        let statements = &syntax.functions[checked.main].body;
        let ids: Vec<_> = statements[..3]
            .iter()
            .map(|s| checked.declarations[*s])
            .collect();
        assert_ne!(ids[0], ids[1]);
        assert_ne!(ids[1], ids[2]);
        assert_ne!(ids[0], ids[2]);
        for id in &ids {
            assert_eq!(checked.bindings[*id].ty, value_type(Scalar::Int));
        }
        let facts: Vec<_> = syntax
            .expressions
            .iter()
            .map(|(id, _)| &checked.expressions[id])
            .collect();
        assert_eq!(facts.len(), 5);
        assert!(facts.iter().all(|fact| fact.ty == value_type(Scalar::Int)));
        assert_eq!(facts[0].value, ExpressionValue::Integer);
        assert_eq!(facts[1].value, ExpressionValue::Reference(ids[0]));
        assert_eq!(facts[2].value, ExpressionValue::Reference(ids[1]));
        assert_eq!(facts[3].value, ExpressionValue::Reference(ids[2]));
        assert_eq!(facts[4].value, ExpressionValue::Integer);
    }

    #[test]
    fn nested_scopes_resolve_binding_identity_and_mutability() {
        let syntax = parse("fn main() -> void { var x = 1; { x = 2; const x = x; { var x = x; x = x; } exit(x); } x = x; const x = x; exit(x); }").unwrap();
        let checked = check_root(&syntax).unwrap();
        let ids: Vec<_> = checked.bindings.iter().map(|(id, _)| id).collect();
        assert_eq!(ids.len(), 4);
        let mutable: Vec<_> = checked.bindings.iter().map(|(_, b)| b.mutable).collect();
        assert_eq!(mutable, [true, false, true, false]);
        let targets: Vec<_> = checked
            .assignments
            .iter()
            .map(|(_, target)| target.binding)
            .collect();
        assert_eq!(targets, [ids[0], ids[2], ids[0]]);
        let references: Vec<_> = checked
            .expressions
            .iter()
            .filter_map(|(_, expression)| match &expression.value {
                ExpressionValue::Reference(id) => Some(*id),
                ExpressionValue::Integer
                | ExpressionValue::Boolean
                | ExpressionValue::Conversion { .. }
                | ExpressionValue::Grouping { .. }
                | ExpressionValue::Unary { .. }
                | ExpressionValue::Binary { .. }
                | ExpressionValue::Comparison { .. }
                | ExpressionValue::Logical { .. }
                | ExpressionValue::Array { .. }
                | ExpressionValue::Index { .. }
                | ExpressionValue::Length { .. }
                | ExpressionValue::LogicalNot { .. }
                | ExpressionValue::Call { .. } => None,
            })
            .collect();
        assert_eq!(
            references,
            [ids[0], ids[1], ids[2], ids[1], ids[0], ids[0], ids[3]]
        );
    }

    #[test]
    fn assignment_errors_use_target_or_value_spans_even_after_nested_exit() {
        for (body, offending, message) in [
            (
                "const x = 1; x = 2;",
                "x",
                "cannot assign to immutable binding `x`",
            ),
            (
                "var x = 1; const x = 2; x = 3;",
                "x",
                "cannot assign to immutable binding `x`",
            ),
            (
                "var x = 1; { const x = 2; x = 3; }",
                "x",
                "cannot assign to immutable binding `x`",
            ),
            (
                "const x = 1; { var x = 2; x = 3; } x = 4;",
                "x",
                "cannot assign to immutable binding `x`",
            ),
            ("x = 2;", "x", "unknown binding `x`"),
            ("{ var x = 1; } x = 2;", "x", "unknown binding `x`"),
            ("{ const x = 1; } exit(x);", "x", "unknown binding `x`"),
            ("{ var x = x; }", "x", "unknown binding `x`"),
            (
                "var x = 1; { x = missing; }",
                "missing",
                "unknown binding `missing`",
            ),
            (
                "var x: u8 = 1; { x = 256; }",
                "256",
                "integer literal out of range for `u8`",
            ),
        ] {
            rejects(body, offending, message);
            rejects(&format!("{{ exit(0); }} {body}"), offending, message);
            rejects(&format!("{{ exit(0); {body} }}"), offending, message);
        }
        for body in [
            "const x = 1; var x = x; x = 2;",
            "const x = 1; { var x = x; x = 2; }",
            "var x = 1; { const x = x; } x = 2;",
        ] {
            let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
            check_root(&syntax).unwrap();
        }
    }

    fn rejects(body: &str, offending: &str, message: &str) {
        let text = format!("/* 🌿 */ fn main() -> void {{ {body} }}");
        let syntax = parse(&text).unwrap();
        let error = check_root(&syntax).unwrap_err();
        let start = text.rfind(offending).unwrap();
        assert_eq!(error.span, start..start + offending.len(), "{body}");
        assert_eq!(error.message, message, "{body}");
    }

    fn accepts(body: &str) {
        let text = format!("fn main() -> void {{ {body} }}");
        let syntax = parse(&text).unwrap();
        check_root(&syntax).unwrap();
    }

    fn rejects_source(text: &str, offending: &str, message: &str) {
        let syntax = parse(text).unwrap();
        let error = check_root(&syntax).unwrap_err();
        let start = text.rfind(offending).unwrap();
        assert_eq!(error.span, start..start + offending.len(), "{text}");
        assert_eq!(error.message, message, "{text}");
    }

    fn accepts_source(text: &str) {
        let syntax = parse(text).unwrap();
        check_root(&syntax).unwrap();
    }

    #[test]
    fn returns_check_against_the_declared_result() {
        let text = "fn nothing() -> void { return; }
                    fn early(flag: bool) -> void {
                        if flag {
                            return;
                        }
                        exit(0);
                    }
                    fn falls_through() -> void {}
                    fn narrow() -> u8 { return 200; }
                    fn wide() -> i64 { return 1 + 2; }
                    fn ready() -> bool { return 1 < 2; }
                    fn branching(flag: bool) -> int {
                        if flag {
                            return 1;
                        } else {
                            return 2;
                        }
                    }
                    fn main() -> void {}";
        let syntax = parse(text).unwrap();
        let checked = check_root(&syntax).unwrap();
        let returned = syntax
            .statements
            .iter()
            .filter_map(|(_, statement)| match &statement.kind {
                StatementKind::Return { value: Some(value) } => Some((
                    &text[syntax.expressions[*value].span.clone()],
                    checked.expressions[*value].ty.clone(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            returned,
            [
                ("200", value_type(Scalar::U8)),
                ("1 + 2", value_type(Scalar::I64)),
                ("1 < 2", value_type(Scalar::Bool)),
                ("1", value_type(Scalar::Int)),
                ("2", value_type(Scalar::Int)),
            ]
        );

        rejects_source(
            "fn main() -> void {} fn nothing() -> void { return 1; }",
            "1",
            "cannot return a value from a `void` function",
        );
        rejects_source(
            "fn main() -> void {} fn total() -> int { return; }",
            "return;",
            "`return` must supply a value of type `int`",
        );
        rejects_source(
            "fn main() -> void {} fn total() -> int { return true; }",
            "true",
            "cannot implicitly convert `bool` to `int`",
        );
        rejects_source(
            "fn main() -> void {} fn narrow() -> u8 { return 256; }",
            "256",
            "integer literal out of range for `u8`",
        );

        // Statements after a `return` are still checked.
        rejects(
            "return; exit(missing);",
            "missing",
            "unknown binding `missing`",
        );
    }

    #[test]
    fn value_returning_functions_must_not_reach_the_end_of_their_body() {
        for body in [
            "if flag { return 1; } else { return 2; }",
            "if flag { return 1; } else if flag { return 2; } else { return 3; }",
            "{ return 1; }",
            "exit(0);",
            "for { }",
            "for { for { break; } }",
            "for :outer { for :inner { break :inner; } }",
            "for { return 1; break; }",
            "if flag { return 1; } else { for { } }",
        ] {
            accepts_source(&format!(
                "fn main() -> void {{}} fn total(flag: bool) -> int {{ {body} }}"
            ));
        }

        for body in [
            "",
            "if flag { return 1; }",
            "if flag { return 1; } else if flag { return 2; }",
            "for flag { return 1; }",
            "for var i = 0; i < 1; i = i + 1 { return 1; }",
            "for { break; }",
            "for { if flag { break; } }",
            "for :outer { for { break :outer; } }",
            "for { { break; } }",
        ] {
            let text = format!("fn main() -> void {{}} fn total(flag: bool) -> int {{ {body} }}");
            rejects_source(
                &text,
                "total",
                "function `total` can reach the end of its body without returning a value",
            );
        }
    }

    #[test]
    fn function_signatures_are_collected_before_call_checking() {
        let text = "fn caller(value: int, flag: bool,) -> int {
                        callee(value, flag);
                        const nested: int = callee(callee(value, flag), flag);
                        return nested;
                     }
                     fn callee(value: int, flag: bool) -> int { return value; }
                     fn recursive(value: int) -> void { recursive(value); }
                     fn mutual_left(value: int) -> int { mutual_right(value); return value; }
                     fn mutual_right(value: int) -> void { mutual_left(value); }
                     fn main() -> void {
                         caller(1, true);
                         callee(2, false);
                         recursive(3);
                         mutual_left(4);
                     }";
        let syntax = parse(text).unwrap();
        let checked = check_root(&syntax).unwrap();

        assert_eq!(checked.function_names.len(), 6);
        assert_eq!(checked.functions.iter().count(), 6);
        assert_eq!(checked.calls.iter().count(), 8);
        assert!(
            checked
                .expressions
                .iter()
                .any(|(_, expression)| matches!(expression.value, ExpressionValue::Call { .. }))
        );
        for (_, signature) in checked.functions.iter() {
            assert!(
                signature
                    .parameters
                    .iter()
                    .all(|parameter| !checked.bindings[*parameter].mutable)
            );
        }
    }

    #[test]
    fn call_arguments_use_parameter_types_and_source_order() {
        let syntax = parse(
            "fn typed(value: u8, flag: bool) -> void {
                 const copy: u8 = value;
                 const ready: bool = flag;
             }
             fn main() -> void {
                 typed(1, true);
                 typed(1 + 2, false);
             }",
        )
        .unwrap();
        let checked = check_root(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| (binding.ty.clone(), binding.mutable))
                .collect::<Vec<_>>(),
            [
                (value_type(Scalar::U8), false),
                (value_type(Scalar::Bool), false),
                (value_type(Scalar::U8), false),
                (value_type(Scalar::Bool), false),
            ]
        );

        rejects_source(
            "fn typed(value: u8, flag: bool) -> void {} fn main() -> void { typed(256, missing); }",
            "256",
            "integer literal out of range for `u8`",
        );
    }

    #[test]
    fn calls_respect_shadowing_context_and_result_kind() {
        rejects_source(
            "fn target() -> void {} fn main() -> void { var target = 0; target(); }",
            "target",
            "cannot call non-function binding `target`",
        );
        rejects_source(
            "fn target(target: int) -> void { target(); } fn main() -> void {}",
            "target",
            "cannot call non-function binding `target`",
        );
        rejects_source(
            "fn target() -> void {} fn main() -> void { const value = target(); }",
            "target",
            "void function `target` cannot be used as a value",
        );
        rejects_source(
            "fn target(value: int) -> int { return value; } fn main() -> void { target(); }",
            "target",
            "function `target` expects 1 argument, found 0",
        );
        rejects_source(
            "fn main() -> void { missing(); }",
            "missing",
            "unknown function `missing`",
        );
        rejects_source(
            "fn target(first: int, second: bool) -> void {} fn main() -> void { target(1); }",
            "target",
            "function `target` expects 2 arguments, found 1",
        );
        rejects_source(
            "fn target(value: int) -> void {} fn main() -> void { target(true); }",
            "true",
            "cannot implicitly convert `bool` to `int`",
        );
        rejects_source(
            "fn target() -> void {} fn main(value: int) -> void {}",
            "value",
            "`main` must not have parameters",
        );
        rejects_source(
            "fn target() -> void {} fn main() -> int {}",
            "int",
            "`main` must return `void`",
        );
        rejects_source(
            "fn target() -> void {} fn main() -> void { const value = target; }",
            "target",
            "unknown binding `target`",
        );
    }

    #[test]
    fn parameters_are_shadowed_by_local_bindings_and_restored_after_their_scope() {
        let syntax = parse(
            "fn typed(value: u8) -> u8 {
                 { var value: bool = true; }
                 const copy: u8 = value;
                 return copy;
             }
             fn main() -> void {}",
        )
        .unwrap();
        let checked = check_root(&syntax).unwrap();
        let typed = syntax
            .functions
            .iter()
            .find(|(_, function)| !function.parameters.is_empty())
            .map(|(id, _)| id)
            .unwrap();
        let parameter = checked.functions[typed].parameters[0];
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| (binding.ty.clone(), binding.mutable))
                .collect::<Vec<_>>(),
            [
                (value_type(Scalar::U8), false),
                (value_type(Scalar::Bool), true),
                (value_type(Scalar::U8), false),
            ]
        );
        let initializer = match &syntax.statements[syntax.functions[typed].body[1]].kind {
            StatementKind::Binding { initializer, .. } => *initializer,
            _ => unreachable!("the shadowing scope ends before the copy"),
        };
        assert_eq!(
            checked.expressions[initializer].value,
            ExpressionValue::Reference(parameter)
        );
    }

    #[test]
    fn parameters_are_immutable_and_duplicate_names_are_rejected() {
        rejects_source(
            "fn target(value: int, value: bool) -> void {} fn main() -> void {}",
            "value",
            "duplicate parameter name `value`",
        );
        rejects_source(
            "fn target(value: int) -> void { value = 1; } fn main() -> void {}",
            "value",
            "cannot assign to immutable binding `value`",
        );
        rejects_source(
            "fn target(value: int) -> void { value += 1; } fn main() -> void {}",
            "value",
            "cannot assign to immutable binding `value`",
        );
    }

    #[test]
    fn calls_are_not_constant_expressions() {
        // Signatures resolve after module-level initializers, so a call is
        // reported where it is written rather than where the initializer ends.
        for (source, marked) in [
            (
                "fn value() -> int { return 1; } const result = value(); fn main() -> void {}",
                "value()",
            ),
            (
                "fn value() -> int { return 1; } const result = 1 + value(); fn main() -> void {}",
                "value()",
            ),
        ] {
            rejects_source(
                source,
                marked,
                "module-level initializer must be a constant expression",
            );
        }
    }

    #[test]
    fn boolean_bindings_assignments_and_constants_are_typed() {
        let syntax = parse(
            "const module_copy = module_ready;
             const module_ready: bool = true;
             fn main() -> void {
                 var ready: bool = false;
                 const copied = module_copy;
                 const negated = !copied;
                 ready = negated;
             }",
        )
        .unwrap();
        let checked = check_root(&syntax).unwrap();
        assert!(
            checked
                .bindings
                .iter()
                .all(|(_, binding)| binding.ty == value_type(Scalar::Bool))
        );
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.clone())
                .collect::<Vec<_>>(),
            [folded(1), folded(1), None, folded(1), folded(0)]
        );
        assert!(
            checked
                .expressions
                .iter()
                .all(|(_, expression)| expression.ty == value_type(Scalar::Bool)
                    && !expression.untyped)
        );
    }

    #[test]
    fn comparisons_follow_operand_types_and_fold_constants() {
        let syntax = parse(
            "fn main() -> void {
                const less = 1 < 2;
                const equal = true == false;
                const ordered = false < true;
                const chained = (1 < 2) == true;
                var value: u8 = 1;
                const runtime = value >= 0;
            }",
        )
        .unwrap();
        let checked = check_root(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| (binding.ty.clone(), binding.constant.clone()))
                .collect::<Vec<_>>(),
            [
                (value_type(Scalar::Bool), folded(1)),
                (value_type(Scalar::Bool), folded(0)),
                (value_type(Scalar::Bool), folded(1)),
                (value_type(Scalar::Bool), folded(1)),
                (value_type(Scalar::U8), None),
                (value_type(Scalar::Bool), None),
            ]
        );

        for (operator, expected) in [
            ("==", false),
            ("!=", true),
            ("<", true),
            ("<=", true),
            (">", false),
            (">=", false),
        ] {
            let syntax = parse(&format!(
                "fn main() -> void {{ const result = 1 {operator} 2; }}"
            ))
            .unwrap();
            let checked = check_root(&syntax).unwrap();
            assert_eq!(
                checked.bindings.iter().next().unwrap().1.constant,
                Some(Constant::Integer(BigInt::from(expected)))
            );
        }

        accepts("var value: u8 = 1; const result = value < 255;");
        rejects(
            "var value: u8 = 1; const result = value < 256;",
            "256",
            "integer literal out of range for `u8`",
        );
        rejects(
            "var left: u8 = 1; var right: u16 = 1; const result = left == right;",
            "==",
            "comparison operands have different types `u8` and `u16`",
        );
        rejects(
            "const result = 1 == true;",
            "==",
            "comparison operands have different types `int` and `bool`",
        );
    }

    #[test]
    fn logical_expressions_fold_constants_and_keep_runtime_short_circuiting() {
        let syntax = parse(
            "fn main() -> void {
                const conjunction = true && false;
                const disjunction = false || true;
                const negated = !false;
                var runtime = true;
                const guarded = false && runtime;
            }",
        )
        .unwrap();
        let checked = check_root(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.clone())
                .collect::<Vec<_>>(),
            [folded(0), folded(1), folded(1), None, None]
        );
        let guarded = match syntax.statements[syntax.functions[checked.main].body[4]].kind {
            StatementKind::Binding { initializer, .. } => initializer,
            _ => unreachable!(),
        };
        assert!(matches!(
            checked.expressions[guarded].value,
            ExpressionValue::Logical {
                operator: LogicalOperator::And,
                ..
            }
        ));
        assert_eq!(checked.expressions[guarded].constant, None);

        rejects(
            "const invalid = false && 1 / 0 == 0;",
            "/",
            "constant `/` divisor is zero",
        );
        rejects(
            "const invalid = true || u8(255) + 1 == 0;",
            "+",
            "constant `+` on `u8` would overflow",
        );
    }

    #[test]
    fn boolean_operators_and_contexts_reject_integer_mixing() {
        for (body, offending, message) in [
            (
                "const value: bool = 1;",
                "1",
                "cannot implicitly convert `int` to `bool`",
            ),
            (
                "const value: int = true;",
                "true",
                "cannot implicitly convert `bool` to `int`",
            ),
            (
                "const value = true + false;",
                "+",
                "integer `+` requires integer operands",
            ),
            (
                "const value = !1;",
                "1",
                "logical operand has type `int`, expected `bool`",
            ),
            (
                "const value = true && 1;",
                "1",
                "logical operand has type `int`, expected `bool`",
            ),
            (
                "const value = u8(true);",
                "u8(true)",
                "cannot convert `bool` to `u8`",
            ),
            (
                "exit(true);",
                "true",
                "cannot implicitly convert `bool` to `int`",
            ),
        ] {
            rejects(body, offending, message);
        }
    }

    #[test]
    fn structured_control_flow_checks_conditions_and_nested_scopes() {
        for body in ["if 1 {}", "for 1 {}"] {
            rejects(body, "1", "cannot implicitly convert `int` to `bool`");
        }

        accepts(
            "var outer = 0;
             if true { var branch = outer; }
             else if false { var branch = outer; }
             else { var branch = outer; }
             for {}
             for false {}
             for var i = outer; i < 3; i += 1 {
                 var i = i;
                 if i == 2 { continue; }
             }
             for outer = 0; outer < 1; outer = outer + 1 {}
             exit(outer);",
        );

        for body in [
            "if true { var hidden = 1; } exit(hidden);",
            "if true {} else { var hidden = 1; } exit(hidden);",
            "for { var hidden = 1; break; } exit(hidden);",
            "for var hidden = 0; hidden < 1; hidden += 1 {} exit(hidden);",
        ] {
            rejects(body, "hidden", "unknown binding `hidden`");
        }

        rejects(
            "for const iterator = 0; iterator < 1; iterator = 1 {}",
            "iterator",
            "cannot assign to immutable binding `iterator`",
        );
    }

    #[test]
    fn loop_control_resolves_enclosing_labels() {
        accepts(
            "var outer = 0;
             for :outer {
                 for :inner {
                     continue;
                     continue :outer;
                     break :inner;
                 }
             }
             for :same { break; }
             for :same { break :same; }",
        );

        for (body, offending, message) in [
            ("break;", "break", "`break` is not inside a loop"),
            (
                "continue :missing;",
                "continue",
                "`continue` is not inside a loop",
            ),
            (
                "for :outer { break :missing; }",
                "missing",
                "unknown enclosing loop label `missing`",
            ),
            (
                "for :same { for :same {} }",
                "same",
                "duplicate enclosing loop label `same`",
            ),
        ] {
            rejects(body, offending, message);
        }
    }

    #[test]
    fn compound_assignments_follow_binary_and_assignment_rules() {
        accepts(
            "var value: u8 = 1;
             value += 1;
             value +%= 255;
             value <<= u16(2);
             for var i: u8 = 0; i < 2; i += 1 {}",
        );
        for (body, offending, message) in [
            (
                "const value = 1; value += 1;",
                "value",
                "cannot assign to immutable binding `value`",
            ),
            (
                "var value = true; value += true;",
                "+=",
                "integer `+=` requires integer operands",
            ),
            (
                "var value: u8 = 1; value += u16(1);",
                "+=",
                "binary operands have different types `u8` and `u16`",
            ),
            (
                "var value = 1; value /= 0;",
                "/=",
                "constant `/=` divisor is zero",
            ),
            (
                "var value = 1; value <<= -1;",
                "<<=",
                "constant `<<=` shift count is negative",
            ),
        ] {
            rejects(body, offending, message);
        }
    }

    #[test]
    fn integer_expression_types_follow_operand_rules() {
        let body = "var a: u8 = 1;
             var b: u8 = 2;
             const add = a + 2;
             const reverse = 2 + a;
             const shift = a << u64(3);
             const exact: u16 = 1 + 2;
             const negative: i8 = -128;
             const complemented = ^a;
             const wrapped = a +% 1;
             const wrapped_negative = -%a;
             exit(0);
             const after = a & ^b;";
        let text = format!("fn main() -> void {{ {body} }}");
        let syntax = parse(&text).unwrap();
        let checked = check_root(&syntax).unwrap();
        let types: Vec<_> = checked
            .bindings
            .iter()
            .map(|(_, binding)| binding.ty.clone())
            .collect();
        assert_eq!(
            types,
            [
                Scalar::U8,
                Scalar::U8,
                Scalar::U8,
                Scalar::U8,
                Scalar::U8,
                Scalar::U16,
                Scalar::I8,
                Scalar::U8,
                Scalar::U8,
                Scalar::U8,
                Scalar::U8,
            ]
            .map(value_type)
        );

        for left in Scalar::ALL_INTEGERS {
            let left_name = left.name();
            for right in Scalar::ALL_INTEGERS {
                let right_name = right.name();
                let body = format!(
                    "var left: {left_name} = 1; var right: {right_name} = 1; const result = left + right;"
                );
                if left == right {
                    accepts(&body);
                } else {
                    rejects(
                        &body,
                        "+",
                        &format!(
                            "binary operands have different types `{left_name}` and `{right_name}`"
                        ),
                    );
                }

                accepts(&format!(
                    "var left: {left_name} = 1; var count: {right_name} = 1; const result = left << count;"
                ));
            }
        }

        for body in [
            "var x: u8 = 1; const y = x + 256;",
            "var x: u8 = 1; const y = 256 + x;",
        ] {
            rejects(body, "256", "integer literal out of range for `u8`");
        }
        for (body, offending, message) in [
            (
                "const x = 1 +% 2;",
                "+%",
                "wrapping arithmetic requires a typed operand",
            ),
            (
                "const x = -%1;",
                "-%",
                "wrapping negation requires a typed operand",
            ),
            (
                "const x = -u8(1);",
                "-",
                "unary `-` is not permitted on `u8`",
            ),
        ] {
            rejects(body, offending, message);
        }
        accepts("var x: u8 = 1; { var x: u16 = 2; const inner = x + 1; } x = x + 1; exit(0);");
        rejects(
            "exit(0); const after = 1 + missing;",
            "missing",
            "unknown binding `missing`",
        );
    }

    #[test]
    fn signed_minima_and_expression_constants_keep_their_contracts() {
        for (name, minimum) in [
            ("i8", 1u128 << 7),
            ("i16", 1u128 << 15),
            ("i32", 1u128 << 31),
            ("i64", 1u128 << 63),
            ("int", 1u128 << (usize::BITS - 1)),
        ] {
            accepts(&format!(
                "const direct: {name} = -{minimum}; const converted = {name}(-{minimum});"
            ));
            let invalid = minimum + 1;
            rejects(
                &format!("const value: {name} = -{invalid};"),
                &format!("-{invalid}"),
                &format!("integer value out of range for `{name}`"),
            );
        }

        let body = "const exact = 1 + 2;
             const copy = exact;
             var runtime = 1;
             const mixed = runtime + 2;
             { const exact = runtime; const shadowed = exact + 1; }
             exit(0);
             const after = copy ^ 1;";
        let text = format!("fn main() -> void {{ {body} }}");
        let syntax = parse(&text).unwrap();
        let checked = check_root(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.is_some())
                .collect::<Vec<_>>(),
            [true, true, false, false, false, false, true]
        );
    }

    #[test]
    fn constant_evaluation_preserves_exact_and_typed_operations() {
        let text = "fn main() -> void {
            const exact: u8 = (250 + 10) / 2;
            const ordinary = 9 * 5 - 3;
            const quotient = -7 / 3;
            const remainder = -7 % 3;
            const complement = ^0;
            const cleared = 15 & ^3;
            const bits = (12 & 10) ^ 3 | 16;
            const large: i64 = 1 << 40;
            const signed_shift = -8 >> 2;
            const wrapped_add = u8(250) +% 10;
            const wrapped_subtract = u8(1) -% 2;
            const wrapped_multiply = u8(200) *% 2;
            const wrapped_negate = -%u8(1);
            const high: u8 = 128;
            const discarded = high << 1;
            const negative: i8 = -1;
            const sign_fill = negative >> 8;
            const huge = u8.truncate((1 << 255) + 42);
            const distant_bit = u8.truncate(((1 << 1000000) * 2) >> 1000001);
            const copy = exact;
            const combined = copy + u8(1);
            var runtime = 2;
            const saved = runtime;
            const not_constant = saved + 1;
        }";
        let syntax = parse(text).unwrap();
        let checked = check_root(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.clone())
                .collect::<Vec<_>>(),
            [
                folded(130),
                folded(42),
                folded(-2),
                folded(-1),
                folded(-1),
                folded(12),
                folded(27),
                Some(Constant::Integer(BigInt::from(1u8) << 40)),
                folded(-2),
                folded(4),
                folded(255),
                folded(144),
                folded(255),
                folded(128),
                folded(0),
                folded(-1),
                folded(-1),
                folded(42),
                folded(1),
                folded(130),
                folded(131),
                None,
                None,
                None,
            ]
        );
    }

    #[test]
    fn constant_failures_are_diagnosed_before_runtime_lowering() {
        for (body, offending, message) in [
            (
                "const x = u8(255) + 1;",
                "+",
                "constant `+` on `u8` would overflow",
            ),
            (
                "const x = u8(0) - 1;",
                "-",
                "constant `-` on `u8` would overflow",
            ),
            (
                "const x = u8(128) * 2;",
                "*",
                "constant `*` on `u8` would overflow",
            ),
            (
                "const minimum: i8 = -128; const x = -minimum;",
                "-",
                "constant unary `-` on `i8` would trap",
            ),
            (
                "const x = i8(-128) / -1;",
                "/",
                "constant `/` on `i8` would trap",
            ),
            (
                "const x = i8(-128) % -1;",
                "%",
                "constant `%` on `i8` would trap",
            ),
            (
                "var x: u8 = 1; const y = x / 0;",
                "/",
                "constant `/` divisor is zero",
            ),
            (
                "var x: u8 = 1; const y = x % 0;",
                "%",
                "constant `%` divisor is zero",
            ),
            (
                "var x: u8 = 1; const y = x << -1;",
                "<<",
                "constant shift count is negative",
            ),
            (
                "const x: u8 = 250 + 10;",
                "250 + 10",
                "integer value out of range for `u8`",
            ),
            (
                "const x = u8(250 + 10);",
                "250 + 10",
                "integer value out of range for `u8`",
            ),
        ] {
            rejects(body, offending, message);
            rejects(&format!("exit(0); {body}"), offending, message);
        }
        rejects(
            "var runtime: u8 = 1; const outer = runtime + (u8(255) + 1);",
            "+",
            "constant `+` on `u8` would overflow",
        );
    }

    #[test]
    fn constant_classification_does_not_depend_on_binding_mutability() {
        let text = "fn main() -> void {
            const immutable = 1 + 2;
            var mutable = 1 + 2;
            const immutable_copy = immutable;
            const mutable_copy = mutable;
        }";
        let syntax = parse(text).unwrap();
        let checked = check_root(&syntax).unwrap();
        let expression_constants: Vec<_> = syntax.functions[checked.main]
            .body
            .iter()
            .map(|statement| match syntax.statements[*statement].kind {
                StatementKind::Binding { initializer, .. } => {
                    checked.expressions[initializer].constant.clone()
                }
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(
            expression_constants,
            [folded(3), folded(3), folded(3), None]
        );
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.clone())
                .collect::<Vec<_>>(),
            [folded(3), None, folded(3), None]
        );
    }

    #[test]
    fn precedence_wrapping_and_shift_boundaries_follow_the_integer_contract() {
        let text = "fn main() -> void {
            const precedence_left = 1 + 2 << 1;
            const precedence_right = 1 << 2 + 1;
            const same_level = 15 & ^3 & 6;
            const bit_levels = 1 | 2 ^ 3 & 4;
            const wrapped_left = u8(250) +% 10;
            const wrapped_right = 250 +% u8(10);
            const high: u8 = 128;
            const discarded = high << 1;
            var runtime_high: u8 = 128;
            const runtime_discarded = runtime_high << 1;
            const overshift = u8(1) << 999999999999999999999999999999999999;
            const negative: i8 = -1;
            const sign_fill = negative >> 8;
            const exact: u16 = 1 << 8;
            const reduced: u8 = 256 >> u8(8);
        }";
        let syntax = parse(text).unwrap();
        let checked = check_root(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.clone())
                .collect::<Vec<_>>(),
            [
                folded(5),
                folded(5),
                folded(4),
                folded(3),
                folded(4),
                folded(4),
                folded(128),
                folded(0),
                None,
                None,
                folded(0),
                folded(-1),
                folded(-1),
                folded(256),
                folded(1),
            ]
        );

        for (body, offending, message) in [
            (
                "const invalid: u8 = 250 +% 10;",
                "+%",
                "wrapping arithmetic requires a typed operand",
            ),
            (
                "const invalid = -%1;",
                "-%",
                "wrapping negation requires a typed operand",
            ),
            (
                "const typed = u8(1) +% 256;",
                "256",
                "integer literal out of range for `u8`",
            ),
            (
                "const invalid: u8 = 1 << 8;",
                "1 << 8",
                "integer value out of range for `u8`",
            ),
        ] {
            rejects(body, offending, message);
            rejects(&format!("exit(0); {body}"), offending, message);
        }
    }

    #[test]
    fn contextual_types_reach_nested_runtime_integer_expressions() {
        let text = "fn main() -> void {
            var count: uint = 1;
            const arithmetic: u64 = (1 << count) + 1;
            const negated: i64 = -(1 << count);
            const complemented: u16 = ^(1 << count);
        }";
        let syntax = parse(text).unwrap();
        let checked = check_root(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.ty.clone())
                .collect::<Vec<_>>(),
            [Scalar::Uint, Scalar::U64, Scalar::I64, Scalar::U16].map(value_type)
        );
        assert!(
            checked
                .expressions
                .iter()
                .all(|(_, expression)| !expression.untyped)
        );

        rejects(
            "var count: uint = 1; const invalid: u8 = -(1 << count);",
            "-",
            "unary `-` is not permitted on `u8`",
        );

        let too_large = BigInt::from(Scalar::Int.max()) + 1u8;
        rejects(
            &format!("var count: uint = 1; const invalid = u8.truncate({too_large} << count);"),
            &too_large.to_string(),
            "integer literal out of range for `int`",
        );
    }

    #[test]
    fn nonconstant_untyped_shift_counts_are_concretized_and_range_checked() {
        accepts("var value: u8 = 1; var n = 1; const shifted = value << ((1 << 2) << n);");

        for body in [
            "var value: u8 = 1; var n = 1; const shifted = value << ((1 << 200) << n);",
            "var value: u8 = 1; var n = 1; const shifted = value << ((1 << 63) << n);",
            "exit(0); var value: u8 = 1; var n = 1; const shifted = value << ((1 << 200) << n);",
        ] {
            let text = format!("fn main() -> void {{ {body} }}");
            let syntax = parse(&text).unwrap();
            let error = check_root(&syntax).unwrap_err();
            assert_eq!(error.message, "integer value out of range for `int`");
            assert!(text[error.span].contains("1 <<"));
        }
    }

    #[test]
    fn untyped_constant_folding_is_bounded_and_discards_child_values() {
        for body in [
            "const value = (1 << 1000000) * (1 << 1000000);",
            "exit(0); const value = (1 << 1000000) * (1 << 1000000);",
        ] {
            rejects(
                body,
                "*",
                "constant expression exceeds compiler resource limit",
            );
        }

        let syntax = parse("fn main() -> void { const value = (1 + 2) * (3 + 4); }").unwrap();
        let checked = check_root(&syntax).unwrap();
        let root = match syntax.statements[syntax.functions[checked.main].body[0]].kind {
            StatementKind::Binding { initializer, .. } => initializer,
            _ => unreachable!(),
        };
        assert_eq!(checked.expressions[root].constant, folded(21));
        assert!(
            checked
                .expressions
                .iter()
                .all(|(id, expression)| id == root || expression.constant.is_none())
        );
    }

    #[test]
    fn names_require_a_preceding_binding_even_after_exit() {
        for body in [
            "const x = x;",
            "exit(x); const x = 1;",
            "var y = x;",
            "exit(0); exit(x);",
            "const y = 1; exit(0); var x = x;",
        ] {
            // In the forward-reference case the later declaration has the same name.
            let text = format!("/* 🌿 */ fn main() -> void {{ {body} }}");
            let syntax = parse(&text).unwrap();
            let error = check_root(&syntax).unwrap_err();
            let start = if body.starts_with("exit(x)") {
                text.find("exit(x)").unwrap() + 5
            } else {
                text.rfind('x').unwrap()
            };
            assert_eq!(error.span, start..start + 1);
            assert_eq!(error.message, "unknown binding `x`");
        }
    }

    #[test]
    fn integer_contract_uses_contextual_literals_and_exact_references() {
        for ty in Scalar::ALL_INTEGERS {
            let name = ty.name();
            let max = u128::from(ty.max());
            for base in [2, 8, 10, 16] {
                let maximum = literal(max, base);
                let syntax = parse(&format!(
                    "fn main() -> void {{ var x: {name} = {maximum}; x = {maximum}; }}"
                ))
                .unwrap();
                let checked = check_root(&syntax).unwrap();
                assert!(
                    checked
                        .bindings
                        .iter()
                        .all(|(_, binding)| binding.ty == value_type(ty))
                );
                assert!(
                    checked
                        .expressions
                        .iter()
                        .all(|(_, expression)| expression.ty == value_type(ty))
                );

                let overflow = literal(max + 1, base);
                rejects(
                    &format!("exit(0); var x: {name} = {overflow};"),
                    &overflow,
                    &format!("integer literal out of range for `{name}`"),
                );
            }
        }

        for source in Scalar::ALL_INTEGERS {
            let source_name = source.name();
            for destination in Scalar::ALL_INTEGERS {
                let destination_name = destination.name();
                for target in [
                    format!("const target: {destination_name} = source;"),
                    format!("var target: {destination_name} = source;"),
                    format!("var target: {destination_name} = 0; target = source;"),
                ] {
                    for scope in [target.clone(), format!("{{ exit(0); {target} }}")] {
                        let body = format!("const source: {source_name} = 1; {scope}");
                        if source == destination {
                            let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
                            check_root(&syntax).unwrap();
                        } else {
                            rejects(
                                &body,
                                "source",
                                &format!(
                                    "cannot implicitly convert `{source_name}` to `{destination_name}`"
                                ),
                            );
                        }
                    }
                }
            }
            let body = format!("const source: {source_name} = 1; exit(source);");
            if source == Scalar::Int {
                let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
                check_root(&syntax).unwrap();
            } else {
                rejects(
                    &body,
                    "source",
                    &format!("cannot implicitly convert `{source_name}` to `int`"),
                );
            }
        }
    }

    #[test]
    fn conversions_evaluate_constants_and_preserve_runtime_operands() {
        let syntax = parse(
            "fn main() -> void {
                const literal: u8 = 42;
                const reduced = u8.truncate(340282366920938463463374607431768211498);
                const signed = i8.truncate(255);
                const widened = u64(literal);
                var runtime: u64 = 42;
                const checked = u8(runtime);
                const truncated = u8.truncate(runtime);
                const later = u16(checked);
            }",
        )
        .unwrap();
        let checked = check_root(&syntax).unwrap();
        let constants: Vec<_> = checked
            .bindings
            .iter()
            .map(|(_, binding)| binding.constant.clone())
            .collect();
        assert_eq!(
            constants,
            [
                folded(42),
                folded(42),
                folded(-1),
                folded(42),
                None,
                None,
                None,
                None
            ]
        );
        assert_eq!(
            checked
                .expressions
                .iter()
                .filter(|(_, expression)| {
                    matches!(expression.value, ExpressionValue::Conversion { .. })
                })
                .count(),
            3
        );

        rejects(
            "const value: u64 = 18446744073709551615; const narrowed = u8(value);",
            "u8(value)",
            "constant conversion to `u8` would trap",
        );
        rejects(
            "const narrowed = u8(256);",
            "256",
            "integer literal out of range for `u8`",
        );
        rejects(
            "exit(0); const value: u64 = 256; const narrowed = u8(value);",
            "u8(value)",
            "constant conversion to `u8` would trap",
        );
    }

    #[test]
    fn truncating_constants_keep_at_least_256_bits_before_reduction() {
        for literal in [
            format!("0x{}", "F".repeat(64)),
            format!("0b{}", "1".repeat(256)),
            format!("0x1{}FF", "0".repeat(128)),
        ] {
            let syntax = parse(&format!(
                "fn main() -> void {{ const result = int(u8.truncate({literal})); }}"
            ))
            .unwrap();
            let checked = check_root(&syntax).unwrap();
            assert_eq!(
                checked.bindings.iter().next().unwrap().1.constant,
                folded(255)
            );
            rejects(
                &format!("exit(0); const result = u8.truncate(u64({literal}));"),
                &literal,
                "integer literal out of range for `u64`",
            );
        }
    }

    #[test]
    fn constant_classification_follows_copies_and_shadowing() {
        let syntax = parse(
            "fn main() -> void {
                const original: u64 = 255;
                const copy = original;
                { var original = copy;
                  const saved = original;
                  const converted = u8(saved); }
                const folded = i8.truncate(copy);
                const extended = i64(folded);
                const wrapped = u64.truncate(extended);
            }",
        )
        .unwrap();
        let checked = check_root(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.clone())
                .collect::<Vec<_>>(),
            [
                folded(255),
                folded(255),
                None,
                None,
                None,
                folded(-1),
                folded(-1),
                Some(Constant::Integer(BigInt::from(u64::MAX)))
            ],
        );
        for body in [
            "const original: u64 = 256; const copy = original; const bad = u8(copy);",
            "const original: u64 = 256; { var original: u64 = 1; } const bad = u8(original);",
            "const negative = i8.truncate(255); const bad = u64(negative);",
        ] {
            let syntax = parse(&format!("fn main() -> void {{ exit(0); {body} }}")).unwrap();
            assert!(
                check_root(&syntax)
                    .unwrap_err()
                    .message
                    .contains("would trap")
            );
        }
    }

    #[test]
    fn native_integer_widths_follow_the_host_and_model_both_specified_widths() {
        assert_eq!(Scalar::Int.width(), usize::BITS);
        assert_eq!(Scalar::Uint.width(), usize::BITS);
        for pointer_width in [32, 64] {
            assert_eq!(Scalar::Int.width_on(pointer_width), pointer_width);
            assert_eq!(Scalar::Uint.width_on(pointer_width), pointer_width);
            assert_eq!(Scalar::I32.width_on(pointer_width), 32);
            assert_eq!(Scalar::U64.width_on(pointer_width), 64);
        }
    }

    #[test]
    fn module_bindings_resolve_forward_references_and_enclose_every_function() {
        let syntax = parse(
            "var counter = base;
             fn helper() -> void { const saved = counter; }
             const base: int = 40;
             fn main() -> void {
                 counter = counter + 2;
                 { const counter: u8 = 1; }
             }",
        )
        .unwrap();
        let checked = check_root(&syntax).unwrap();
        let module_statements: Vec<_> = syntax
            .files
            .iter()
            .flat_map(|file| &file.items)
            .filter_map(|item| match item {
                TopLevelItem::Binding { binding, .. } => Some(*binding),
                TopLevelItem::Function { .. } => None,
            })
            .collect();
        let counter = checked.declarations[module_statements[0]];
        let base = checked.declarations[module_statements[1]];
        assert!(checked.bindings[counter].mutable);
        assert_eq!(checked.bindings[counter].ty, value_type(Scalar::Int));
        assert_eq!(checked.bindings[counter].constant, None);
        assert_eq!(checked.bindings[base].constant, folded(40));
        assert!(
            checked
                .assignments
                .iter()
                .any(|(_, target)| target.binding == counter)
        );
        assert_eq!(checked.bindings.len(), 4);
    }

    #[test]
    fn module_binding_failures_identify_the_declaration_contract() {
        for (text, offending, message) in [
            (
                "var value = 1; const value = 2; fn main() -> void {}",
                "value = 2",
                "duplicate module-level name `value`",
            ),
            (
                "const value = value; fn main() -> void {}",
                "value;",
                "module-level initializer cycle involving `value`",
            ),
            (
                "const first = second; const second = third; const third = first; fn main() -> void {}",
                "first;",
                "module-level initializer cycle involving `first`",
            ),
            (
                "var runtime = 1; const invalid = runtime; fn main() -> void {}",
                "runtime;",
                "module-level initializer must be a constant expression",
            ),
            (
                "const value = missing; fn main() -> void {}",
                "missing",
                "unknown binding `missing`",
            ),
        ] {
            let syntax = parse(text).unwrap();
            let error = check_root(&syntax).unwrap_err();
            let start = text.rfind(offending).unwrap();
            let expected_len = offending
                .trim_end_matches(';')
                .split(' ')
                .next()
                .unwrap()
                .len();
            assert_eq!(error.span, start..start + expected_len, "{text}");
            assert_eq!(error.message, message, "{text}");
        }
    }

    #[test]
    fn module_dependency_ordering_does_not_recurse_on_the_compiler_stack() {
        let mut source = String::new();
        for index in 0..10_000 {
            source.push_str(&format!("const value{index} = value{};\n", index + 1));
        }
        source.push_str("const value10000 = 1; fn main() -> void {}");
        check_root(&parse(&source).unwrap()).unwrap();
    }

    #[test]
    fn entry_checks_keep_their_spans_and_function_bodies_are_all_checked() {
        for (text, span, message) in [
            ("/* 🌿 */", 0..0, "missing `main` function"),
            (
                "fn main() -> void {} fn main() -> void {}",
                24..28,
                "duplicate `main` function",
            ),
        ] {
            let syntax = parse(text).unwrap();
            let error = check_root(&syntax).unwrap_err();
            assert_eq!(error.span, span);
            assert_eq!(error.message, message);
        }
        for (text, name) in [
            (
                "fn helper() -> void {} fn helper() -> void {} fn main() -> void {}",
                "helper",
            ),
            ("const main = 1; fn main() -> void {}", "main"),
        ] {
            let syntax = parse(text).unwrap();
            let error = check_root(&syntax).unwrap_err();
            let start = text.rfind(name).unwrap();
            assert_eq!(error.span, start..start + name.len());
            assert_eq!(
                error.message,
                format!("duplicate module-level name `{name}`")
            );
        }
        let text = "fn helper() -> void { exit(missing); } fn main() -> void {}";
        let syntax = parse(text).unwrap();
        let error = check_root(&syntax).unwrap_err();
        let start = text.find("missing").unwrap();
        assert_eq!(error.span, start..start + "missing".len());
        assert_eq!(error.message, "unknown binding `missing`");
    }
}
