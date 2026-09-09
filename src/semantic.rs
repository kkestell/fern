use crate::{
    diagnostic::Diagnostic,
    frontend::{
        BinaryOperator, Call, ComparisonOperator, Expression, ExpressionKind, ForHeader, Function,
        FunctionResult, LogicalOperator, PathComponent, QualifiedName, Statement, StatementKind,
        Syntax, TopLevelItem, TypeAnnotation, UnaryOperator, integer_parts,
    },
    module::Module,
    types::Type,
};
use la_arena::{Arena, ArenaMap, Idx};
use lasso::Spur;
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use std::collections::{HashMap, HashSet};

fn annotation_type(annotation: Option<&TypeAnnotation>) -> Option<Type> {
    annotation.map(|annotation| annotation.ty)
}

#[derive(Debug)]
pub(crate) struct Binding {
    pub ty: Type,
    pub mutable: bool,
    pub constant: Option<BigInt>,
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
}

#[derive(Debug)]
pub(crate) struct CheckedExpression {
    pub ty: Type,
    pub untyped: bool,
    pub value: ExpressionValue,
    pub constant: Option<BigInt>,
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

#[derive(Debug)]
pub(crate) struct CheckedProgram<'a> {
    pub syntax: &'a Syntax,
    pub main: Idx<Function>,
    pub module_bindings: Vec<Idx<Statement>>,
    pub expressions: ArenaMap<Idx<Expression>, CheckedExpression>,
    pub declarations: ArenaMap<Idx<Statement>, Idx<Binding>>,
    pub bindings: Arena<Binding>,
    pub assignments: ArenaMap<Idx<Statement>, Idx<Binding>>,
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
            let function = &syntax.functions[id];
            if syntax.names.resolve(&function.name) != "main" {
                continue;
            }
            if main.is_some() {
                return Err(Diagnostic::new(
                    function.name_span.clone(),
                    "duplicate `main` function",
                ));
            }
            if let Some(parameter) = function.parameters.first() {
                return Err(Diagnostic::new(
                    parameter.name_span.clone(),
                    "`main` must not have parameters",
                ));
            }
            if let FunctionResult::Value(annotation) = &function.result {
                return Err(Diagnostic::new(
                    annotation.span.clone(),
                    "`main` must return `void`",
                ));
            }
            main = Some(id);
        }
    }
    main.ok_or_else(|| Diagnostic::new(0..0, "missing `main` function"))
}

impl CheckedProgram<'_> {
    /// Checks one module: its module-level names, its signatures, each of its
    /// files' imports, its module-level initializers, and its function bodies.
    /// Appends the namespace that later modules import from.
    fn check_module(&mut self, module: &Module, imports: &[Vec<usize>]) -> Result<(), Diagnostic> {
        let syntax = self.syntax;
        let items: Vec<(usize, TopLevelItem)> = module
            .files
            .clone()
            .flat_map(|file| {
                syntax.files[file]
                    .items
                    .iter()
                    .map(move |item| (file, *item))
            })
            .collect();

        let mut module_names = HashSet::new();
        let mut module_declarations = HashMap::new();
        let mut module_statements = Vec::new();
        let mut statement_files = HashMap::new();
        self.function_names = HashMap::new();
        for &(file, item) in &items {
            match item {
                TopLevelItem::Function { function: id, .. } => {
                    let function = &syntax.functions[id];
                    if !module_names.insert(function.name) {
                        return Err(Diagnostic::new(
                            function.name_span.clone(),
                            format!(
                                "duplicate module-level name `{}`",
                                syntax.names.resolve(&function.name)
                            ),
                        ));
                    }
                    let mut parameter_names = HashSet::new();
                    for parameter in &function.parameters {
                        if !parameter_names.insert(parameter.name) {
                            return Err(Diagnostic::new(
                                parameter.name_span.clone(),
                                format!(
                                    "duplicate parameter name `{}`",
                                    syntax.names.resolve(&parameter.name)
                                ),
                            ));
                        }
                    }
                    self.function_names.insert(function.name, id);
                }
                TopLevelItem::Binding { binding: id, .. } => {
                    let StatementKind::Binding {
                        name, name_span, ..
                    } = &syntax.statements[id].kind
                    else {
                        unreachable!("frontend only permits bindings at module level")
                    };
                    if !module_names.insert(*name) {
                        return Err(Diagnostic::new(
                            name_span.clone(),
                            format!(
                                "duplicate module-level name `{}`",
                                syntax.names.resolve(name)
                            ),
                        ));
                    }
                    module_declarations.insert(*name, id);
                    module_statements.push(id);
                    statement_files.insert(id, file);
                }
            }
        }

        let mut module_scope = HashMap::new();
        let mut namespace = Namespace::new();
        for &(_, item) in &items {
            let TopLevelItem::Binding {
                binding: statement,
                public,
            } = item
            else {
                continue;
            };
            let StatementKind::Binding {
                name,
                mutable,
                annotation,
                ..
            } = &syntax.statements[statement].kind
            else {
                unreachable!("frontend only permits bindings at module level")
            };
            let binding = self.bindings.alloc(Binding {
                ty: annotation_type(annotation.as_ref()).unwrap_or(Type::Int),
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

        for &(_, item) in &items {
            let TopLevelItem::Function { function, public } = item else {
                continue;
            };
            let function_syntax = &syntax.functions[function];
            let parameters = function_syntax
                .parameters
                .iter()
                .map(|parameter| {
                    self.bindings.alloc(Binding {
                        ty: parameter.annotation.ty,
                        mutable: false,
                        constant: None,
                    })
                })
                .collect();
            let result = match &function_syntax.result {
                FunctionResult::Void => None,
                FunctionResult::Value(annotation) => Some(annotation.ty),
            };
            self.functions
                .insert(function, FunctionSignature { parameters, result });
            namespace.insert(
                function_syntax.name,
                Declaration {
                    public,
                    kind: DeclarationKind::Function(function),
                },
            );
        }

        for file in module.files.clone() {
            let mut file_imports = FileImports::default();
            for (index, import) in syntax.files[file].imports.iter().enumerate() {
                let resolved = imports[file][index];
                let path: Vec<&str> = import
                    .path
                    .iter()
                    .map(|component| syntax.names.resolve(&component.name))
                    .collect();
                let path = path.join("::");
                let target = &self.namespaces[resolved];
                match &import.selection {
                    None => {
                        let last = import.path.last().expect("an import path has components");
                        let imported = Imported::Module(resolved);
                        introduce(&mut file_imports, &namespace, last, imported, syntax)?;
                    }
                    Some(selection) => {
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
                            introduce(&mut file_imports, &namespace, component, imported, syntax)?;
                        }
                    }
                }
            }
            self.imports[file] = file_imports;
        }

        let mut dependencies = HashMap::new();
        for &statement in &module_statements {
            let StatementKind::Binding { initializer, .. } = &syntax.statements[statement].kind
            else {
                unreachable!("frontend only permits bindings at module level")
            };
            let mut references = Vec::new();
            collect_references(syntax, *initializer, &mut references);
            dependencies.insert(
                statement,
                references
                    .into_iter()
                    .filter_map(|(name, span)| {
                        module_declarations.get(&name).copied().map(|declaration| {
                            ModuleDependency {
                                declaration,
                                name,
                                span,
                            }
                        })
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let order = order_module_bindings(syntax, &module_statements, &dependencies)?;

        for &statement in &order {
            let StatementKind::Binding {
                mutable,
                annotation,
                initializer,
                ..
            } = &syntax.statements[statement].kind
            else {
                unreachable!("frontend only permits bindings at module level")
            };
            self.file = statement_files[&statement];
            let destination = annotation_type(annotation.as_ref());
            let expression = self.check_expression(
                *initializer,
                std::slice::from_ref(&module_scope),
                destination,
            )?;
            if expression.constant.is_none() {
                return Err(Diagnostic::new(
                    syntax.expressions[*initializer].span.clone(),
                    "module-level initializer must be a constant expression",
                ));
            }
            let binding = self.declarations[statement];
            self.bindings[binding].ty = expression.ty;
            self.bindings[binding].constant = if *mutable {
                None
            } else {
                expression.constant.clone()
            };
            self.expressions.insert(*initializer, expression);
        }
        self.module_bindings.extend(order);

        for &(file, item) in &items {
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
            let result = self.functions[function].result;
            let body = &syntax.functions[function].body;
            self.check_body(body, result, &mut scopes, &mut Vec::new())?;
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

        self.namespaces.push(namespace);
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

fn collect_references(
    syntax: &Syntax,
    expression: Idx<Expression>,
    references: &mut Vec<(Spur, std::ops::Range<usize>)>,
) {
    let expression = &syntax.expressions[expression];
    match &expression.kind {
        ExpressionKind::Integer(_) | ExpressionKind::Boolean(_) => {}
        ExpressionKind::Reference(name) if name.qualifier.is_none() => {
            references.push((name.name, expression.span.clone()));
        }
        ExpressionKind::Reference(_) => {}
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
        } => collect_references(syntax, *expression, references),
        ExpressionKind::Binary { left, right, .. }
        | ExpressionKind::Comparison { left, right, .. }
        | ExpressionKind::Logical { left, right, .. } => {
            collect_references(syntax, *left, references);
            collect_references(syntax, *right, references);
        }
        ExpressionKind::Call(call) => {
            for &argument in &call.arguments {
                collect_references(syntax, argument, references);
            }
        }
    }
}

impl CheckedProgram<'_> {
    fn check_body(
        &mut self,
        body: &[Idx<Statement>],
        result: Option<Type>,
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
        result: Option<Type>,
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
            } => {
                let destination = annotation_type(annotation.as_ref());
                let expression = self.check_expression(*initializer, scopes, destination)?;
                let ty = expression.ty;
                let constant = if *mutable {
                    None
                } else {
                    expression.constant.clone()
                };
                self.expressions.insert(*initializer, expression);
                let binding = self.bindings.alloc(Binding {
                    ty,
                    mutable: *mutable,
                    constant,
                });
                self.declarations.insert(statement, binding);
                scopes.last_mut().unwrap().insert(*name, binding);
            }
            StatementKind::Assignment { target, value } => {
                let binding = self.assignment_target(target, scopes)?;
                let expression =
                    self.check_expression(*value, scopes, Some(self.bindings[binding].ty))?;
                self.expressions.insert(*value, expression);
                self.assignments.insert(statement, binding);
            }
            StatementKind::CompoundAssignment {
                target,
                operator,
                operator_span,
                value,
            } => {
                let binding = self.assignment_target(target, scopes)?;
                let ty = self.bindings[binding].ty;
                let compound_operator = format!("{}=", operator.spelling());
                let left = CheckedExpression {
                    ty,
                    untyped: false,
                    value: ExpressionValue::Reference(binding),
                    constant: None,
                };
                let right = self.infer_expression(*value, scopes)?;
                let (_, right, _, _, _) = self.check_integer_binary(
                    *operator,
                    operator_span,
                    &compound_operator,
                    CheckedBinaryOperand {
                        id: None,
                        expression: left,
                    },
                    CheckedBinaryOperand {
                        id: Some(*value),
                        expression: right,
                    },
                )?;
                self.expressions.insert(*value, right.expression);
                self.assignments.insert(statement, binding);
            }
            StatementKind::Block { body } => self.check_body(body, result, scopes, loops)?,
            StatementKind::Exit { argument } => {
                let expression = self.check_expression(*argument, scopes, Some(Type::Int))?;
                self.expressions.insert(*argument, expression);
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
            }
            StatementKind::For {
                label,
                header,
                body,
            } => {
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
                let label_name = label.as_ref().map(|label| label.name);
                let has_header_scope = match header {
                    ForHeader::Infinite => false,
                    ForHeader::Condition(condition) => {
                        self.check_condition(*condition, scopes)?;
                        false
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
                        true
                    }
                };
                loops.push(label_name);
                self.check_body(body, result, scopes, loops)?;
                loops.pop();
                if has_header_scope {
                    scopes.pop();
                }
            }
            StatementKind::Break { label } | StatementKind::Continue { label } => {
                let keyword = if matches!(
                    &self.syntax.statements[statement].kind,
                    StatementKind::Break { .. }
                ) {
                    "break"
                } else {
                    "continue"
                };
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
            }
            StatementKind::Call { call } => {
                let (function, _) = self.check_call(call, scopes, false)?;
                self.calls.insert(statement, function);
            }
            StatementKind::Return { value } => match (result, value) {
                (None, None) => {}
                (None, Some(value)) => {
                    return Err(Diagnostic::new(
                        self.syntax.expressions[*value].span.clone(),
                        "cannot return a value from a `void` function",
                    ));
                }
                (Some(result), None) => {
                    return Err(Diagnostic::new(
                        self.syntax.statements[statement].span.clone(),
                        format!("`return` must supply a value of type `{}`", result.name()),
                    ));
                }
                (Some(result), Some(value)) => {
                    let expression = self.check_expression(*value, scopes, Some(result))?;
                    self.expressions.insert(*value, expression);
                }
            },
        }
        Ok(())
    }

    fn assignment_target(
        &mut self,
        target: &QualifiedName,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<Idx<Binding>, Diagnostic> {
        let binding = self.resolve(target, scopes)?;
        if self.bindings[binding].mutable {
            Ok(binding)
        } else {
            Err(Diagnostic::new(
                target.span.clone(),
                format!(
                    "cannot assign to immutable binding `{}`",
                    self.syntax.names.resolve(&target.name)
                ),
            ))
        }
    }

    fn check_condition(
        &mut self,
        condition: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<(), Diagnostic> {
        let expression = self.check_expression(condition, scopes, Some(Type::Bool))?;
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
                self.check_expression(argument, scopes, Some(self.bindings[parameter].ty))?;
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
        let mut checked = self.infer_expression(id, scopes)?;
        if checked.untyped {
            let destination = destination.unwrap_or(checked.ty);
            self.concretize(id, &mut checked, destination)?;
        } else if let Some(destination) = destination
            && checked.ty != destination
        {
            return Err(Diagnostic::new(
                self.syntax.expressions[id].span.clone(),
                format!(
                    "cannot implicitly convert `{}` to `{}`",
                    checked.ty.name(),
                    destination.name()
                ),
            ));
        }
        Ok(checked)
    }

    fn infer_expression(
        &mut self,
        id: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let expression = &self.syntax.expressions[id];
        let error = |message| Diagnostic::new(expression.span.clone(), message);
        let checked = match &expression.kind {
            ExpressionKind::Integer(spelling) => {
                let (base, digits, suffix) = integer_parts(spelling);
                debug_assert!(suffix.is_empty(), "frontend rejects literal suffixes");
                let value = BigUint::parse_bytes(digits.as_bytes(), base)
                    .expect("frontend validated integer digits");
                let value = BigInt::from(value);
                CheckedExpression {
                    ty: Type::Int,
                    untyped: true,
                    value: ExpressionValue::Integer,
                    constant: Some(value),
                }
            }
            ExpressionKind::Boolean(value) => CheckedExpression {
                ty: Type::Bool,
                untyped: true,
                value: ExpressionValue::Boolean,
                constant: Some(BigInt::from(*value)),
            },
            ExpressionKind::Reference(name) => {
                let binding = self.resolve(name, scopes)?;
                CheckedExpression {
                    ty: self.bindings[binding].ty,
                    untyped: false,
                    value: ExpressionValue::Reference(binding),
                    constant: self.bindings[binding].constant.clone(),
                }
            }
            ExpressionKind::Grouping { expression: inner } => {
                let mut checked = self.infer_expression(*inner, scopes)?;
                let result = CheckedExpression {
                    ty: checked.ty,
                    untyped: checked.untyped,
                    value: ExpressionValue::Grouping { expression: *inner },
                    constant: checked.constant.clone(),
                };
                if result.constant.is_some() {
                    checked.constant = None;
                }
                self.expressions.insert(*inner, checked);
                result
            }
            ExpressionKind::Unary {
                operator,
                operator_span,
                operand,
            } => {
                let operand_id = *operand;
                let mut checked_operand = self.infer_expression(operand_id, scopes)?;
                if !checked_operand.ty.is_integer() {
                    return Err(Diagnostic::new(
                        operator_span.clone(),
                        "integer unary operator requires an integer operand",
                    ));
                }
                if *operator == UnaryOperator::Negate
                    && !checked_operand.untyped
                    && !checked_operand.ty.signed()
                {
                    return Err(Diagnostic::new(
                        operator_span.clone(),
                        format!(
                            "unary `-` is not permitted on `{}`",
                            checked_operand.ty.name()
                        ),
                    ));
                }
                if *operator == UnaryOperator::WrappingNegate && checked_operand.untyped {
                    return Err(Diagnostic::new(
                        operator_span.clone(),
                        "wrapping negation requires a typed operand",
                    ));
                }
                let constant =
                    self.evaluate_unary(*operator, operator_span.clone(), &checked_operand)?;
                let result = CheckedExpression {
                    ty: checked_operand.ty,
                    untyped: checked_operand.untyped,
                    value: ExpressionValue::Unary {
                        operator: *operator,
                        operator_span: operator_span.clone(),
                        operand: operand_id,
                    },
                    constant,
                };
                if result.constant.is_some() {
                    checked_operand.constant = None;
                }
                self.expressions.insert(operand_id, checked_operand);
                result
            }
            ExpressionKind::Binary {
                operator,
                operator_span,
                left,
                right,
            } => {
                let left_id = *left;
                let right_id = *right;
                let checked_left = self.infer_expression(left_id, scopes)?;
                let checked_right = self.infer_expression(right_id, scopes)?;
                let (mut checked_left, mut checked_right, ty, untyped, constant) = self
                    .check_integer_binary(
                        *operator,
                        operator_span,
                        operator.spelling(),
                        CheckedBinaryOperand {
                            id: Some(left_id),
                            expression: checked_left,
                        },
                        CheckedBinaryOperand {
                            id: Some(right_id),
                            expression: checked_right,
                        },
                    )?;
                let result = CheckedExpression {
                    ty,
                    untyped,
                    value: ExpressionValue::Binary {
                        operator: *operator,
                        operator_span: operator_span.clone(),
                        left: left_id,
                        right: right_id,
                    },
                    constant,
                };
                if result.constant.is_some() {
                    checked_left.expression.constant = None;
                    checked_right.expression.constant = None;
                }
                self.expressions.insert(left_id, checked_left.expression);
                self.expressions.insert(right_id, checked_right.expression);
                result
            }
            ExpressionKind::Comparison {
                operator,
                operator_span,
                left,
                right,
            } => {
                let left_id = *left;
                let right_id = *right;
                let mut checked_left = self.infer_expression(left_id, scopes)?;
                let mut checked_right = self.infer_expression(right_id, scopes)?;
                match (checked_left.untyped, checked_right.untyped) {
                    (false, false) if checked_left.ty != checked_right.ty => {
                        return Err(Diagnostic::new(
                            operator_span.clone(),
                            format!(
                                "comparison operands have different types `{}` and `{}`",
                                checked_left.ty.name(),
                                checked_right.ty.name()
                            ),
                        ));
                    }
                    (false, true) => {
                        self.concretize(right_id, &mut checked_right, checked_left.ty)?;
                    }
                    (true, false) => {
                        self.concretize(left_id, &mut checked_left, checked_right.ty)?;
                    }
                    (true, true) if checked_left.ty != checked_right.ty => {
                        return Err(Diagnostic::new(
                            operator_span.clone(),
                            format!(
                                "comparison operands have different types `{}` and `{}`",
                                checked_left.ty.name(),
                                checked_right.ty.name()
                            ),
                        ));
                    }
                    _ => {}
                }
                let constant = match (
                    checked_left.constant.as_ref(),
                    checked_right.constant.as_ref(),
                ) {
                    (Some(left), Some(right)) => Some(BigInt::from(match operator {
                        ComparisonOperator::Equal => left == right,
                        ComparisonOperator::NotEqual => left != right,
                        ComparisonOperator::Less => left < right,
                        ComparisonOperator::LessEqual => left <= right,
                        ComparisonOperator::Greater => left > right,
                        ComparisonOperator::GreaterEqual => left >= right,
                    })),
                    _ => None,
                };
                let untyped = constant.is_some();
                let result = CheckedExpression {
                    ty: Type::Bool,
                    untyped,
                    value: ExpressionValue::Comparison {
                        operator: *operator,
                        operator_span: operator_span.clone(),
                        left: left_id,
                        right: right_id,
                    },
                    constant,
                };
                if result.constant.is_some() {
                    checked_left.constant = None;
                    checked_right.constant = None;
                }
                self.expressions.insert(left_id, checked_left);
                self.expressions.insert(right_id, checked_right);
                result
            }
            ExpressionKind::Logical {
                operator,
                operator_span,
                left,
                right,
            } => {
                let left_id = *left;
                let right_id = *right;
                let mut checked_left = self.infer_expression(left_id, scopes)?;
                let mut checked_right = self.infer_expression(right_id, scopes)?;
                require_boolean(self.syntax, left_id, &checked_left)?;
                require_boolean(self.syntax, right_id, &checked_right)?;
                if checked_left.untyped && !checked_right.untyped {
                    self.concretize(left_id, &mut checked_left, Type::Bool)?;
                } else if !checked_left.untyped && checked_right.untyped {
                    self.concretize(right_id, &mut checked_right, Type::Bool)?;
                }
                let constant = match (
                    checked_left.constant.as_ref(),
                    checked_right.constant.as_ref(),
                ) {
                    (Some(left), Some(right)) => Some(BigInt::from(match operator {
                        LogicalOperator::And => constant_boolean(left) && constant_boolean(right),
                        LogicalOperator::Or => constant_boolean(left) || constant_boolean(right),
                    })),
                    _ => None,
                };
                let untyped = checked_left.untyped && checked_right.untyped;
                let result = CheckedExpression {
                    ty: Type::Bool,
                    untyped,
                    value: ExpressionValue::Logical {
                        operator: *operator,
                        operator_span: operator_span.clone(),
                        left: left_id,
                        right: right_id,
                    },
                    constant,
                };
                if result.constant.is_some() {
                    checked_left.constant = None;
                    checked_right.constant = None;
                }
                self.expressions.insert(left_id, checked_left);
                self.expressions.insert(right_id, checked_right);
                result
            }
            ExpressionKind::LogicalNot {
                operator_span,
                operand,
            } => {
                let operand_id = *operand;
                let mut checked_operand = self.infer_expression(operand_id, scopes)?;
                require_boolean(self.syntax, operand_id, &checked_operand)?;
                let constant = checked_operand
                    .constant
                    .as_ref()
                    .map(|value| BigInt::from(!constant_boolean(value)));
                let result = CheckedExpression {
                    ty: Type::Bool,
                    untyped: checked_operand.untyped,
                    value: ExpressionValue::LogicalNot {
                        operator_span: operator_span.clone(),
                        operand: operand_id,
                    },
                    constant,
                };
                if result.constant.is_some() {
                    checked_operand.constant = None;
                }
                self.expressions.insert(operand_id, checked_operand);
                result
            }
            ExpressionKind::Conversion {
                destination: annotation,
                truncating,
                operand,
            } => {
                let destination = annotation.ty;
                let operand_id = *operand;
                let mut checked_operand = self.infer_expression(operand_id, scopes)?;
                if !checked_operand.ty.is_integer() {
                    return Err(error(format!(
                        "cannot convert `{}` to `{}`",
                        checked_operand.ty.name(),
                        destination.name()
                    )));
                }
                if checked_operand.untyped && (!*truncating || checked_operand.constant.is_none()) {
                    let operand_type = if *truncating { Type::Int } else { destination };
                    self.concretize(operand_id, &mut checked_operand, operand_type)?;
                }
                let constant = if let Some(value) = checked_operand.constant.as_ref() {
                    Some(if *truncating {
                        truncate_integer(value, destination)
                    } else if integer_fits(value, destination) {
                        value.clone()
                    } else {
                        return Err(error(format!(
                            "constant conversion to `{}` would trap",
                            destination.name()
                        )));
                    })
                } else {
                    None
                };
                let value = if constant.is_some() {
                    ExpressionValue::Integer
                } else {
                    ExpressionValue::Conversion {
                        operand: operand_id,
                        truncating: *truncating,
                    }
                };
                if constant.is_some() {
                    checked_operand.constant = None;
                }
                self.expressions.insert(operand_id, checked_operand);
                CheckedExpression {
                    ty: destination,
                    untyped: false,
                    value,
                    constant,
                }
            }
            ExpressionKind::Call(call) => {
                let (function, result) = self.check_call(call, scopes, true)?;
                CheckedExpression {
                    ty: result.expect("value-context call has a value result"),
                    untyped: false,
                    value: ExpressionValue::Call { function },
                    constant: None,
                }
            }
        };
        Ok(checked)
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
            Type,
            bool,
            Option<BigInt>,
        ),
        Diagnostic,
    > {
        if !left.expression.ty.is_integer() || !right.expression.ty.is_integer() {
            return Err(Diagnostic::new(
                operator_span.clone(),
                format!("integer `{spelling}` requires integer operands"),
            ));
        }
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
                (false, false) if left.expression.ty != right.expression.ty => {
                    return Err(Diagnostic::new(
                        operator_span.clone(),
                        format!(
                            "binary operands have different types `{}` and `{}`",
                            left.expression.ty.name(),
                            right.expression.ty.name()
                        ),
                    ));
                }
                (false, true) => self.concretize(
                    right
                        .id
                        .expect("an untyped right operand has an expression"),
                    &mut right.expression,
                    left.expression.ty,
                )?,
                (true, false) => self.concretize(
                    left.id.expect("an untyped left operand has an expression"),
                    &mut left.expression,
                    right.expression.ty,
                )?,
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
                Type::Int,
            )?;
        }
        let (ty, untyped) = if shift {
            (left.expression.ty, left.expression.untyped)
        } else if left.expression.untyped {
            (right.expression.ty, right.expression.untyped)
        } else {
            (left.expression.ty, false)
        };
        let constant = self.evaluate_binary(
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
        destination: Type,
    ) -> Result<(), Diagnostic> {
        debug_assert!(checked.untyped);
        if checked.ty.is_integer() != destination.is_integer() {
            return Err(Diagnostic::new(
                self.syntax.expressions[id].span.clone(),
                format!(
                    "cannot implicitly convert `{}` to `{}`",
                    checked.ty.name(),
                    destination.name()
                ),
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
        destination: Type,
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
        destination: Type,
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
                        format!("unary `-` is not permitted on `{}`", destination.name()),
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

    fn evaluate_unary(
        &self,
        operator: UnaryOperator,
        operator_span: std::ops::Range<usize>,
        operand: &CheckedExpression,
    ) -> Result<Option<BigInt>, Diagnostic> {
        let Some(value) = operand.constant.as_ref() else {
            return Ok(None);
        };
        let result = match operator {
            UnaryOperator::Negate => -value,
            UnaryOperator::WrappingNegate => truncate_integer(&-value, operand.ty),
            UnaryOperator::Complement if operand.untyped => !value,
            UnaryOperator::Complement => truncate_integer(&!value, operand.ty),
        };
        if operand.untyped && result.bits() > MAX_UNTYPED_INTEGER_BITS {
            return Err(Diagnostic::new(
                operator_span,
                "constant expression exceeds compiler resource limit",
            ));
        }
        if operator == UnaryOperator::Negate
            && !operand.untyped
            && !integer_fits(&result, operand.ty)
        {
            return Err(Diagnostic::new(
                operator_span,
                format!("constant unary `-` on `{}` would trap", operand.ty.name()),
            ));
        }
        Ok(Some(result))
    }

    fn evaluate_binary(
        &self,
        operator: BinaryOperator,
        operator_span: std::ops::Range<usize>,
        spelling: &str,
        ty: Type,
        untyped: bool,
        operands: (&CheckedExpression, &CheckedExpression),
    ) -> Result<Option<BigInt>, Diagnostic> {
        let (left, right) = operands;
        let right_constant = right.constant.as_ref();
        if matches!(operator, BinaryOperator::Divide | BinaryOperator::Remainder)
            && right_constant == Some(&BigInt::from(0u8))
        {
            return Err(Diagnostic::new(
                operator_span,
                format!("constant `{spelling}` divisor is zero"),
            ));
        }
        if matches!(
            operator,
            BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight
        ) && right_constant.is_some_and(|count| count < &BigInt::from(0u8))
        {
            let message = if spelling.ends_with('=') {
                format!("constant `{spelling}` shift count is negative")
            } else {
                "constant shift count is negative".to_owned()
            };
            return Err(Diagnostic::new(operator_span, message));
        }

        let (Some(left), Some(right)) = (left.constant.as_ref(), right_constant) else {
            return Ok(None);
        };
        if untyped
            && operator == BinaryOperator::Multiply
            && left.bits().saturating_add(right.bits()).saturating_sub(1) > MAX_UNTYPED_INTEGER_BITS
        {
            return Err(Diagnostic::new(
                operator_span,
                "constant expression exceeds compiler resource limit",
            ));
        }
        let result = match operator {
            BinaryOperator::Multiply => left * right,
            BinaryOperator::Divide => {
                if !untyped && is_minimum(left, ty) && right == &BigInt::from(-1) {
                    return Err(Diagnostic::new(
                        operator_span,
                        format!("constant `/` on `{}` would trap", ty.name()),
                    ));
                }
                left / right
            }
            BinaryOperator::Remainder => {
                if !untyped && is_minimum(left, ty) && right == &BigInt::from(-1) {
                    return Err(Diagnostic::new(
                        operator_span,
                        format!("constant `%` on `{}` would trap", ty.name()),
                    ));
                }
                left % right
            }
            BinaryOperator::WrappingMultiply => truncate_integer(&(left * right), ty),
            BinaryOperator::Add => left + right,
            BinaryOperator::Subtract => left - right,
            BinaryOperator::WrappingAdd => truncate_integer(&(left + right), ty),
            BinaryOperator::WrappingSubtract => truncate_integer(&(left - right), ty),
            BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => {
                return self.evaluate_shift(operator, operator_span, ty, untyped, left, right);
            }
            BinaryOperator::And => left & right,
            BinaryOperator::Xor => left ^ right,
            BinaryOperator::Or => left | right,
        };
        let checked_arithmetic = matches!(
            operator,
            BinaryOperator::Multiply | BinaryOperator::Add | BinaryOperator::Subtract
        );
        if checked_arithmetic && !untyped && !integer_fits(&result, ty) {
            return Err(Diagnostic::new(
                operator_span,
                format!(
                    "constant `{}` on `{}` would overflow",
                    operator.spelling(),
                    ty.name()
                ),
            ));
        }
        if untyped && result.bits() > MAX_UNTYPED_INTEGER_BITS {
            return Err(Diagnostic::new(
                operator_span,
                "constant expression exceeds compiler resource limit",
            ));
        }
        Ok(Some(result))
    }

    fn evaluate_shift(
        &self,
        operator: BinaryOperator,
        operator_span: std::ops::Range<usize>,
        ty: Type,
        untyped: bool,
        left: &BigInt,
        right: &BigInt,
    ) -> Result<Option<BigInt>, Diagnostic> {
        debug_assert!(right >= &BigInt::from(0u8));
        if untyped {
            const MAX_CONSTANT_SHIFT: usize = 1_000_000;
            let count = right.to_usize();
            if operator == BinaryOperator::ShiftRight && count.is_none() {
                return Ok(Some(if left < &BigInt::from(0u8) {
                    BigInt::from(-1)
                } else {
                    BigInt::from(0u8)
                }));
            }
            let Some(count) = count else {
                return Err(Diagnostic::new(
                    operator_span,
                    "constant shift exceeds compiler resource limit",
                ));
            };
            if operator == BinaryOperator::ShiftLeft
                && (count > MAX_CONSTANT_SHIFT
                    || count
                        .try_into()
                        .unwrap_or(u64::MAX)
                        .saturating_add(left.bits())
                        > MAX_UNTYPED_INTEGER_BITS)
            {
                return Err(Diagnostic::new(
                    operator_span,
                    "constant expression exceeds compiler resource limit",
                ));
            }
            return Ok(Some(if operator == BinaryOperator::ShiftLeft {
                left << count
            } else {
                left >> count
            }));
        }

        if right >= &BigInt::from(ty.width()) {
            return Ok(Some(
                if operator == BinaryOperator::ShiftRight
                    && ty.signed()
                    && left < &BigInt::from(0u8)
                {
                    BigInt::from(-1)
                } else {
                    BigInt::from(0u8)
                },
            ));
        }
        let count = right
            .to_usize()
            .expect("count below every Fern integer width fits usize");
        Ok(Some(if operator == BinaryOperator::ShiftLeft {
            truncate_integer(&(left << count), ty)
        } else {
            left >> count
        }))
    }
}

const MAX_UNTYPED_INTEGER_BITS: u64 = 2_000_000;

fn concretize_value(
    syntax: &Syntax,
    id: Idx<Expression>,
    checked: &mut CheckedExpression,
    destination: Type,
) -> Result<Option<bool>, Diagnostic> {
    if !checked.untyped {
        return Ok(None);
    }
    if checked.ty.is_integer()
        && checked
            .constant
            .as_ref()
            .is_some_and(|value| !integer_fits(value, destination))
    {
        return Err(out_of_range(syntax, id, destination));
    }
    let constant = checked.constant.is_some();
    checked.ty = destination;
    checked.untyped = false;
    Ok(Some(constant))
}

fn require_boolean(
    syntax: &Syntax,
    id: Idx<Expression>,
    checked: &CheckedExpression,
) -> Result<(), Diagnostic> {
    if checked.ty == Type::Bool {
        Ok(())
    } else {
        Err(Diagnostic::new(
            syntax.expressions[id].span.clone(),
            format!(
                "logical operand has type `{}`, expected `bool`",
                checked.ty.name()
            ),
        ))
    }
}

fn constant_boolean(value: &BigInt) -> bool {
    debug_assert!(value == &BigInt::from(0u8) || value == &BigInt::from(1u8));
    value == &BigInt::from(1u8)
}

fn out_of_range(syntax: &Syntax, id: Idx<Expression>, destination: Type) -> Diagnostic {
    let literal = matches!(syntax.expressions[id].kind, ExpressionKind::Integer(_));
    Diagnostic::new(
        syntax.expressions[id].span.clone(),
        format!(
            "integer {} out of range for `{}`",
            if literal { "literal" } else { "value" },
            destination.name()
        ),
    )
}

fn is_minimum(value: &BigInt, ty: Type) -> bool {
    value == &BigInt::from(ty.min())
}

fn integer_fits(value: &BigInt, ty: Type) -> bool {
    value >= &BigInt::from(ty.min()) && value <= &BigInt::from(ty.max())
}

fn integer_from_bits(bits: BigInt, ty: Type) -> BigInt {
    if ty.signed() && bits >= (BigInt::from(1u8) << (ty.width() - 1)) {
        bits - (BigInt::from(1u8) << ty.width())
    } else {
        bits
    }
}

fn truncate_integer(value: &BigInt, ty: Type) -> BigInt {
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
            assert_eq!(checked.bindings[*id].ty, Type::Int);
        }
        let facts: Vec<_> = syntax
            .expressions
            .iter()
            .map(|(id, _)| &checked.expressions[id])
            .collect();
        assert_eq!(facts.len(), 5);
        assert!(facts.iter().all(|fact| fact.ty == Type::Int));
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
        let targets: Vec<_> = checked.assignments.iter().map(|(_, id)| *id).collect();
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
                    checked.expressions[*value].ty,
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            returned,
            [
                ("200", Type::U8),
                ("1 + 2", Type::I64),
                ("1 < 2", Type::Bool),
                ("1", Type::Int),
                ("2", Type::Int),
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
                .map(|(_, binding)| (binding.ty, binding.mutable))
                .collect::<Vec<_>>(),
            [
                (Type::U8, false),
                (Type::Bool, false),
                (Type::U8, false),
                (Type::Bool, false),
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
                .map(|(_, binding)| (binding.ty, binding.mutable))
                .collect::<Vec<_>>(),
            [(Type::U8, false), (Type::Bool, true), (Type::U8, false)]
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
        rejects_source(
            "fn value() -> int { return 1; } const result = value(); fn main() -> void {}",
            "value()",
            "module-level initializer must be a constant expression",
        );
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
                .all(|(_, binding)| binding.ty == Type::Bool)
        );
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.clone())
                .collect::<Vec<_>>(),
            [Some(big(1)), Some(big(1)), None, Some(big(1)), Some(big(0))]
        );
        assert!(
            checked
                .expressions
                .iter()
                .all(|(_, expression)| expression.ty == Type::Bool && !expression.untyped)
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
                .map(|(_, binding)| (binding.ty, binding.constant.clone()))
                .collect::<Vec<_>>(),
            [
                (Type::Bool, Some(big(1))),
                (Type::Bool, Some(big(0))),
                (Type::Bool, Some(big(1))),
                (Type::Bool, Some(big(1))),
                (Type::U8, None),
                (Type::Bool, None),
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
                Some(BigInt::from(expected))
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
            [Some(big(0)), Some(big(1)), Some(big(1)), None, None]
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
            .map(|(_, binding)| binding.ty)
            .collect();
        assert_eq!(
            types,
            [
                Type::U8,
                Type::U8,
                Type::U8,
                Type::U8,
                Type::U8,
                Type::U16,
                Type::I8,
                Type::U8,
                Type::U8,
                Type::U8,
                Type::U8,
            ]
        );

        for left in Type::ALL_INTEGERS {
            let left_name = left.name();
            for right in Type::ALL_INTEGERS {
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
                Some(big(130)),
                Some(big(42)),
                Some(big(-2)),
                Some(big(-1)),
                Some(big(-1)),
                Some(big(12)),
                Some(big(27)),
                Some(BigInt::from(1u8) << 40),
                Some(big(-2)),
                Some(big(4)),
                Some(big(255)),
                Some(big(144)),
                Some(big(255)),
                Some(big(128)),
                Some(big(0)),
                Some(big(-1)),
                Some(big(-1)),
                Some(big(42)),
                Some(big(1)),
                Some(big(130)),
                Some(big(131)),
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
            [Some(big(3)), Some(big(3)), Some(big(3)), None]
        );
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.clone())
                .collect::<Vec<_>>(),
            [Some(big(3)), None, Some(big(3)), None]
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
                Some(big(5)),
                Some(big(5)),
                Some(big(4)),
                Some(big(3)),
                Some(big(4)),
                Some(big(4)),
                Some(big(128)),
                Some(big(0)),
                None,
                None,
                Some(big(0)),
                Some(big(-1)),
                Some(big(-1)),
                Some(big(256)),
                Some(big(1)),
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
                .map(|(_, binding)| binding.ty)
                .collect::<Vec<_>>(),
            [Type::Uint, Type::U64, Type::I64, Type::U16]
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

        let too_large = BigInt::from(Type::Int.max()) + 1u8;
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
        assert_eq!(checked.expressions[root].constant, Some(big(21)));
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
        for ty in Type::ALL_INTEGERS {
            let name = ty.name();
            let max = u128::from(ty.max());
            for base in [2, 8, 10, 16] {
                let maximum = literal(max, base);
                let syntax = parse(&format!(
                    "fn main() -> void {{ var x: {name} = {maximum}; x = {maximum}; }}"
                ))
                .unwrap();
                let checked = check_root(&syntax).unwrap();
                assert!(checked.bindings.iter().all(|(_, binding)| binding.ty == ty));
                assert!(
                    checked
                        .expressions
                        .iter()
                        .all(|(_, expression)| expression.ty == ty)
                );

                let overflow = literal(max + 1, base);
                rejects(
                    &format!("exit(0); var x: {name} = {overflow};"),
                    &overflow,
                    &format!("integer literal out of range for `{name}`"),
                );
            }
        }

        for source in Type::ALL_INTEGERS {
            let source_name = source.name();
            for destination in Type::ALL_INTEGERS {
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
            if source == Type::Int {
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
                Some(big(42)),
                Some(big(42)),
                Some(big(-1)),
                Some(big(42)),
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
                Some(big(255))
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
                Some(big(255)),
                Some(big(255)),
                None,
                None,
                None,
                Some(big(-1)),
                Some(big(-1)),
                Some(BigInt::from(u64::MAX))
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
        assert_eq!(Type::Int.width(), usize::BITS);
        assert_eq!(Type::Uint.width(), usize::BITS);
        for pointer_width in [32, 64] {
            assert_eq!(Type::Int.width_on(pointer_width), pointer_width);
            assert_eq!(Type::Uint.width_on(pointer_width), pointer_width);
            assert_eq!(Type::I32.width_on(pointer_width), 32);
            assert_eq!(Type::U64.width_on(pointer_width), 64);
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
        assert_eq!(checked.bindings[counter].ty, Type::Int);
        assert_eq!(checked.bindings[counter].constant, None);
        assert_eq!(checked.bindings[base].constant, Some(big(40)));
        assert!(
            checked
                .assignments
                .iter()
                .any(|(_, target)| *target == counter)
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
