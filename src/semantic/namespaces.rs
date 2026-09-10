//! Module namespaces, imports, declaration ordering, and entry selection.

use crate::{
    diagnostic::Diagnostic,
    frontend::syntax::{
        AnnotationKind, Call, Expression, ExpressionKind, Function, FunctionResult, Import,
        PathComponent, QualifiedName, Statement, StatementKind, Syntax, TopLevelItem,
        TypeAnnotation, find_call, walk_expression,
    },
    module::{File, Module, ModuleId},
    source::FileId,
    types::Scalar,
};
use la_arena::{Arena, ArenaMap, Idx};
use lasso::Spur;
use std::collections::{HashMap, HashSet};

use super::{model::*, statements::body_terminates};
/// A module-level declaration, as another file or module sees it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Declaration {
    pub(super) public: bool,
    pub(super) kind: DeclarationKind,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum DeclarationKind {
    Binding(Idx<Binding>),
    Function(Idx<Function>),
}

/// One module's declarations by name.
pub(super) type Namespace = HashMap<Spur, Declaration>;

/// What a name introduced by a `use` declaration refers to. A whole-module
/// import holds the module's index into `CheckedProgram::namespaces` rather
/// than a copy of its namespace.
#[derive(Debug, Clone, Copy)]
pub(super) enum Imported {
    Module(ModuleId),
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
    pub(super) fn unknown(self) -> &'static str {
        match self {
            Self::Function => "unknown function",
            Self::Value => "unknown binding",
        }
    }
}

/// One file's imported names. `introduced` keeps declaration order and spans,
/// so an unreferenced import reports the first one.
#[derive(Debug, Default)]
pub(super) struct FileImports {
    pub(super) names: HashMap<Spur, Imported>,
    introduced: Vec<(Spur, std::ops::Range<usize>)>,
    pub(super) used: HashSet<Spur>,
}

/// The module-level bindings of one module, in source order.
#[derive(Default)]
struct ModuleBindings {
    /// The statement declaring each binding, by name.
    pub(super) declarations: HashMap<Spur, Idx<Statement>>,
    pub(super) statements: Vec<Idx<Statement>>,
    /// The file each binding is declared in.
    pub(super) files: HashMap<Idx<Statement>, FileId>,
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
    files: &[File],
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
        namespaces: HashMap::new(),
        imports: HashMap::new(),
        file: FileId(0),
    };
    // Dependencies come before dependents, so an imported module's namespace is
    // built and its bindings are ordered before any module that imports it.
    for (index, module) in modules.iter().enumerate() {
        checked.check_module(ModuleId(index), module, files)?;
    }
    Ok(checked)
}

/// Checks `syntax` as one root module with no imports.
#[cfg(test)]
pub(crate) fn check_root(syntax: &Syntax) -> Result<CheckedProgram<'_>, Diagnostic> {
    let (modules, files) = crate::module::single(syntax);
    check(syntax, &modules, &files)
}

/// The root module's sole `main`, with the entry-point signature. A `main` in a
/// dependency module is an ordinary function.
fn entry_point(syntax: &Syntax, root: &Module) -> Result<Idx<Function>, Diagnostic> {
    let mut main = None;
    for file in &root.files {
        for item in &syntax.files[file.0].items {
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
    fn check_module(
        &mut self,
        id: ModuleId,
        module: &Module,
        files: &[File],
    ) -> Result<(), Diagnostic> {
        let items = self.module_items(module);
        let bindings = self.declare_module_names(&items)?;
        let mut namespace = Namespace::new();
        let module_scope = self.declare_module_bindings(&items, &mut namespace)?;
        self.declare_function_names(&items, &mut namespace);
        self.resolve_imports(module, files, &namespace)?;
        self.check_module_initializers(&bindings, &module_scope)?;
        self.resolve_signatures(&items, &module_scope)?;
        self.check_function_bodies(&items, &module_scope)?;
        self.check_imports_used(module)?;
        self.namespaces.insert(id, namespace);
        Ok(())
    }

    /// Every top-level item of the module, paired with the file it comes from.
    fn module_items(&self, module: &Module) -> Vec<(FileId, TopLevelItem)> {
        let syntax = self.syntax;
        module
            .files
            .iter()
            .copied()
            .flat_map(|file| {
                syntax.files[file.0]
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
        items: &[(FileId, TopLevelItem)],
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
        items: &[(FileId, TopLevelItem)],
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
        items: &[(FileId, TopLevelItem)],
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
        items: &[(FileId, TopLevelItem)],
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

    fn resolve_imports(
        &mut self,
        module: &Module,
        files: &[File],
        namespace: &Namespace,
    ) -> Result<(), Diagnostic> {
        let syntax = self.syntax;
        for file in module.files.iter().copied() {
            let mut file_imports = FileImports::default();
            for (index, import) in syntax.files[file.0].imports.iter().enumerate() {
                self.resolve_import(
                    import,
                    files[file.0].imports[index],
                    namespace,
                    &mut file_imports,
                )?;
            }
            self.imports.insert(file, file_imports);
        }
        Ok(())
    }

    /// Introduces the names one `use` declaration brings into its file, either
    /// the module itself or the declarations it selects.
    fn resolve_import(
        &self,
        import: &Import,
        resolved: ModuleId,
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
        let target = self
            .namespaces
            .get(&resolved)
            .expect("imports resolve only completed modules");
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
        items: &[(FileId, TopLevelItem)],
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
        for file in &module.files {
            let file_imports = self
                .imports
                .get(file)
                .expect("every module file has resolved imports");
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

impl CheckedProgram<'_> {
    pub(super) fn resolve_call(
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

    pub(super) fn resolve(
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
        match self
            .imports
            .get(&self.file)
            .expect("checking selects a file with resolved imports")
            .names
            .get(&name.name)
        {
            Some(&Imported::Declaration(kind)) => {
                self.imports
                    .get_mut(&self.file)
                    .expect("checking selects a file with resolved imports")
                    .used
                    .insert(name.name);
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
        let module = match self
            .imports
            .get(&self.file)
            .expect("checking selects a file with resolved imports")
            .names
            .get(&qualifier.name)
        {
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
        self.imports
            .get_mut(&self.file)
            .expect("checking selects a file with resolved imports")
            .used
            .insert(qualifier.name);
        let Some(declaration) = self.namespaces[&module].get(&name.name).copied() else {
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
}
