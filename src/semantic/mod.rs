mod annotations;
mod constants;
mod expressions;
pub(crate) mod model;
pub(crate) mod namespaces;
mod statements;

pub(crate) use annotations::reject_heap_syntax;
#[cfg(test)]
mod tests;
