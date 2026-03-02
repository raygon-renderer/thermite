use syn::visit::Visit;

/// A visitor that searches an AST for a specific lifetime.
struct LifetimeFinder<'a> {
    target: &'a syn::Lifetime,
    found: bool,
}

impl<'a> LifetimeFinder<'a> {
    fn new(target: &'a syn::Lifetime) -> Self {
        Self { target, found: false }
    }
}

impl<'a, 'ast> Visit<'ast> for LifetimeFinder<'a> {
    fn visit_lifetime(&mut self, node: &'ast syn::Lifetime) {
        if self.found || node.ident == self.target.ident {
            self.found = true;
            return;
        }

        syn::visit::visit_lifetime(self, node);
    }
}

/// Determines if a lifetime parameter is late-bound based on syntactic usage.
pub fn is_late_bound<'a>(
    lf_param: &syn::LifetimeParam,
    where_clause: Option<&syn::WhereClause>,
    inputs: impl IntoIterator<Item = &'a syn::FnArg>,
) -> bool {
    // 1. Explicit bounds on the lifetime itself make it early-bound (e.g., <'a: 'b>)
    if !lf_param.bounds.is_empty() {
        return false;
    }

    let mut finder = LifetimeFinder::new(&lf_param.lifetime);

    // 2. Usage in a `where` clause elevates it to early-bound
    if let Some(wc) = where_clause {
        finder.visit_where_clause(wc);

        if finder.found {
            return false;
        }
    }

    // 3. If it is found in the input arguments (and wasn't bounded above), it is late-bound.
    for input in inputs {
        finder.visit_fn_arg(input);
    }

    finder.found
}
