//! A small Hindley–Milner-style inference experiment with *equality-only*
//! constructor pattern matching. No unions, intersections, negation,
//! subtraction, semantic subtyping, or tallying.
//!
//! IMPORTANT: `Variant(tag, payload)` here denotes a distinct structural
//! constructor type. Consequently A(t) and B(u) do not unify. To support
//! several constructors of one datatype in a match, introduce a nominal ADT
//! (e.g. Option(t)) and constructor signatures, as in ML/Haskell.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_TYPE_VAR: AtomicUsize = AtomicUsize::new(0);

type TypePtr = Rc<RefCell<Type>>;
type Id = String;

#[derive(Debug, Clone, PartialEq)]
struct TypeVar {
    name: String,
}

#[derive(Debug, Clone, PartialEq)]
struct TypeCon {
    name: String,
    args: Vec<TypePtr>,
}

#[derive(Debug, Clone, PartialEq)]
struct Scheme {
    for_all: Vec<TypePtr>,
    ty: TypePtr,
}

#[derive(Debug, Clone, PartialEq)]
enum Type {
    Var(TypeVar),
    Con(TypeCon),
    Variant(String, TypePtr),
    Scheme(Scheme),
}

impl Into<TypePtr> for Type {
    fn into(self) -> TypePtr {
        Rc::new(RefCell::new(self))
    }
}

impl Type {
    fn var(name: &str) -> TypePtr {
        Type::Var(TypeVar { name: name.into() }).into()
    }

    fn con(name: &str) -> TypePtr {
        Self::con_with_args(name, vec![])
    }

    fn con_with_args(name: &str, args: Vec<TypePtr>) -> TypePtr {
        Type::Con(TypeCon { name: name.into(), args }).into()
    }

    fn variant(name: &str, payload: TypePtr) -> TypePtr {
        Type::Variant(name.into(), payload).into()
    }

    fn scheme(for_all: Vec<TypePtr>, ty: TypePtr) -> TypePtr {
        Type::Scheme(Scheme { for_all, ty }).into()
    }

    /// Apply all known substitutions, recursively, without changing the AST.
    fn find(&self, env: &TypeEnv) -> TypePtr {
        match self {
            Type::Var(var) => match env.get_substitution(var) {
                Some(t) => t.borrow().find(env),
                None => self.clone().into(),
            },
            Type::Con(con) => Type::con_with_args(
                &con.name,
                con.args.iter().map(|a| a.borrow().find(env)).collect(),
            ),
            Type::Variant(tag, payload) => Type::variant(tag, payload.borrow().find(env)),
            Type::Scheme(scheme) => Type::scheme(
                scheme.for_all.clone(),
                scheme.ty.borrow().find(env),
            ),
        }
    }
}

// Preserved from the previous HM prototype; these are deliberately NOT
// involved in pattern matching. They are still placeholders for a proper
// scheme implementation (generalization over free variables and fresh
// instantiation of quantified variables).
#[allow(dead_code)]
fn instanciate(ty: &TypePtr) -> Type {
    let inner = ty.borrow();
    match &*inner {
        Type::Scheme(s) => s.ty.borrow().clone(),
        _ => inner.clone(),
    }
}

#[allow(dead_code)]
fn generalize(ty: &TypePtr) -> Type {
    let inner = ty.borrow();
    match &*inner {
        Type::Var(_) => Type::Scheme(Scheme {
            for_all: vec![ty.clone()],
            ty: ty.clone(),
        }),
        Type::Con(c) => Type::Scheme(Scheme {
            for_all: c.args.clone(),
            ty: ty.clone(),
        }),
        _ => inner.clone(),
    }
}

#[derive(Debug, Clone)]
enum Constraint {
    Equals(TypePtr, TypePtr),
}

#[derive(Debug, Default, Clone)]
struct TypeEnv {
    variables: HashMap<Id, TypePtr>,
    substitutions: HashMap<String, TypePtr>,
    constraints: Vec<Constraint>,
}

impl TypeEnv {
    fn get(&self, name: &str) -> Option<&TypePtr> {
        self.variables.get(name)
    }

    fn insert(&mut self, name: String, ty: TypePtr) -> Option<TypePtr> {
        self.variables.insert(name, ty)
    }

    fn substitute(&mut self, var: &TypeVar, ty: TypePtr) {
        self.substitutions.insert(var.name.clone(), ty);
    }

    fn get_substitution(&self, var: &TypeVar) -> Option<TypePtr> {
        self.substitutions.get(&var.name).cloned()
    }

    fn fresh(&self) -> TypePtr {
        let id = NEXT_TYPE_VAR.fetch_add(1, Ordering::Relaxed);
        Type::var(&format!("t{id}"))
    }

    fn equal(&mut self, a: TypePtr, b: TypePtr) {
        self.constraints.push(Constraint::Equals(a, b));
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Kind {
    Var,
    Num,
    Assignation,
    Function,
    If,
    Apply,
    Match,
    Arm,
    Variant,
    PatternWildcard,
    PatternTag,
    PatternBind,
}

#[derive(Debug, Clone)]
struct Node<'a> {
    lexem: &'a str,
    kind: Kind,
    children: Vec<Node<'a>>,
    r#type: RefCell<Option<TypePtr>>,
}

impl Node<'_> {
    fn get_type(&self, env: &mut TypeEnv) -> TypePtr {
        // Only source-language variables are looked up by name. Numerals,
        // constructor expressions, matches, applications etc. belong to
        // their own AST node, even when they have identical lexemes.
        if matches!(self.kind, Kind::Var) {
            if let Some(t) = env.get(self.lexem).cloned() {
                self.r#type.borrow_mut().replace(t.clone());
                return t;
            }
            let t = self.r#type.borrow_mut()
                .get_or_insert_with(|| Type::var(self.lexem))
                .clone();
            env.insert(self.lexem.into(), t.clone());
            return t;
        }

        let t = self.r#type.borrow_mut()
            .get_or_insert_with(|| env.fresh())
            .clone();
        // Retain the prototype's convention for a named function definition.
        if matches!(self.kind, Kind::Function) {
            env.insert(self.lexem.into(), t.clone());
        }
        t
    }

    fn find(&self, env: &mut TypeEnv) -> TypePtr {
        self.get_type(env).borrow().find(env)
    }

    fn set_type_equals(&self, ty: TypePtr, env: &mut TypeEnv) {
        let own = self.get_type(env);
        env.equal(own, ty);
    }
}

/// Reject an infinite type such as alpha = A(alpha).
fn occurs_in(var_name: &str, ty: &TypePtr, env: &TypeEnv) -> bool {
    let resolved = ty.borrow().find(env);
    let t = resolved.borrow().clone();
    match t {
        Type::Var(v) => v.name == var_name,
        Type::Con(c) => c.args.iter().any(|x| occurs_in(var_name, x, env)),
        Type::Variant(_, payload) => occurs_in(var_name, &payload, env),
        Type::Scheme(s) => occurs_in(var_name, &s.ty, env),
    }
}

/// Ordinary first-order unification: ONLY equality.
fn unify(left: TypePtr, right: TypePtr, env: &mut TypeEnv) {
    let left = left.borrow().find(env);
    let right = right.borrow().find(env);
    let lt = left.borrow().clone();
    let rt = right.borrow().clone();

    match (lt, rt) {
        (Type::Var(a), Type::Var(b)) if a.name == b.name => {},
        (Type::Var(a), _) => {
            assert!(!occurs_in(&a.name, &right, env), "occurs check failed for {}", a.name);
            env.substitute(&a, right);
        }
        (_, Type::Var(_)) => unify(right, left, env),
        (Type::Con(a), Type::Con(b)) => {
            assert_eq!(a.name, b.name, "different type constructors");
            assert_eq!(a.args.len(), b.args.len(), "different type arities");
            for (x, y) in a.args.into_iter().zip(b.args) {
                unify(x, y, env);
            }
        }
        (Type::Variant(tag_a, payload_a), Type::Variant(tag_b, payload_b)) => {
            assert_eq!(tag_a, tag_b, "different variant tags");
            unify(payload_a, payload_b, env);
        }
        (Type::Scheme(_), _) | (_, Type::Scheme(_)) => {
            panic!("instantiate a scheme before unification")
        }
        (a, b) => panic!("cannot unify {a:?} with {b:?}"),
    }
}

/// A pattern checks the *same* type as the selected scrutinee.
/// It emits equations; it never computes an intersection or type difference.
///
///   _       : no equation, no variable
///   v       : v has selected's type
///   A(p)    : selected = A(beta); recursively check p against beta
fn bind_pattern(pattern: &Node<'_>, selected: TypePtr, env: &mut TypeEnv) {
    match pattern.kind {
        Kind::PatternWildcard => {},
        Kind::PatternBind => {
            pattern.r#type.borrow_mut().replace(selected.clone());
            env.insert(pattern.lexem.into(), selected);
        }
        Kind::PatternTag => {
            let [payload_pattern] = &pattern.children[..] else {
                panic!("tag pattern requires exactly one payload pattern");
            };
            let payload = env.fresh();
            let expected = Type::variant(pattern.lexem, payload.clone());
            env.equal(selected, expected);
            bind_pattern(payload_pattern, payload, env);
        }
        _ => panic!("unsupported pattern: {:?}", pattern.kind),
    }
}

fn inferno<'a>(ast: &'a Node<'a>, env: TypeEnv) -> TypeEnv {
    solve(inferno_rec(ast, env))
}

fn inferno_rec<'a>(ast: &'a Node<'a>, mut env: TypeEnv) -> TypeEnv {
    match ast.kind {
        Kind::Num => ast.set_type_equals(Type::con("u32"), &mut env),
        Kind::Var => {
            ast.get_type(&mut env);
        }
        Kind::Assignation => {
            let [left, right] = &ast.children[..] else {
                panic!("assignment requires two children")
            };
            env = inferno_rec(left, env);
            env = inferno_rec(right, env);
            let lt = left.get_type(&mut env);
            let rt = right.get_type(&mut env);
            env.equal(lt, rt);
            ast.set_type_equals(Type::con("()"), &mut env);
        }
        Kind::Function => {
            let [arg, body] = &ast.children[..] else {
                panic!("function requires parameter and body")
            };
            let outer_variables = env.variables.clone();
            let arg_ty = env.fresh();
            arg.r#type.borrow_mut().replace(arg_ty.clone());
            env.insert(arg.lexem.into(), arg_ty.clone());
            env = inferno_rec(body, env);
            let body_ty = body.get_type(&mut env);
            env.variables = outer_variables;
            ast.set_type_equals(Type::con_with_args("->", vec![arg_ty, body_ty]), &mut env);
        }
        Kind::Apply => {
            let [function, argument] = &ast.children[..] else {
                panic!("apply requires function and argument")
            };
            env = inferno_rec(function, env);
            env = inferno_rec(argument, env);
            let function_ty = function.get_type(&mut env);
            let argument_ty = argument.get_type(&mut env);
            let result_ty = env.fresh();
            env.equal(function_ty, Type::con_with_args("->", vec![argument_ty, result_ty.clone()]));
            ast.r#type.borrow_mut().replace(result_ty);
        }
        Kind::Variant => {
            let [argument] = &ast.children[..] else {
                panic!("variant constructor requires one payload")
            };
            env = inferno_rec(argument, env);
            let payload_ty = argument.get_type(&mut env);
            ast.r#type.borrow_mut().replace(Type::variant(ast.lexem, payload_ty));
        }
        Kind::Match => {
            let [scrutinee, arms @ ..] = &ast.children[..] else {
                panic!("match requires scrutinee")
            };
            assert!(!arms.is_empty(), "empty match");
            env = inferno_rec(scrutinee, env);
            let selected = scrutinee.get_type(&mut env);
            // All branch bodies must have ONE common type.
            let match_result = env.fresh();

            for arm in arms {
                let [pattern, body] = &arm.children[..] else {
                    panic!("arm requires pattern and body")
                };
                // Pattern-bound identifiers are scoped to this one arm.
                let outer_variables = env.variables.clone();
                bind_pattern(pattern, selected.clone(), &mut env);
                env = inferno_rec(body, env);
                let body_ty = body.get_type(&mut env);
                env.equal(match_result.clone(), body_ty);
                env.variables = outer_variables;
            }

            ast.r#type.borrow_mut().replace(match_result);
        }
        _ => panic!("unexpected node in expression position: {:?}", ast.kind),
    }
    env
}

fn solve(mut env: TypeEnv) -> TypeEnv {
    for constraint in std::mem::take(&mut env.constraints) {
        let Constraint::Equals(a, b) = constraint;
        unify(a, b, &mut env);
    }
    env
}

fn main() {
    // An unconstrained payload remains polymorphic in the *monotype* sense:
    // x = A(beta), result = beta. No artificial u32 assumption.
    let ast = match_node(var("x"), vec![arm(tag("A", bind("value")), var("value"))]);
    let mut env = inferno(&ast, TypeEnv::default());
    println!("x: {:?}", ast.children[0].find(&mut env).borrow());
    println!("result: {:?}", ast.find(&mut env).borrow());
}

// Tiny test / example AST builders (not part of the algorithm).
fn node<'a>(lexem: &'a str, kind: Kind, children: Vec<Node<'a>>) -> Node<'a> {
    Node { lexem, kind, children, r#type: Default::default() }
}
fn var(name: &str) -> Node<'_> { node(name, Kind::Var, vec![]) }
fn num() -> Node<'static> { node("42", Kind::Num, vec![]) }
fn bind(name: &str) -> Node<'_> { node(name, Kind::PatternBind, vec![]) }
fn wildcard() -> Node<'static> { node("_", Kind::PatternWildcard, vec![]) }
fn tag<'a>(name: &'a str, payload: Node<'a>) -> Node<'a> {
    node(name, Kind::PatternTag, vec![payload])
}
fn variant<'a>(name: &'a str, payload: Node<'a>) -> Node<'a> {
    node(name, Kind::Variant, vec![payload])
}
fn arm<'a>(pattern: Node<'a>, body: Node<'a>) -> Node<'a> {
    node("=>", Kind::Arm, vec![pattern, body])
}
fn match_node<'a>(selected: Node<'a>, arms: Vec<Node<'a>>) -> Node<'a> {
    let mut children = vec![selected];
    children.extend(arms);
    node("match", Kind::Match, children)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_u32(t: &TypePtr) {
        assert_eq!(*t.borrow(), *Type::con("u32").borrow());
    }

    #[test]
    fn match_unknown_scrutinee_has_symbolic_payload() {
        let ast = match_node(var("x"), vec![arm(tag("A", bind("value")), var("value"))]);
        let mut env = inferno(&ast, TypeEnv::default());
        let x = ast.children[0].find(&mut env);
        let result = ast.find(&mut env);
        match x.borrow().clone() {
            Type::Variant(tag, payload) => {
                assert_eq!(tag, "A");
                assert!(matches!(&*payload.borrow(), Type::Var(_)));
                assert_eq!(*payload.borrow(), *result.borrow());
            }
            other => panic!("expected x = A(beta), got {other:?}"),
        }
        assert!(env.constraints.is_empty());
    }

    #[test]
    fn known_variant_extracts_payload() {
        let ast = match_node(variant("A", num()), vec![arm(tag("A", bind("v")), var("v"))]);
        let mut env = inferno(&ast, TypeEnv::default());
        assert_u32(&ast.find(&mut env));
    }

    #[test]
    fn nested_patterns_unify_recursively() {
        let ast = match_node(
            variant("A", variant("B", num())),
            vec![arm(tag("A", tag("B", bind("v"))), var("v"))],
        );
        let mut env = inferno(&ast, TypeEnv::default());
        assert_u32(&ast.find(&mut env));
    }

	#[test]
	fn paper_example_2_classical_limitation() {
	    let ast = match_node(
	        var("x"),
	        vec![
	            arm(
	                tag("A", wildcard()),
	                variant("B", num()),
	            ),
	            arm(
	                bind("y"),
	                var("y"),
	            ),
	        ],
	    );
	
	    let _ = inferno(&ast, TypeEnv::default());
	}

    #[test]
    fn wildcard_does_not_restrict_scrutinee() {
        let ast = match_node(var("x"), vec![arm(wildcard(), num())]);
        let mut env = inferno(&ast, TypeEnv::default());
        assert_u32(&ast.find(&mut env));
        assert!(matches!(&*ast.children[0].find(&mut env).borrow(), Type::Var(_)));
    }

    #[test]
    fn variable_pattern_aliases_scrutinee() {
        let ast = match_node(var("x"), vec![arm(bind("v"), var("v"))]);
        let mut env = inferno(&ast, TypeEnv::default());
        assert_eq!(*ast.find(&mut env).borrow(), *ast.children[0].find(&mut env).borrow());
        assert!(env.get("v").is_none(), "pattern binding leaked out of arm");
    }

    #[test]
    fn two_bodies_unify_to_one_result_type() {
        let ast = match_node(
            variant("A", num()),
            vec![
                arm(tag("A", bind("v")), var("v")),
                arm(wildcard(), num()),
            ],
        );
        let mut env = inferno(&ast, TypeEnv::default());
        assert_u32(&ast.find(&mut env));
    }

    #[test]
    #[should_panic(expected = "different variant tags")]
    fn different_tag_is_not_structurally_unifiable() {
        let ast = match_node(variant("B", num()), vec![arm(tag("A", bind("v")), var("v"))]);
        let _ = inferno(&ast, TypeEnv::default());
    }

    #[test]
    #[should_panic(expected = "occurs check failed")]
    fn occurs_check_blocks_infinite_variant() {
        let mut env = TypeEnv::default();
        let x = Type::var("x");
        unify(x.clone(), Type::variant("A", x), &mut env);
    }

    #[test]
    fn assignments_continue_to_work() {
        let a_is_b = node("=", Kind::Assignation, vec![var("a"), var("b")]);
        let b_is_num = node("=", Kind::Assignation, vec![var("b"), num()]);
        let mut env = inferno(&a_is_b, TypeEnv::default());
        env = inferno(&b_is_num, env);
        assert_u32(&a_is_b.children[0].find(&mut env));
    }

    #[test]
    fn function_and_application_continue_to_work() {
        let identity = node("id", Kind::Function, vec![var("a"), var("a")]);
        let env = inferno(&identity, TypeEnv::default());
        let apply = node("id()", Kind::Apply, vec![var("id"), num()]);
        let mut env = inferno(&apply, env);
        assert_u32(&apply.find(&mut env));
    }
}

