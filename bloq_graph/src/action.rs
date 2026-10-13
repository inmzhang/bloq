//! Classical measurements, expressions, feedback, and branch actions.

use bloq_utils::{Direction, PauliBasis};
use glam::IVec3;
use std::collections::BTreeSet;

/// A binary operator in an action expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// Logical XOR.
    Xor,
    /// Logical AND.
    And,
    /// Logical OR.
    Or,
}

impl BinaryOp {
    /// Precedence level: higher number = tighter binding.
    pub const fn precedence(&self) -> u8 {
        match self {
            BinaryOp::Or => 1,
            BinaryOp::Xor => 2,
            BinaryOp::And => 3,
        }
    }

    fn symbol(&self) -> &'static str {
        match self {
            BinaryOp::Xor => "^",
            BinaryOp::And => "&",
            BinaryOp::Or => "|",
        }
    }
}

/// A boolean expression tree over measurement outcome variables.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    /// A named variable reference (e.g., a measurement outcome name).
    Var(String),
    /// Logical NOT of a sub-expression.
    Not(Box<Expr>),
    /// A binary operation combining two sub-expressions.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
}

impl Expr {
    /// Variable occurrences from left to right, including repeated names.
    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        let mut pending = vec![self];
        std::iter::from_fn(move || {
            while let Some(expr) = pending.pop() {
                match expr {
                    Self::Var(name) => return Some(name.as_str()),
                    Self::Not(inner) => pending.push(inner),
                    Self::Binary(_, left, right) => {
                        pending.extend([right.as_ref(), left.as_ref()]);
                    }
                }
            }
            None
        })
    }

    pub(crate) fn negated(self) -> Self {
        match self {
            Self::Not(inner) => *inner,
            expr => Self::Not(Box::new(expr)),
        }
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Var(name) => write!(f, "{}", name),
            Expr::Not(expr) => {
                if matches!(expr.as_ref(), Expr::Binary(..)) {
                    write!(f, "!({})", expr)
                } else {
                    write!(f, "!{}", expr)
                }
            }
            Expr::Binary(op, lhs, rhs) => {
                let prec = op.precedence();
                // Parenthesize LHS if it has strictly lower precedence
                if needs_parens(lhs, prec, true) {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }
                write!(f, " {} ", op.symbol())?;
                // Parenthesize RHS if lower-or-equal precedence (left-assoc)
                if needs_parens(rhs, prec, false) {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
        }
    }
}

/// Returns true if the child expression needs parentheses given the parent's precedence.
/// `is_left` indicates whether the child is the left operand (left-assoc: left child
/// only needs parens if strictly lower precedence; right child needs parens if
/// lower-or-equal to preserve associativity).
fn needs_parens(expr: &Expr, parent_prec: u8, is_left: bool) -> bool {
    if let Expr::Binary(child_op, _, _) = expr {
        let child_prec = child_op.precedence();
        if is_left {
            child_prec < parent_prec
        } else {
            child_prec <= parent_prec
        }
    } else {
        false
    }
}

/// A single Pauli correction site for an [`Action::Feedback`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeedbackTarget {
    /// Pauli basis of the correction to apply.
    pub pauli: PauliBasis,
    /// Block position the correction acts on.
    pub target: IVec3,
    /// A wire leaving `target`, in that endpoint's Pauli frame. `None` selects
    /// the first outgoing temporal wire, then the last incoming temporal wire,
    /// then a sole spatial wire, ordered by the neighboring source position.
    pub direction: Option<Direction>,
}

/// The target of a measurement action — either a node or a specific edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeasureTarget {
    /// Measurement on a node position.
    Node(IVec3),
    /// Measurement on an edge identified by source position and direction.
    Edge {
        /// Canonical source endpoint of the measured edge.
        src: IVec3,
        /// Direction from `src` to the other endpoint.
        dir: Direction,
    },
}

/// A classical-control action attached to a block graph program.
///
/// Actions encode classical control flow driven by measurement outcomes:
/// variable definitions, conditional feedback, selective resolution,
/// named structural branching, and post-selection via discard conditions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Define a named boolean variable from an expression.
    Let {
        /// Name bound by the action.
        name: String,
        /// Boolean expression assigned to `name`.
        expr: Expr,
    },
    /// Bind a measurement outcome variable to a node or edge.
    Measure {
        /// Node or edge whose outcome is recorded.
        target: MeasureTarget,
        /// Name assigned to the measurement record.
        name: String,
    },
    /// Discard the shot if the expression evaluates to true (post-selection).
    DiscardIf(Expr),
    /// Resolve a selective block at `target` using `condition`.
    Resolve {
        /// Selective block to resolve.
        target: IVec3,
        /// Boolean expression selecting the block basis.
        condition: Expr,
    },
    /// Resolve the explicit branch region internally keyed by `target`.
    /// [`BlockGraph::to_blog_text`](crate::BlockGraph::to_blog_text) writes its
    /// public name because an action alone does not carry that name.
    Branch {
        /// Internal key position of the branch region.
        target: IVec3,
        /// Boolean expression selecting the branch choice.
        condition: Expr,
    },
    /// Apply Pauli feedback to one or more targets, optionally gated on a condition.
    Feedback {
        /// Pauli correction sites affected by the action.
        targets: Vec<FeedbackTarget>,
        /// Optional Boolean guard; absence means unconditional feedback.
        condition: Option<Expr>,
    },
}

impl Action {
    /// Boolean variable names consumed by this action, in sorted order.
    pub fn referenced_names(&self) -> BTreeSet<&str> {
        let expression = match self {
            Self::Let { expr, .. } | Self::DiscardIf(expr) => Some(expr),
            Self::Resolve { condition, .. } | Self::Branch { condition, .. } => Some(condition),
            Self::Feedback { condition, .. } => condition.as_ref(),
            Self::Measure { .. } => None,
        };
        expression.into_iter().flat_map(Expr::names).collect()
    }

    pub(crate) fn try_with_orientation(
        &self,
        orientation: crate::ModuleOrientation,
    ) -> Result<Self, crate::BlockGraphError> {
        self.try_map_targets(
            |position| orientation.try_rotate_position(position),
            |direction| orientation.rotate_direction(direction),
        )
    }

    /// Returns a copy of the action with every position translated by `offset`.
    ///
    /// # Panics
    ///
    /// Panics if a translated position exceeds the coordinate range. Use
    /// [`try_with_shift`](Self::try_with_shift) for untrusted offsets.
    pub fn with_shift(&self, offset: impl Into<IVec3>) -> Self {
        self.try_with_shift(offset)
            .expect("shifted action positions must fit in the i32 coordinate range")
    }

    /// Returns a copy of the action with every position translated by `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::BlockGraphError::CoordinateOverflow`] if a translated
    /// position or measurement-edge endpoint exceeds the coordinate range.
    pub fn try_with_shift(&self, offset: impl Into<IVec3>) -> Result<Self, crate::BlockGraphError> {
        let offset = offset.into();
        self.try_map_targets(
            |position| crate::checked_add_position(position, offset),
            |direction| direction,
        )
    }

    fn try_map_targets(
        &self,
        position: impl Fn(IVec3) -> Result<IVec3, crate::BlockGraphError>,
        direction: impl Fn(Direction) -> Direction,
    ) -> Result<Self, crate::BlockGraphError> {
        let mut action = self.clone();
        match &mut action {
            Action::Resolve { target, .. } | Action::Branch { target, .. } => {
                *target = position(*target)?;
            }
            Action::Feedback { targets, .. } => {
                for target in targets {
                    target.target = position(target.target)?;
                    target.direction = target.direction.map(&direction);
                    if let Some(dir) = target.direction {
                        crate::checked_add_position(target.target, dir.to_ivec3())?;
                    }
                }
            }
            Action::Measure { target, .. } => match target {
                MeasureTarget::Node(target) => *target = position(*target)?,
                MeasureTarget::Edge { src, dir } => {
                    *src = position(*src)?;
                    *dir = direction(*dir);
                    crate::checked_add_position(*src, dir.to_ivec3())?;
                }
            },
            Action::Let { .. } | Action::DiscardIf(..) => {}
        }
        Ok(action)
    }

    pub(crate) fn display_with_ids(&self, p2i: impl Fn(IVec3) -> Option<u32>) -> String {
        let target_str = |target| {
            if let Some(id) = p2i(target) {
                return id.to_string();
            }
            target.to_string()
        };
        match self {
            Action::Let { name, expr } => format!("{name} = {expr}"),
            Action::Measure { target, name } => {
                let target_str_val = match target {
                    MeasureTarget::Node(pos) => target_str(*pos),
                    MeasureTarget::Edge { src, dir } => {
                        format!("{} -> {}", target_str(*src), dir)
                    }
                };
                format!("{name} = measure {target_str_val}")
            }
            Action::DiscardIf(expr) => format!("discard if {expr}"),
            Action::Resolve { target, condition } | Action::Branch { target, condition } => {
                format!("resolve {} if {}", target_str(*target), condition)
            }
            Action::Feedback { targets, condition } => {
                let ts = targets
                    .iter()
                    .map(|t| match t.direction {
                        Some(dir) => format!("{} {} -> {}", t.pauli, target_str(t.target), dir),
                        None => format!("{} {}", t.pauli, target_str(t.target)),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                if let Some(cond) = condition {
                    format!("feedback {} if {}", ts, cond)
                } else {
                    format!("feedback {}", ts)
                }
            }
        }
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.display_with_ids(|_| None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_shift_rejects_overflowing_measurement_edge() {
        let action = Action::Measure {
            target: MeasureTarget::Edge {
                src: IVec3::new(i32::MAX - 1, 0, 0),
                dir: Direction::XPLUS,
            },
            name: "m".into(),
        };

        assert!(matches!(
            action.try_with_shift(IVec3::X),
            Err(crate::BlockGraphError::CoordinateOverflow { .. })
        ));
    }

    #[test]
    fn expression_display_preserves_precedence_and_associativity() -> Result<(), crate::ParseError>
    {
        for (source, expected) in [
            ("a | b & c", "a | b & c"),
            ("(a | b) & c", "(a | b) & c"),
            ("!(a | b)", "!(a | b)"),
            ("!a", "!a"),
            ("a ^ (b ^ c)", "a ^ (b ^ c)"),
            ("(a ^ b) ^ c", "a ^ b ^ c"),
        ] {
            let actions = crate::parse_actions(&format!("result = {source}"), |_, _| None)?;
            let [Action::Let { expr, .. }] = actions.as_slice() else {
                unreachable!()
            };
            assert_eq!(expr.to_string(), expected, "{source}");
        }
        Ok::<_, crate::ParseError>(())
    }

    #[test]
    fn branch_display_and_shift_preserve_condition() {
        let action = Action::Branch {
            target: IVec3::new(1, 2, 3),
            condition: Expr::Var("m".into()),
        };

        assert_eq!(action.to_string(), "resolve [1, 2, 3] if m");
        assert_eq!(
            action.with_shift(IVec3::new(4, 5, 6)),
            Action::Branch {
                target: IVec3::new(5, 7, 9),
                condition: Expr::Var("m".into()),
            }
        );
    }
}
