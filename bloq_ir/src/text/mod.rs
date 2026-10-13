//! The human-readable Bloq IR text exchange format.
//!
//! Writing and re-parsing preserves templates (circuits with
//! explicit measurement ids, side tables, `boundary_flows`), the node graph
//! with exact node ids (holes from removals included, so the deterministic
//! schedule survives), and all provenance channels.
//! [`Bloq`]'s `Display` impl renders this format.
//!
//! # Layout
//!
//! ```text
//! BLOQIR 1
//! metadata bloq_compile.code_distance u64 3
//! metadata bloq_compile.convention string fixed-bulk
//! logical-input (0,0,0) instance i0 x X(8,0) z Z(8,0)
//! logical-output (0,0,1) instance i0 x X(8,0) z Z(8,0)
//!
//! template t0 {
//!   circuit {                           # entry body b0
//!     H (0,0)
//!     TICK
//!     M(0.01) (0,0):m0                  # optional readout-flip probability
//!     DEPOLARIZE1(0.001) (0,0)
//!     MPP X(1,1)*X(1,3):m1
//!     REPEAT 3 b1                       # bodies are flat, by id
//!     CPAULI X[m0]@(0,0) Z[m2]@(1,0)
//!   }
//!   body b1 {
//!     MX (1,0):m2
//!   }
//!   meas m9 @ (4,4)                     # registry records no op implies
//!   detector -m0*m2 @ (0.5,1)           # optional sign; terms: mN | sN
//!   detector body(b1) m2*s0
//!   loop(b1,s0) init m0 next m2
//!   flow X(0,0) -> Z(1,0)*X(2,0) meas m0*m2 sign -1 center (1,1) restart
//!   restart m0*m2
//! }
//!
//! bundle b0 owners t0 {
//!   detector o0:m0 @ (0,1)             # owner slots are bundle-local
//! }
//!
//! graph {
//!   n0 quantum {
//!     timeline rounds 3 6                 # cumulative rounds per occupied z layer
//!     instance i0 t0 @ (8,0)
//!     detector i0:m0*i1:m2 @ (1,2)      # node parity terms: iN:mK
//!     use b0 i0 @ (0,0)                # bind owners to global instances
//!     guard 0 b0                        # guard use index 0 as a whole
//!     restart i0:m0
//!     from blocks (0,0,0) (1,0,0)
//!   }
//!   n2 observable fragment measurements i0:m0*i0:m2 from generator 0
//!   n3 compute in0 ^ !(in1 & in2) from frame x (0,0,1)
//!   n6 observable fragment operators i0 input X(0,0)*Z(0,1)
//!   n7 observable 0
//!   result n7:flip                      # optional local bit result; default false
//!   bindings n6                         # optional fragment/region exports; default empty
//!   n8 discard in0
//!   n9 rus in0 {
//!     body { ... }                      # a nested graph level
//!   }
//!   n11 rus in0 source n1:flip {
//!     body { ... }
//!   }
//!   n0 -> n2 order
//!   n2 -> n7 compose 0
//!   n6 -> n7 compose 1
//!   n7 -> n3 value 0 flip                # omitted port means corrected
//!   n3 -> n7 value 2 feedback 3
//!   n0 -> n1 quantum (0,0,0)>(0,0,1) padding (0,0,0)>(0,0,1) offset (8,0) one t1 loop t2
//! }
//! ```
//!
//! `circuit bN { ... }` names a nonzero entry body; `circuit { ... }` means b0.
//! All other bodies retain their explicit ids.
//!
//! Node ids (`nK`) are written explicitly and gaps in the sequence are
//! reconstructed as vacant graph slots on parse. What is *not* preserved
//! (by this codec or the binary one) is vacancy past the last live node and
//! the free-slot reuse order — history that only affects the ids future
//! `add_node` calls hand out, never the program's semantics.
//!
//! Metadata keys and string values use percent-escaped UTF-8 tokens; `~`
//! represents an empty token. `#` starts a comment anywhere.
//!
//! The measurement registry is implied: `M`/`MX`/`MY`/`MPP` op targets define
//! `(id, qubit)` pairs, and `meas` lines add the records no op implies.
//!
//! Parsing rejects structurally malformed input (duplicate node ids, dangling
//! edge endpoints, one measurement id bound to two qubits, overflowing
//! coordinate translations, exhausted measurement-output ids, an implausibly
//! sparse node-id space, or sparse circuit body ids) but does not check IR
//! semantics — run [`Bloq::validate`] on untrusted input. There is no
//! cross-version compatibility: the reader rejects any header version other
//! than its own.

mod parse;
mod write;

use std::fmt::{self, Write as _};

use crate::Bloq;

pub use parse::TextParseError;

/// The pre-release text format version. IR changes update the codec without
/// bumping this value until the format is released.
pub const BLOQ_TEXT_VERSION: u32 = 1;

/// The conventional file extension for the text format.
pub const BLOQ_TEXT_EXTENSION: &str = "bloqir";

fn encode_metadata_token(value: &str) -> String {
    if value.is_empty() {
        return "~".to_owned();
    }
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            encoded.push(char::from(byte));
        } else {
            write!(encoded, "%{byte:02X}").expect("writing to a String cannot fail");
        }
    }
    encoded
}

fn decode_metadata_token(token: &str) -> Result<String, &'static str> {
    if token == "~" {
        return Ok(String::new());
    }
    let source = token.as_bytes();
    let mut decoded = Vec::with_capacity(source.len());
    let mut index = 0;
    while index < source.len() {
        if source[index] != b'%' {
            decoded.push(source[index]);
            index += 1;
            continue;
        }
        let hex = source
            .get(index + 1..index + 3)
            .ok_or("incomplete percent escape")?;
        let digit = |byte: u8| match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        };
        let high = digit(hex[0]).ok_or("invalid percent escape")?;
        let low = digit(hex[1]).ok_or("invalid percent escape")?;
        decoded.push((high << 4) | low);
        index += 3;
    }
    String::from_utf8(decoded).map_err(|_| "metadata token is not UTF-8")
}

impl Bloq {
    /// Render this program in the text exchange format (also `Display`).
    pub fn to_text(&self) -> String {
        write::write_bloq_text(self)
    }

    /// Parse a program from the text exchange format.
    ///
    /// # Errors
    ///
    /// Returns [`TextParseError`] for malformed syntax, unsupported versions,
    /// invalid references, or structurally invalid serialized data.
    pub fn from_text(text: &str) -> Result<Self, TextParseError> {
        let mut bloq = parse::parse_bloq_text(text)?;
        bloq.top_mut().share_classical_data();
        Ok(bloq)
    }
}

impl fmt::Display for Bloq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

#[cfg(test)]
mod tests {
    use bloq_circuit::GateType;

    use crate::test_fixture::sample_bloq;
    use crate::{
        Bloq, BloqNode, BloqNodeId, BloqValidationError, ClassicalNode, MetadataValue,
        NodeTemplateInstanceMergeError, QuantumTimeline,
    };

    /// Structural equality via the binary codec: identical postcard bytes ⇔
    /// identical programs (slot occupancy included).
    fn assert_same_program(a: &Bloq, b: &Bloq) {
        assert_eq!(a.to_binary(), b.to_binary());
    }

    #[test]
    fn text_round_trip_is_lossless() {
        let bloq = sample_bloq();
        let text = bloq.to_text();
        assert!(text.starts_with("BLOQIR 1\n"));
        let restored = Bloq::from_text(&text).unwrap_or_else(|error| {
            panic!("fixture text must parse: {error}\n---\n{text}");
        });
        assert_same_program(&bloq, &restored);
        // And the text itself is a fixpoint.
        assert_eq!(text, restored.to_text());
    }

    #[test]
    fn detector_bundle_round_trips_preserve_sharing_and_guard() {
        let bloq = Bloq::from_text(
            "BLOQIR 1\n\
             template t0 {\n  circuit {\n    M (0,0):m0\n  }\n}\n\
             bundle b0 owners t0 {\n  detector -o0:m0 @ (1,2)\n}\n\
             graph {\n\
               n0 quantum {\n    instance i0 t0 @ (0,0)\n    detector -i0:m0 @ (1,2)\n    use b0 i0 @ (0,0)\n    guard 0 b0\n  }\n\
               n1 quantum {\n    instance i1 t0 @ (2,0)\n    use b0 i1 @ (2,0)\n  }\n\
             }\n",
        )
        .unwrap();
        let text = bloq.to_text();
        assert_eq!(text.matches("bundle b0 owners").count(), 1);
        assert_eq!(text.matches("use b0").count(), 2);
        for restored in [
            Bloq::from_text(&text).unwrap(),
            Bloq::from_binary(&bloq.to_binary()).unwrap(),
        ] {
            assert_same_program(&bloq, &restored);
            assert_eq!(restored.detector_bundles().len(), 1);
            for node in [BloqNodeId(0), BloqNodeId(1)] {
                assert_eq!(restored[node].expect_quantum().detector_bundles.len(), 1);
            }
            assert_eq!(
                restored[BloqNodeId(0)].expect_quantum().guards[0].detector_bundles,
                vec![0]
            );
            let mut rows = restored
                .node_detectors(restored[BloqNodeId(0)].expect_quantum())
                .unwrap();
            assert_eq!(
                rows.next().unwrap().to_owned(),
                rows.next().unwrap().to_owned()
            );
            let second = restored
                .node_detectors(restored[BloqNodeId(1)].expect_quantum())
                .unwrap()
                .next()
                .unwrap()
                .to_owned();
            assert_eq!(second.coords.unwrap().as_slice(), &[3.0, 2.0]);
        }
    }

    #[test]
    fn malformed_bundle_tokens_return_parse_errors() {
        for text in [
            "BLOQIR 1\nbundle b1 owners {\n}\ngraph {\n}\n",
            "BLOQIR 1\nbundle b0 owners {\n detector i0:m0\n}\ngraph {\n}\n",
            "BLOQIR 1\nbundle b0 owners {\n detector o4294967296:m0\n}\ngraph {\n}\n",
            "BLOQIR 1\ngraph {\n n0 quantum {\n use b0 i0\n }\n}\n",
        ] {
            assert!(Bloq::from_text(text).is_err(), "{text}");
        }
    }

    #[test]
    fn malformed_bundle_bindings_return_typed_errors() {
        use crate::DetectorBundleError;

        let bloq = Bloq::from_text(
            "BLOQIR 1\n\
             bundle b0 owners t0 t0 {\n detector o0:m0\n}\n\
             graph {\n n0 quantum {\n use b0 i0 @ (0,0)\n }\n}\n",
        )
        .unwrap();
        assert!(matches!(
            bloq.node_detectors(bloq[BloqNodeId(0)].expect_quantum()),
            Err(DetectorBundleError::BindingArity {
                expected: 2,
                actual: 1,
                ..
            })
        ));

        let bloq = Bloq::from_text(
            "BLOQIR 1\n\
             bundle b0 owners t0 {\n detector o1:m0\n}\n\
             graph {\n n0 quantum {\n use b0 i0 @ (0,0)\n }\n}\n",
        )
        .unwrap();
        assert!(matches!(
            bloq.node_detectors(bloq[BloqNodeId(0)].expect_quantum()),
            Err(DetectorBundleError::InvalidLocalOwner {
                owner: 1,
                owners: 1
            })
        ));
    }

    #[test]
    fn expression_codecs_preserve_arity_nesting_and_selection_order() {
        use crate::ClassicalExpr::{self, And, Const, In, Or, Xor};
        let mut expressions = Vec::new();
        for build in [Xor, And, Or] {
            expressions.extend([
                build(Box::new([])),
                build(Box::new([In(0)])),
                build(Box::new([In(0), In(1), In(2)])),
                build(Box::new([
                    build(Box::new([In(0), In(1)])),
                    build(Box::new([In(2), Const(true)])),
                ])),
            ]);
        }
        expressions.extend([
            ClassicalExpr::Parity {
                inputs: Box::new([]),
                constant: true,
            },
            ClassicalExpr::Parity {
                inputs: Box::new([u32::MAX, 0, u32::MAX]),
                constant: false,
            },
            ClassicalExpr::parity([0, 1, 2], true),
        ]);
        expressions.push(ClassicalExpr::select(
            In(0),
            Xor(Box::new([In(1), Const(true), Const(false)])),
            ClassicalExpr::select(In(2), Or(Box::new([])), And(Box::new([]))),
        ));
        for expr in expressions {
            let mut program = Bloq::new();
            program.add_node(BloqNode::classical(ClassicalNode::Compute { expr }));
            let text = program.to_text();
            assert_same_program(&program, &Bloq::from_text(&text).unwrap());
            assert_same_program(&program, &Bloq::from_binary(&program.to_binary()).unwrap());
        }
        let flat = Bloq::from_text("BLOQIR 1\ngraph {\n n0 compute in0 ^ in1 ^ in2\n}\n").unwrap();
        let ClassicalNode::Compute {
            expr: Xor(operands),
        } = flat[BloqNodeId(0)].try_classical().unwrap()
        else {
            panic!("flat parity");
        };
        assert_eq!(operands.len(), 3);
        for malformed in [
            "parity()",
            "parity(in0)",
            "parity(0, 1)",
            "parity(0, !in0)",
            "select()",
            "select(0)",
            "select(0,1)",
            "select(0,1,0,1)",
            "select(0 1 0)",
            "select(0,1,0,)",
            "xor(0,)",
            "in0 xor in1",
        ] {
            assert!(
                Bloq::from_text(&format!(
                    "BLOQIR 1\ngraph {{\n n0 compute {malformed}\n}}\n"
                ))
                .is_err(),
                "{malformed}"
            );
        }
    }

    #[test]
    fn nonzero_entry_body_round_trips_without_renumbering() {
        use bloq_circuit::{CircuitBody, CoordCircuit, Op};

        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [glam::IVec2::ZERO]).unwrap();
        let entry = circuit.add_body(CircuitBody::new());
        circuit.set_entry_body(entry).unwrap();
        circuit.push_repeat(bloq_circuit::BodyId(0), 2);
        let mut bloq = Bloq::new();
        let template = bloq
            .templates_mut()
            .insert(crate::BloqTemplate::new(circuit));
        let mut quantum = BloqNode::from_members(vec![]);
        quantum
            .expect_quantum_mut()
            .instances
            .push(crate::TemplateInstance::new(
                crate::TemplateInstanceId(0),
                template,
                glam::IVec2::ZERO,
            ));
        bloq.add_node(quantum);
        bloq.validate().unwrap();
        let restored = Bloq::from_binary(&bloq.to_binary()).unwrap();
        let text = restored.to_text();
        assert!(text.contains("circuit b1 {"));
        let parsed = Bloq::from_text(&text).unwrap();
        assert_same_program(&bloq, &parsed);
        let circuit = &parsed.templates()[crate::TemplateId(0)].circuit;
        assert_eq!(circuit.entry_body(), entry);
        assert!(matches!(
            circuit.body(entry).unwrap().ops(),
            [Op::Repeat {
                body: bloq_circuit::BodyId(0),
                repetitions: 2
            }]
        ));
        Bloq::from_text("BLOQIR 1\ntemplate t0 {\ncircuit {\n}\ncircuit b1 {\n}\n}\ngraph {\n}\n")
            .unwrap_err();
    }

    #[test]
    fn quantum_timeline_round_trips() {
        let mut bloq = sample_bloq();
        let quantum = bloq
            .node_mut(BloqNodeId(0))
            .expect("fixture quantum node")
            .expect_quantum_mut();
        quantum.timeline = Some(QuantumTimeline {
            layer_round_ends: vec![3, 3, 8],
        });

        let text = bloq.to_text();
        for restored in [
            Bloq::from_text(&text).expect("timeline parses"),
            Bloq::from_binary(&bloq.to_binary()).expect("timeline decodes"),
        ] {
            assert_same_program(&bloq, &restored);
            assert_eq!(
                restored[BloqNodeId(0)].expect_quantum().timeline,
                bloq[BloqNodeId(0)].expect_quantum().timeline
            );
        }
    }

    #[test]
    fn spatial_port_provenance_round_trips() {
        let bloq = Bloq::from_text(
            "BLOQIR 1\n\
             template t0 {\n\
               circuit {\n\
               }\n\
             }\n\
             graph {\n\
               n0 quantum {\n\
                 instance i0 t0 @ (0,0) from spatial-port input cube (1,2,3)\n\
                 from blocks (1,2,3)\n\
               }\n\
               n1 quantum {\n\
                 instance i1 t0 @ (0,0) from spatial-port output temporal-port (4,5,6)\n\
                 from spatial-port output (4,5,6)\n\
               }\n\
             }\n",
        )
        .expect("spatial-Port provenance parses");

        let text = bloq.to_text();
        let restored = Bloq::from_text(&text).expect("spatial-Port provenance re-parses");

        assert_same_program(&bloq, &restored);
        assert!(text.contains("from spatial-port input cube (1,2,3)"));
        assert!(text.contains("from spatial-port output (4,5,6)"));
    }

    #[test]
    fn text_round_trip_of_empty_program() {
        let bloq = Bloq::new();
        let restored = Bloq::from_text(&bloq.to_text()).expect("empty program parses");
        assert_same_program(&bloq, &restored);
    }

    #[test]
    fn signed_parities_and_flows_round_trip() {
        let bloq = Bloq::from_text(
            "BLOQIR 1\n\
             template t0 {\n\
               circuit {\n\
               }\n\
               detector -0\n\
               flow _ -> Z(0,0) sign -1\n\
             }\n\
             graph {\n\
             }\n",
        )
        .expect("signed IR parses");

        let text = bloq.to_text();
        assert!(text.contains("detector -0"));
        assert!(text.contains("flow _ -> Z(0,0) sign -1"));
        let restored = Bloq::from_text(&text).expect("written signed IR parses");
        assert_same_program(&bloq, &restored);
    }

    #[test]
    fn metadata_round_trip_is_lossless() {
        let mut bloq = Bloq::new();
        bloq.insert_metadata("compiler key#%", MetadataValue::U64(u64::MAX));
        bloq.insert_metadata(
            "compiler.string",
            MetadataValue::String("value # % \u{96ea}".to_owned()),
        );
        bloq.insert_metadata("empty", MetadataValue::String(String::new()));

        let text = bloq.to_text();
        let restored = Bloq::from_text(&text).expect("metadata parses");

        assert_eq!(bloq.metadata(), restored.metadata());
        assert_same_program(&bloq, &restored);
        assert_eq!(text, restored.to_text());
    }

    #[test]
    fn from_text_rejects_duplicate_metadata_keys() {
        let error = Bloq::from_text(
            "BLOQIR 1\nmetadata key u64 1\nmetadata key string value\ngraph {\n}\n",
        )
        .expect_err("duplicate metadata key");

        assert_eq!(error.line, 3);
        assert!(error.to_string().contains("duplicate metadata key `key`"));
    }

    #[test]
    fn output_declarations_have_explicit_defaults_and_reject_duplicates() {
        let program =
            Bloq::from_text("BLOQIR 1\ngraph {\n n0 compute 1\n result false\n bindings\n}\n")
                .unwrap();
        assert_eq!(program.value_output(), None);
        assert!(program.boundary_outputs().is_empty());
        assert!(!program.to_text().contains("result"));
        for declaration in ["result false\n result n0", "bindings\n bindings n0"] {
            let source = format!("BLOQIR 1\ngraph {{\n n0 compute 1\n {declaration}\n}}\n");
            assert!(
                Bloq::from_text(&source)
                    .unwrap_err()
                    .to_string()
                    .contains("duplicate")
            );
        }
    }

    #[test]
    fn text_round_trip_of_empty_fragment() {
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            Vec::new(),
        )));
        let text = bloq.to_text();
        assert!(text.contains("n0 observable fragment\n"));

        let restored = Bloq::from_text(&text).expect("writer's empty fragment parses");

        assert_same_program(&bloq, &restored);
    }

    #[test]
    fn output_ports_and_composition_round_trip_in_every_reference() {
        let source = "BLOQIR 1\ngraph {\n\
          n0 observable 0\n\
          n1 observable fragment\n\
          n2 compute in0\n\
          n3 quantum {\n  }\n\
          n4 quantum {\n  }\n\
          n5 rus in0 source n0:flip {\n\
            body {\n n0 observable 1\n result n0:flip\n }\n\
          }\n\
          n1 -> n0 compose 0\n\
          n0 -> n2 value 0 flip readout\n\
          n0 -> n3 value 0 corrected\n\
          n3 -> n4 quantum when n0:flip\n\
          result n0:corrected\n\
        }\n";
        let program = Bloq::from_text(source).unwrap();
        let text = program.to_text();
        assert!(text.contains("n1 -> n0 compose 0"));
        assert!(text.contains("value 0 flip readout"));
        assert!(text.contains("quantum when n0:flip"));
        assert!(text.contains("source n0:flip"));
        assert!(text.contains("result n0:flip"));
        assert!(!text.contains("corrected"), "corrected is the default port");
        assert_same_program(&program, &Bloq::from_text(&text).unwrap());
        assert_same_program(&program, &Bloq::from_binary(&program.to_binary()).unwrap());
    }

    #[test]
    fn obsolete_nodes_and_unknown_ports_are_rejected() {
        for statement in [
            "n0 accumulate",
            "n0 include",
            "n0 observable-raw 0",
            "n0 decode-accept 0",
            "n0 readout-recipe",
            "n0 compute 0\n result n0:raw",
            "n0 compute 0\n n0 -> n0 value 0 raw",
            "n0 observable fragment\n n0 -> n0 compose 0 flip",
        ] {
            let source = format!("BLOQIR 1\ngraph {{\n {statement}\n}}\n");
            assert!(Bloq::from_text(&source).is_err(), "accepted {statement}");
        }
    }

    #[test]
    fn from_text_rejects_unknown_version() {
        let error =
            Bloq::from_text("BLOQIR 99\ngraph {\n}\n").expect_err("version 99 is unsupported");
        assert_eq!(error.line, 1);
    }

    #[test]
    fn from_text_reports_line_numbers() {
        let error =
            Bloq::from_text("BLOQIR 1\ngraph {\n  n0 wobble\n}\n").expect_err("unknown node kind");
        assert_eq!(error.line, 3);
    }

    #[test]
    fn rus_without_body_local_restart_source_round_trips() {
        let source = "BLOQIR 1\ngraph {\n n0 rus 0 {\n body {\n }\n }\n}\n";
        let program = Bloq::from_text(source).unwrap();
        program.validate().unwrap();
        assert!(!program.to_text().contains("source"));
        assert_same_program(&program, &Bloq::from_text(&program.to_text()).unwrap());
        assert_same_program(&program, &Bloq::from_binary(&program.to_binary()).unwrap());
    }

    #[test]
    fn rus_unfed_restart_input_without_source_fails_validation() {
        let program =
            Bloq::from_text("BLOQIR 1\ngraph {\n n0 rus in0 {\n body {\n }\n }\n}\n").unwrap();
        assert!(program.validate().is_err());
    }

    #[test]
    fn from_text_rejects_duplicate_node_at_its_line() {
        let error = Bloq::from_text("BLOQIR 1\ngraph {\n  n0 observable 0\n  n0 observable 1\n}\n")
            .expect_err("duplicate node id");
        assert_eq!(error.line, 4);
    }

    #[test]
    fn from_text_rejects_dangling_edge_endpoint_at_its_line() {
        let error = Bloq::from_text("BLOQIR 1\ngraph {\n  n0 observable 0\n  n0 -> n7 order\n}\n")
            .expect_err("edge endpoint is not a node");
        assert_eq!(error.line, 4);
    }

    /// A single huge node id must be rejected up front — materializing its
    /// vacant slots would cost O(id), an OOM lever on untrusted input.
    #[test]
    fn from_text_rejects_implausibly_sparse_node_ids() {
        let error = Bloq::from_text("BLOQIR 1\ngraph {\n  n4294967295 observable 0\n}\n")
            .expect_err("sparse id space");
        assert_eq!(error.line, 3);
    }

    #[test]
    fn from_text_rejects_sparse_body_ids_without_allocating_to_the_id() {
        let error =
            Bloq::from_text("BLOQIR 1\ntemplate t0 {\n  body b4294967295 {\n  }\n}\ngraph {\n}\n")
                .expect_err("sparse body id");
        assert_eq!(error.line, 3);
    }

    #[test]
    fn from_text_rejects_empty_mpp_products() {
        let error = Bloq::from_text(
            "BLOQIR 1\ntemplate t0 {\n  circuit {\n    MPP _:m0\n  }\n}\ngraph {\n}\n",
        )
        .expect_err("empty MPP product");
        assert_eq!(error.line, 4);
    }

    #[test]
    fn zero_product_mpp_is_invalid_before_and_after_binary_round_trip() {
        let mut bloq = Bloq::from_text(
            "BLOQIR 1\ntemplate t0 {\ncircuit {\n}\n}\ngraph {\nn0 quantum {\ninstance i0 t0 @ (0,0)\n}\n}\n",
        ).unwrap();
        bloq.templates_mut()
            .make_mut(crate::TemplateId(0))
            .unwrap()
            .circuit
            .body_mut(bloq_circuit::BodyId(0))
            .unwrap()
            .ops_mut()
            .push(bloq_circuit::Op::MPP {
                products: vec![],
                measurements: vec![],
            });
        let restored = Bloq::from_binary(&bloq.to_binary()).unwrap();
        for program in [&bloq, &restored] {
            assert!(matches!(
                program.validate(),
                Err(BloqValidationError::InvalidTemplateCircuit {
                    source: NodeTemplateInstanceMergeError::EmptyMpp,
                    ..
                })
            ));
            Bloq::from_text(&program.to_text()).unwrap_err();
        }
    }

    #[test]
    fn cyclic_repeat_from_text_validates_as_an_error() {
        let bloq = Bloq::from_text(
            "BLOQIR 1\ntemplate t0 {\n  circuit {\n    REPEAT 2 b0\n  }\n}\ngraph {\n  n0 quantum {\n    instance i0 t0 @ (0,0)\n  }\n}\n",
        )
        .expect("cyclic repeat is semantic, not syntax");
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::InvalidTemplateCircuit {
                source: NodeTemplateInstanceMergeError::CyclicBody(bloq_circuit::BodyId(0)),
                ..
            }
        ));
    }

    /// One measurement id bound to two different qubits must be a parse
    /// error, not the `MeasRegistry::reserve` panic.
    #[test]
    fn from_text_rejects_conflicting_measurement_records() {
        let error = Bloq::from_text(
            "BLOQIR 1\ntemplate t0 {\n  circuit {\n    M (0,0):m0\n  }\n  \
             meas m0 @ (1,1)\n}\ngraph {\n}\n",
        )
        .expect_err("conflicting measurement qubits");
        assert_eq!(error.line, 6);
    }

    #[test]
    fn from_text_rejects_conflicting_implied_measurements_at_op_line() {
        let error = Bloq::from_text(
            "BLOQIR 1\ntemplate t0 {\n  circuit {\n    M (0,0):m0 (1,0):m0\n  }\n}\ngraph {\n}\n",
        )
        .expect_err("one implied id cannot name two qubits");

        assert_eq!(error.line, 4);
    }

    #[test]
    fn from_text_rejects_exhausted_measurement_outputs() {
        let cases = [
            (
                "BLOQIR 1\ntemplate t0 {\n  meas m4294967295 @ (0,0)\n}\ngraph {\n}\n",
                3,
            ),
            (
                "BLOQIR 1\ntemplate t0 {\n  circuit {\n    M (0,0):m4294967295\n  }\n}\ngraph {\n}\n",
                4,
            ),
            (
                "BLOQIR 1\ntemplate t0 {\n  circuit {\n    MPP X(0,0):m4294967295\n  }\n}\ngraph {\n}\n",
                4,
            ),
        ];

        for (source, line) in cases {
            let error = Bloq::from_text(source).expect_err("mMAX cannot be registered");
            assert_eq!(error.line, line);
            assert!(
                error
                    .to_string()
                    .contains("exhausts the measurement id space")
            );
        }
    }

    #[test]
    fn from_text_rejects_overflowing_instance_translation() {
        let error = Bloq::from_text(
            "BLOQIR 1\ntemplate t0 {\n  circuit {\n    H (2147483647,0)\n  }\n}\ngraph {\n  n0 quantum {\n    instance i0 t0 @ (1,0)\n  }\n}\n",
        )
        .expect_err("translated coordinate exceeds i32");

        assert!(error.to_string().contains("overflows after offset"));
    }

    /// Id sigils are digits-only; `u32::from_str`'s leading `+` is malformed.
    #[test]
    fn from_text_rejects_plus_prefixed_ids() {
        let error = Bloq::from_text("BLOQIR 1\ngraph {\n  n+0 observable 0\n}\n")
            .expect_err("`n+0` is not a node id");
        assert_eq!(error.line, 3);
    }

    #[test]
    fn from_text_rejects_plus_prefixed_correction_and_parity_ids() {
        let cases = [
            (
                "BLOQIR 1\ntemplate t0 {\n  circuit {\n    CPAULI X[m+0]@(0,0)\n  }\n}\ngraph {\n}\n",
                4,
                "expected `m<N>`",
            ),
            (
                "BLOQIR 1\ntemplate t0 {\n  loop(b0,s0) init s+0 next 0\n}\ngraph {\n}\n",
                3,
                "expected `s<N>`",
            ),
            (
                "BLOQIR 1\ntemplate t0 {\n  detector s+0\n}\ngraph {\n}\n",
                3,
                "expected `s<N>`",
            ),
        ];

        for (source, line, expected) in cases {
            let error = Bloq::from_text(source).expect_err("plus-prefixed id is malformed");
            assert_eq!(error.line, line);
            assert!(error.to_string().contains(expected));
        }
    }
}
