//! The compact binary Bloq IR exchange format.
//!
//! Layout: a 4-byte magic (`b"BLOQ"`), one version byte, then the whole
//! [`Bloq`] postcard-encoded (varint integers, length-prefixed sequences).
//! The payload is the IR's serde representation — self-contained (templates,
//! shared detector bundles, instance ids, side tables, `boundary_flows`), and it round-trips all
//! provenance channels, so a save/load loses nothing.
//!
//! Node ids are preserved exactly, holes included: [`SubGraph`] serializes live
//! node-id pairs and ordered live edges, then reconstructs vacant node slots.
//! Classical definitions are stored once per graph level and referenced by its
//! nodes; decoding restores sharing without changing invocation identity.
//! The deterministic schedule (keyed on node slot index) is identical after a
//! round trip; vacant edge slots are not preserved.
//!
//! The version stays at 1 during rapid development, even when the serialized
//! shape changes. A reader rejects other version bytes, but the shared version
//! does not promise compatibility with older pre-release payloads. Postcard is
//! not self-describing, so artifacts must match the current IR field layout.

use crate::Bloq;

/// Compact consecutive records of one instance without changing their order.
/// Human-readable serde retains the ordinary array of record objects.
pub(crate) mod instance_measurements {
    use std::fmt;

    use serde::{
        Deserialize, Deserializer, Serialize, Serializer,
        de::{Error, SeqAccess, Visitor},
        ser::SerializeSeq,
    };

    use crate::{InstanceMeasurement, TemplateInstanceId};

    pub(crate) fn serialize<S: Serializer>(
        records: &[InstanceMeasurement],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            return records.serialize(serializer);
        }
        let runs = records.chunk_by(|a, b| a.instance == b.instance);
        // A flat integer stream: total records, then (run length, instance,
        // local ids...) per run. The total permits one decoded allocation.
        let mut sequence =
            serializer.serialize_seq(Some(1 + 2 * runs.clone().count() + records.len()))?;
        sequence.serialize_element(&records.len())?;
        for run in runs {
            sequence.serialize_element(&run.len())?;
            sequence.serialize_element(&run[0].instance)?;
            for record in run {
                sequence.serialize_element(&record.measurement)?;
            }
        }
        sequence.end()
    }

    struct Records;

    impl<'de> Visitor<'de> for Records {
        type Value = Vec<InstanceMeasurement>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a record count followed by instance runs")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
            let count: usize = sequence
                .next_element()?
                .ok_or_else(|| A::Error::custom("missing record count"))?;
            // Like serde's Vec decoder, cap speculative allocation from an
            // untrusted count at 1 MiB. Further growth needs actual records.
            let mut records =
                Vec::with_capacity(count.min((1 << 20) / size_of::<InstanceMeasurement>()));
            while let Some(length) = sequence.next_element::<usize>()? {
                if length == 0 || length > count - records.len() {
                    return Err(A::Error::custom("invalid record run length"));
                }
                let instance: TemplateInstanceId = sequence
                    .next_element()?
                    .ok_or_else(|| A::Error::custom("record run needs an instance"))?;
                for _ in 0..length {
                    let measurement = sequence
                        .next_element()?
                        .ok_or_else(|| A::Error::custom("record run is truncated"))?;
                    records.push(InstanceMeasurement {
                        instance,
                        measurement,
                    });
                }
            }
            if records.len() != count {
                return Err(A::Error::custom("record count does not match its runs"));
            }
            Ok(records)
        }
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<InstanceMeasurement>, D::Error> {
        if deserializer.is_human_readable() {
            return Vec::deserialize(deserializer);
        }
        deserializer.deserialize_seq(Records)
    }
}

/// File magic prefixing every binary `.bloq` artifact.
pub const BLOQ_BINARY_MAGIC: [u8; 4] = *b"BLOQ";

/// Pre-release binary format version; stays at 1 during rapid development.
pub const BLOQ_BINARY_VERSION: u8 = 1;

/// The conventional file extension for the binary format.
pub const BLOQ_BINARY_EXTENSION: &str = "bloq";

/// Why a byte buffer failed to decode as a binary `.bloq` artifact.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BinaryDecodeError {
    /// Input does not start with the Bloq binary magic bytes.
    #[error("not a bloq binary artifact: missing `BLOQ` magic")]
    BadMagic,
    /// Input ends before its version byte or payload.
    #[error("truncated bloq binary artifact: magic without a version byte")]
    Truncated,
    /// Input uses an unsupported binary format version.
    #[error(
        "unsupported bloq binary version {found} (this reader supports only \
         version {BLOQ_BINARY_VERSION}; there is no cross-version compatibility)"
    )]
    UnsupportedVersion {
        /// Version read from the input.
        found: u8,
    },
    /// Postcard could not decode the payload.
    #[error("malformed bloq binary payload")]
    Payload(#[source] postcard::Error),
    /// Bytes remain after the decoded program.
    #[error("{extra} trailing byte(s) after the bloq binary payload")]
    TrailingData {
        /// Number of trailing bytes.
        extra: usize,
    },
}

impl Bloq {
    /// Encode this program as a binary `.bloq` artifact.
    ///
    /// # Panics
    ///
    /// Panics if the in-memory IR violates postcard's sequence contract.
    pub fn to_binary(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(1024);
        bytes.extend_from_slice(&BLOQ_BINARY_MAGIC);
        bytes.push(BLOQ_BINARY_VERSION);
        postcard::to_extend(self, bytes)
            .expect("postcard serialization into a Vec cannot fail for serde-derived IR types")
    }

    /// Decode a binary `.bloq` artifact.
    ///
    /// Decoding rejects structurally malformed payloads (trailing bytes,
    /// duplicate node ids, dangling edges, an implausibly sparse node-id
    /// space, unsorted registry/PauliMap entries) but does not check IR
    /// semantics — run [`Bloq::validate`] on untrusted input.
    ///
    /// # Errors
    ///
    /// Returns [`BinaryDecodeError`] for an invalid header, unsupported
    /// version, malformed payload, or trailing data.
    pub fn from_binary(bytes: &[u8]) -> Result<Self, BinaryDecodeError> {
        let payload = bytes
            .strip_prefix(&BLOQ_BINARY_MAGIC)
            .ok_or(BinaryDecodeError::BadMagic)?;
        let (&version, payload) = payload.split_first().ok_or(BinaryDecodeError::Truncated)?;
        if version != BLOQ_BINARY_VERSION {
            return Err(BinaryDecodeError::UnsupportedVersion { found: version });
        }
        // `take_from_bytes` rather than `from_bytes`: a payload followed by
        // trailing junk (padding, a concatenated second artifact) must be
        // rejected, not silently decoded as its first program.
        let (bloq, rest) =
            postcard::take_from_bytes(payload).map_err(BinaryDecodeError::Payload)?;
        if !rest.is_empty() {
            return Err(BinaryDecodeError::TrailingData { extra: rest.len() });
        }
        Ok(bloq)
    }
}

#[cfg(test)]
mod tests {
    use crate::{BloqNodeId, test_fixture::sample_bloq};

    use super::*;

    #[test]
    fn retry_region_keeps_its_binary_tag_and_removed_regions_are_rejected() {
        use crate::{ClassicalExpr, RegionNode, SubGraph};

        let region = RegionNode::RepeatUntilSuccess {
            body: SubGraph::new(),
            restart_condition: ClassicalExpr::Const(false),
            restart_source: None,
        };
        let bytes = postcard::to_extend(&region, Vec::new()).unwrap();
        assert_eq!(bytes[0], 1, "preserve the pre-removal retry tag");
        let restored: RegionNode = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(postcard::to_extend(&restored, Vec::new()).unwrap(), bytes);

        let mut program = Bloq::new();
        program.add_node(crate::BloqNode::region(region));
        let mut artifact = program.to_binary();
        let offsets = artifact
            .windows(bytes.len())
            .enumerate()
            .filter_map(|(offset, payload)| (payload == bytes).then_some(offset))
            .collect::<Vec<_>>();
        assert_eq!(offsets.len(), 1, "the single serialized region");
        Bloq::from_binary(&artifact).unwrap();
        artifact[offsets[0]] = 0;
        assert!(matches!(
            Bloq::from_binary(&artifact),
            Err(BinaryDecodeError::Payload(_))
        ));

        // The removed conditional region used tag 0. Reject it immediately,
        // before any old body or predicate can be read as a retry payload.
        postcard::from_bytes::<RegionNode>(&[0]).unwrap_err();
        serde_json::from_str::<RegionNode>(r#"{"Cond":{"condition":{"Const":false},"body":{}}}"#)
            .unwrap_err();
        Bloq::from_text("BLOQIR 1\ngraph {\n n0 cond 0 {\n body {\n }\n }\n}").unwrap_err();
    }

    #[test]
    fn record_runs_preserve_order_duplicates_and_full_width_ids() {
        use crate::{ClassicalNode, InstanceMeasurement, TemplateInstanceId};

        for pairs in [
            vec![],
            vec![(0, 0)],
            vec![
                (300, 8),
                (300, 2),
                (300, 2),
                (0, 0),
                (300, 1),
                (u32::MAX, u32::MAX),
            ],
            (0..260).map(|i| (i / 130, i)).collect(),
        ] {
            let measurements = pairs
                .into_iter()
                .map(|(instance, measurement)| InstanceMeasurement {
                    instance: TemplateInstanceId(instance),
                    measurement,
                })
                .collect::<Vec<_>>();
            let node = ClassicalNode::observable_fragment(measurements.clone(), Vec::new());
            let bytes = postcard::to_extend(&node, Vec::new()).unwrap();
            let restored: ClassicalNode = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(restored, node);
            assert_eq!(postcard::to_extend(&restored, Vec::new()).unwrap(), bytes);
            assert_eq!(
                serde_json::to_value(&node).unwrap(),
                serde_json::json!({"Observable": {"index": null, "measurements": measurements, "operators": []}})
            );
            assert_eq!(
                serde_json::from_value::<ClassicalNode>(serde_json::to_value(&node).unwrap())
                    .unwrap(),
                node
            );
            for length in 0..bytes.len() {
                postcard::from_bytes::<ClassicalNode>(&bytes[..length]).unwrap_err();
            }
        }
        #[derive(serde::Deserialize)]
        struct Records(#[serde(with = "super::instance_measurements")] Vec<InstanceMeasurement>);
        assert!(
            postcard::from_bytes::<Records>(&postcard::to_extend(&vec![0u64], Vec::new()).unwrap())
                .unwrap()
                .0
                .is_empty()
        );
        // Count mismatches, empty runs, and partial runs are malformed.
        for stream in [
            vec![],
            vec![1],
            vec![1, 0],
            vec![1, 2, 0, 0, 1],
            vec![1, 1],
            vec![1, 1, 0],
            vec![0, 1, 0, 0],
            vec![u64::MAX],
        ] {
            let bytes = postcard::to_extend(&stream, Vec::new()).unwrap();
            assert!(postcard::from_bytes::<Records>(&bytes).is_err());
        }
    }

    #[test]
    fn binary_round_trip_preserves_ids_edges_and_bytes() {
        let bloq = sample_bloq();
        let bytes = bloq.to_binary();
        assert_eq!(&bytes[..5], b"BLOQ\x01");
        let restored = Bloq::from_binary(&bytes).expect("round trip decodes");

        // Node ids survive exactly, hole included (the fixture removes n1).
        let original_ids: Vec<BloqNodeId> = bloq.node_ids().collect();
        let restored_ids: Vec<BloqNodeId> = restored.node_ids().collect();
        assert_eq!(original_ids, restored_ids);
        assert!(!original_ids.contains(&BloqNodeId(1)));

        // The deterministic schedule (SEM-ORD) is keyed on slot ids.
        assert_eq!(
            bloq.deterministic_emit_order().expect("acyclic"),
            restored.deterministic_emit_order().expect("acyclic"),
        );

        assert_eq!(bloq.edge_count(), restored.edge_count());
        assert_eq!(bloq.templates().len(), restored.templates().len());

        // Re-encoding the decoded program is byte-identical (stable encoding).
        assert_eq!(bytes, restored.to_binary());

        // Editing a boxed seam after cloning still detaches its graph level.
        let mut edited = restored.clone();
        let mut seams = 0;
        edited.top_mut().for_each_edge_mut(|edge| {
            if let crate::BloqEdge::Quantum(quantum) = edge {
                quantum.pipes.clear();
                seams += 1;
            }
        });
        assert!(seams > 0);
        assert_eq!(bytes, restored.to_binary());
        assert_ne!(bytes, edited.to_binary());
    }

    #[test]
    fn from_binary_rejects_bad_magic_and_version() {
        assert!(matches!(
            Bloq::from_binary(b"nope"),
            Err(BinaryDecodeError::BadMagic)
        ));
        assert!(matches!(
            Bloq::from_binary(b"BLOQ"),
            Err(BinaryDecodeError::Truncated)
        ));
        let mut bytes = sample_bloq().to_binary();
        bytes[4] = 99;
        assert!(matches!(
            Bloq::from_binary(&bytes),
            Err(BinaryDecodeError::UnsupportedVersion { found: 99 })
        ));
    }

    #[test]
    fn from_binary_rejects_trailing_bytes() {
        // A padded or concatenated buffer must not decode as its first program.
        let mut bytes = sample_bloq().to_binary();
        bytes.extend_from_slice(b"junk");
        assert!(matches!(
            Bloq::from_binary(&bytes),
            Err(BinaryDecodeError::TrailingData { extra: 4 })
        ));
    }
}
