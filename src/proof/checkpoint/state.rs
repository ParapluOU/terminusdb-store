//! The checkpoint artifact (what the chain records and a verifier holds) and the prover's
//! checkpoint state (the artifact plus the witness it commits to).

use std::io;

use ark_serialize::CanonicalSerialize;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::{Field, Group};
use sha2::{Digest, Sha256};

use super::fixed_commit::{append_segmented, segment_count, DICT_SEGMENT_LEN, MAX_DICT_LEN};
use super::smt::{Smt, SmtKey, NS_NV_INDEX, NS_PRED_INDEX, NS_TRIPLE_INDEX};
use super::term_tree::{object_leaf_parts, term_leaf, TermTree, NS_NODE_VALUE, NS_PREDICATE, TERM_TREE_DEPTH};
use super::IdTriple;
use crate::proof::argument::StoreProverContext;
use crate::proof::commitment::{
    node_value_op_value, node_value_sort_key, predicate_sort_key, DictFacts, NODE_DATATYPE_TAG,
};
use crate::proof::dict::{hash_object, hash_term};
use crate::proof::{invalid, CanonicalObject, ProofHash};

const CHECKPOINT_ROOT_DOMAIN: &[u8] = b"terminus-store/pow/checkpoint-root/v3";
const TRIPLE_KEY_DOMAIN: &[u8] = b"terminus-store/pow/checkpoint/triple-key/v1";

/// A checkpoint `C_k`: commitments to the materialized state of the layer chain at head `X_k`.
///
/// Everything is in the CHAIN's id space (ids are stable along the chain), so the dictionary
/// columns, the term tree and the term index only grow, and only the present-triple columns are
/// re-committed per checkpoint. `checkpoint_root` binds every field; it is what the chain records
/// next to `head_root`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointArtifact {
    /// `X_k`: the head layer's `resolution_root`.
    pub head_root: ProofHash,
    /// The head layer's `commitment_root` (the next delta's `parent_state_root`).
    pub head_commitment_root: ProofHash,
    /// The checkpoint this one was proven from (`None` for a genesis checkpoint).
    pub prev_checkpoint_root: Option<ProofHash>,
    /// Number of present triples.
    pub present_count: u64,
    /// Node/value dictionary size (ids `1..=node_value_count`).
    pub node_value_count: u64,
    /// Predicate dictionary size (ids `1..=predicate_count`).
    pub predicate_count: u64,
    /// Present S, P, O id columns, SPO-sorted, each committed with `commit_column` zero-padded to
    /// [`present_domain_log`]; identities when empty.
    pub present_commitments: [ArkGT; 3],
    /// Node/value dictionary columns `hash`, `value`, `datatype` (row `id - 1`), one fixed-layout
    /// commitment per segment of [`DICT_SEGMENT_LEN`] rows (`fixed_commit`), segment-major.
    pub nv_dict_segments: Vec<[ArkGT; 3]>,
    /// Predicate dictionary `hash` column (row `id - 1`), segmented likewise.
    pub pred_dict_segments: Vec<ArkGT>,
    /// Id-indexed append-only term trees.
    pub nv_term_tree_root: ProofHash,
    pub pred_term_tree_root: ProofHash,
    /// Keyed term indexes (`hash(term) -> id`).
    pub nv_term_index_root: ProofHash,
    pub pred_term_index_root: ProofHash,
    /// Keyed triple index over the present triples.
    pub triple_index_root: ProofHash,
    /// Whole-dictionary facts.
    pub facts: DictFacts,
    /// `H(domain, every field above)`.
    pub checkpoint_root: ProofHash,
}

fn put_gt(hasher: &mut Sha256, value: &ArkGT) -> io::Result<()> {
    let mut bytes = Vec::new();
    value
        .serialize_compressed(&mut bytes)
        .map_err(|error| invalid(format!("checkpoint commitment serialization failed: {error}")))?;
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(&bytes);
    Ok(())
}

impl CheckpointArtifact {
    /// `checkpoint_root = H(domain, X_k, head commitment root, previous checkpoint root, counts,
    /// the commitments, the Merkle roots, facts)`.
    pub fn compute_root(&self) -> io::Result<ProofHash> {
        let mut hasher = Sha256::new();
        hasher.update(CHECKPOINT_ROOT_DOMAIN);
        hasher.update(self.head_root);
        hasher.update(self.head_commitment_root);
        match &self.prev_checkpoint_root {
            None => hasher.update([0u8]),
            Some(root) => {
                hasher.update([1u8]);
                hasher.update(root);
            }
        }
        for count in [self.present_count, self.node_value_count, self.predicate_count] {
            hasher.update(count.to_be_bytes());
        }
        for commitment in &self.present_commitments {
            put_gt(&mut hasher, commitment)?;
        }
        hasher.update((self.nv_dict_segments.len() as u64).to_be_bytes());
        for segment in &self.nv_dict_segments {
            for commitment in segment {
                put_gt(&mut hasher, commitment)?;
            }
        }
        hasher.update((self.pred_dict_segments.len() as u64).to_be_bytes());
        for commitment in &self.pred_dict_segments {
            put_gt(&mut hasher, commitment)?;
        }
        for root in [
            &self.nv_term_tree_root,
            &self.pred_term_tree_root,
            &self.nv_term_index_root,
            &self.pred_term_index_root,
            &self.triple_index_root,
        ] {
            hasher.update(root);
        }
        hasher.update([self.facts.to_byte()]);
        Ok(hasher.finalize().into())
    }

    /// Structural self-consistency: the root recomputes, the counts fit their structures, and an
    /// empty relation/dictionary carries identity commitments.
    pub fn validate(&self) -> io::Result<()> {
        if self.compute_root()? != self.checkpoint_root {
            return Err(invalid("checkpoint root does not recompute from the artifact"));
        }
        let tree_cap = (1u64 << TERM_TREE_DEPTH) - 1;
        if self.node_value_count > tree_cap.min(MAX_DICT_LEN) || self.predicate_count > tree_cap.min(MAX_DICT_LEN) {
            return Err(invalid("checkpoint dictionary exceeds its committed structures"));
        }
        if self.nv_dict_segments.len() != segment_count(self.node_value_count, DICT_SEGMENT_LEN)
            || self.pred_dict_segments.len() != segment_count(self.predicate_count, DICT_SEGMENT_LEN)
        {
            return Err(invalid("checkpoint dictionary segment count does not match its length"));
        }
        present_domain_log(self.present_count)?;
        let identity = <ArkGT as Group>::identity();
        if self.present_count == 0 && self.present_commitments.iter().any(|c| *c != identity) {
            return Err(invalid("empty checkpoint relation must carry identity commitments"));
        }
        if self.present_count > 0 && (self.node_value_count == 0 || self.predicate_count == 0) {
            return Err(invalid("checkpoint has triples but an empty dictionary"));
        }
        Ok(())
    }

    /// An artifact of an empty dictionary — the starting point the dictionary of a from-scratch
    /// checkpoint is appended to. Only its counts and segments are meaningful.
    pub(crate) fn empty_dictionary() -> Self {
        let identity = <ArkGT as Group>::identity();
        CheckpointArtifact {
            head_root: [0; 32],
            head_commitment_root: [0; 32],
            prev_checkpoint_root: None,
            present_count: 0,
            node_value_count: 0,
            predicate_count: 0,
            present_commitments: [identity; 3],
            nv_dict_segments: Vec::new(),
            pred_dict_segments: Vec::new(),
            nv_term_tree_root: [0; 32],
            pred_term_tree_root: [0; 32],
            nv_term_index_root: [0; 32],
            pred_term_index_root: [0; 32],
            triple_index_root: [0; 32],
            facts: DictFacts::default(),
            checkpoint_root: [0; 32],
        }
    }

    /// Every field except the chaining (`prev_checkpoint_root`, `checkpoint_root`): two
    /// checkpoints of the same head commit the same state iff their digests agree.
    pub fn state_digest(&self) -> io::Result<ProofHash> {
        let mut copy = self.clone();
        copy.prev_checkpoint_root = None;
        copy.compute_root()
    }
}

/// Largest present relation a checkpoint commits (the PF4 record cap).
pub const MAX_CHECKPOINT_TRIPLES: u64 = 10_000_000;

/// Log of the padded domain the present columns are committed at: `next_pow2(n).max(2)`. For
/// `n >= 2` this is exactly the domain a base layer commits its materialized columns at; for
/// `n = 1` the two commitments coincide (a single row-0, column-0 entry).
pub fn present_domain_log(n: u64) -> io::Result<usize> {
    if n > MAX_CHECKPOINT_TRIPLES {
        return Err(invalid("checkpoint present relation exceeds the record cap"));
    }
    Ok((n as usize).next_power_of_two().max(2).trailing_zeros() as usize)
}

/// The present S, P, O columns zero-padded to `2^present_domain_log(n)`.
pub fn present_columns(present: &[IdTriple]) -> io::Result<[Vec<ArkFr>; 3]> {
    let size = 1usize << present_domain_log(present.len() as u64)?;
    let mut columns = [vec![ArkFr::zero(); size], vec![ArkFr::zero(); size], vec![ArkFr::zero(); size]];
    for (row, triple) in present.iter().enumerate() {
        for (column, id) in columns.iter_mut().zip(triple.iter()) {
            column[row] = ArkFr::from_u64(*id);
        }
    }
    Ok(columns)
}

/// Commit the present columns (identities when empty).
pub fn commit_present(context: &StoreProverContext<'_>, present: &[IdTriple]) -> io::Result<[ArkGT; 3]> {
    if present.is_empty() {
        let identity = <ArkGT as Group>::identity();
        return Ok([identity; 3]);
    }
    let [s, p, o] = present_columns(present)?;
    Ok([context.commit_column(&s)?, context.commit_column(&p)?, context.commit_column(&o)?])
}

/// The triple-index key of an id triple.
pub fn triple_key(triple: &IdTriple) -> SmtKey {
    let mut hasher = Sha256::new();
    hasher.update(TRIPLE_KEY_DOMAIN);
    for id in triple {
        hasher.update(id.to_be_bytes());
    }
    hasher.finalize().into()
}

/// The value the triple index stores for a present triple.
pub const TRIPLE_PRESENT: u64 = 1;

/// One node/value dictionary row: `(hash, value, datatype)`.
pub fn nv_dict_row(object: &CanonicalObject) -> io::Result<[ArkFr; 3]> {
    let tag = match object {
        CanonicalObject::Value { datatype, .. } => *datatype as u8 as u64,
        CanonicalObject::Node(_) => NODE_DATATYPE_TAG,
    };
    Ok([hash_object(object), node_value_op_value(object)?, ArkFr::from_u64(tag)])
}

/// The node/value dictionary segment of `objects` as three columns.
pub fn nv_dict_segment(objects: &[CanonicalObject]) -> io::Result<[Vec<ArkFr>; 3]> {
    let mut out = [Vec::with_capacity(objects.len()), Vec::with_capacity(objects.len()), Vec::with_capacity(objects.len())];
    for object in objects {
        let row = nv_dict_row(object)?;
        for (column, value) in out.iter_mut().zip(row) {
            column.push(value);
        }
    }
    Ok(out)
}

/// The predicate dictionary segment (hash column).
pub fn pred_dict_segment(predicates: &[Vec<u8>]) -> Vec<ArkFr> {
    predicates.iter().map(|p| hash_term(p)).collect()
}

pub fn nv_term_leaf(id: u64, object: &CanonicalObject) -> ProofHash {
    let (tag, bytes) = object_leaf_parts(object);
    term_leaf(NS_NODE_VALUE, id, tag, bytes)
}

pub fn pred_term_leaf(id: u64, predicate: &[u8]) -> ProofHash {
    term_leaf(NS_PREDICATE, id, NODE_DATATYPE_TAG as u8, predicate)
}

/// The prover's checkpoint: the artifact plus everything it commits to.
#[derive(Clone, Debug)]
pub struct CheckpointState {
    pub artifact: CheckpointArtifact,
    /// Present id triples, strictly SPO-sorted.
    pub(super) present: Vec<IdTriple>,
    /// Node/value terms, `nv_terms[id - 1]`.
    pub(super) nv_terms: Vec<CanonicalObject>,
    /// Predicate IRIs, `pred_terms[id - 1]`.
    pub(super) pred_terms: Vec<Vec<u8>>,
    pub(super) nv_tree: TermTree,
    pub(super) pred_tree: TermTree,
    pub(super) nv_index: Smt,
    pub(super) pred_index: Smt,
    pub(super) triple_index: Smt,
}

impl CheckpointState {
    /// Build a checkpoint of a materialized state from scratch.
    pub fn from_parts(
        context: &StoreProverContext<'_>,
        head_root: ProofHash,
        head_commitment_root: ProofHash,
        prev_checkpoint_root: Option<ProofHash>,
        mut present: Vec<IdTriple>,
        nv_terms: Vec<CanonicalObject>,
        pred_terms: Vec<Vec<u8>>,
    ) -> io::Result<Self> {
        present.sort();
        if present.windows(2).any(|w| w[0] == w[1]) {
            return Err(invalid("checkpoint present relation has a duplicate triple"));
        }
        let nv_count = nv_terms.len() as u64;
        let pred_count = pred_terms.len() as u64;
        if present
            .iter()
            .any(|t| t[0] == 0 || t[1] == 0 || t[2] == 0 || t[0] > nv_count || t[2] > nv_count || t[1] > pred_count)
        {
            return Err(invalid("checkpoint present triple id lies outside the dictionary"));
        }
        let empty = CheckpointArtifact::empty_dictionary();
        let (nv_dict_segments, pred_dict_segments) = append_dictionary(&empty, &nv_terms, &pred_terms)?;
        let mut nv_tree = TermTree::new(NS_NODE_VALUE);
        for (i, object) in nv_terms.iter().enumerate() {
            nv_tree.push(nv_term_leaf(i as u64 + 1, object))?;
        }
        let mut pred_tree = TermTree::new(NS_PREDICATE);
        for (i, predicate) in pred_terms.iter().enumerate() {
            pred_tree.push(pred_term_leaf(i as u64 + 1, predicate))?;
        }
        let nv_index = Smt::from_entries(
            NS_NV_INDEX,
            nv_terms.iter().enumerate().map(|(i, o)| (node_value_sort_key(o), i as u64 + 1)),
        )?;
        let pred_index = Smt::from_entries(
            NS_PRED_INDEX,
            pred_terms.iter().enumerate().map(|(i, p)| (predicate_sort_key(p), i as u64 + 1)),
        )?;
        let triple_index =
            Smt::from_entries(NS_TRIPLE_INDEX, present.iter().map(|t| (triple_key(t), TRIPLE_PRESENT)))?;
        let facts = DictFacts::from_objects(nv_terms.iter());
        let present_commitments = commit_present(context, &present)?;
        let mut artifact = CheckpointArtifact {
            head_root,
            head_commitment_root,
            prev_checkpoint_root,
            present_count: present.len() as u64,
            node_value_count: nv_count,
            predicate_count: pred_count,
            present_commitments,
            nv_dict_segments,
            pred_dict_segments,
            nv_term_tree_root: nv_tree.root(),
            pred_term_tree_root: pred_tree.root(),
            nv_term_index_root: nv_index.root(),
            pred_term_index_root: pred_index.root(),
            triple_index_root: triple_index.root(),
            facts,
            checkpoint_root: [0; 32],
        };
        artifact.checkpoint_root = artifact.compute_root()?;
        artifact.validate()?;
        Ok(Self { artifact, present, nv_terms, pred_terms, nv_tree, pred_tree, nv_index, pred_index, triple_index })
    }

    /// The present id triples (SPO-sorted).
    pub fn present(&self) -> &[IdTriple] {
        &self.present
    }

    /// The node/value term of chain id `id`.
    pub fn node_value(&self, id: u64) -> Option<&CanonicalObject> {
        id.checked_sub(1).and_then(|i| self.nv_terms.get(i as usize))
    }

    /// The term-tree opening of chain id `id` in the node/value (`predicate = false`) or predicate
    /// namespace.
    pub(crate) fn term_opening(&self, predicate: bool, id: u64) -> io::Result<crate::proof::commitment::DictOpening> {
        let (tag, bytes, tree) = if predicate {
            let iri = self.predicate(id).ok_or_else(|| invalid("checkpoint has no predicate with this id"))?;
            (NODE_DATATYPE_TAG as u8, iri.to_vec(), &self.pred_tree)
        } else {
            let object = self.node_value(id).ok_or_else(|| invalid("checkpoint has no node/value with this id"))?;
            let (tag, bytes) = object_leaf_parts(object);
            (tag, bytes.to_vec(), &self.nv_tree)
        };
        Ok(crate::proof::commitment::DictOpening {
            id,
            datatype_tag: tag,
            term_bytes: bytes,
            position: (id - 1) as usize,
            siblings: tree.siblings(id - 1)?,
        })
    }

    /// The keyed term-index path of `key` in the node/value (`predicate = false`) or predicate
    /// index (a membership or a non-membership path).
    pub(crate) fn index_path(&self, predicate: bool, key: &SmtKey) -> super::smt::SmtPath {
        if predicate { self.pred_index.path(key) } else { self.nv_index.path(key) }
    }

    /// The predicate IRI of chain id `id`.
    pub fn predicate(&self, id: u64) -> Option<&[u8]> {
        id.checked_sub(1).and_then(|i| self.pred_terms.get(i as usize)).map(|p| p.as_slice())
    }
}

/// The checkpoint dictionary commitments after appending new terms: the previous segments with
/// each new row committed at its segment-local offset (see `fixed_commit`).
pub fn append_dictionary(
    prev: &CheckpointArtifact,
    new_nv: &[CanonicalObject],
    new_pred: &[Vec<u8>],
) -> io::Result<(Vec<[ArkGT; 3]>, Vec<ArkGT>)> {
    append_dictionary_segmented(prev, new_nv, new_pred, DICT_SEGMENT_LEN)
}

/// [`append_dictionary`] at an explicit segment length (tests exercise segment boundaries with a
/// short length; production is [`DICT_SEGMENT_LEN`]).
pub(crate) fn append_dictionary_segmented(
    prev: &CheckpointArtifact,
    new_nv: &[CanonicalObject],
    new_pred: &[Vec<u8>],
    segment_len: usize,
) -> io::Result<(Vec<[ArkGT; 3]>, Vec<ArkGT>)> {
    let nv_offset = prev.node_value_count as usize;
    let rows = nv_dict_segment(new_nv)?;
    let mut columns: [Vec<ArkGT>; 3] = Default::default();
    for (c, column) in columns.iter_mut().enumerate() {
        let old: Vec<ArkGT> = prev.nv_dict_segments.iter().map(|s| s[c]).collect();
        *column = append_segmented(&old, nv_offset, &rows[c], segment_len)?;
    }
    let nv = (0..columns[0].len()).map(|k| [columns[0][k], columns[1][k], columns[2][k]]).collect();
    let pred = append_segmented(
        &prev.pred_dict_segments,
        prev.predicate_count as usize,
        &pred_dict_segment(new_pred),
        segment_len,
    )?;
    Ok((nv, pred))
}
