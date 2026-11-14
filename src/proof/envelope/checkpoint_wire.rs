//! Wire formats of the proven-checkpoint artifacts (`crate::proof::checkpoint`): the checkpoint
//! artifact, the inductive checkpoint proof and the genesis proof. Each is framed like the store's
//! other proof formats — `MAGIC ‖ VERSION ‖ body ‖ SHA-256 checksum` — under the shared size cap
//! and the global decoded-item budget. Decoding is parsing only; trust checks live in
//! `verify_checkpoint` / `verify_genesis_checkpoint`.

use std::convert::TryInto;
use std::io;

use dory_pcs::backends::arkworks::ArkGT;
use sha2::{Digest, Sha256};

use super::wire::{invalid, wire_struct, Reader, Wire, CHECKSUM_BYTES, MAX_QUERY_ENVELOPE_BYTES};
use crate::proof::checkpoint::smt::{SmtDeleteProof, SmtPath, SmtSibling, SmtTerminal};
use crate::proof::checkpoint::{
    CheckpointArtifact, CheckpointProof, CheckpointViewProof, DeltaUpdateProof, GenesisProof, IdTriple,
    PaddingProof, MAX_CHECKPOINT_TRIPLES,
};
use crate::proof::commitment::DictFacts;

/// Framing magic of a [`CheckpointArtifact`] — permanent; never change these bytes.
pub(crate) const CHECKPOINT_ARTIFACT_MAGIC: &[u8; 12] = b"TDBWQCKPT\0\0\0";
/// Framing magic of a [`CheckpointProof`] — permanent; never change these bytes.
pub(crate) const CHECKPOINT_PROOF_MAGIC: &[u8; 12] = b"TDBWQCKPP\0\0\0";
/// Framing magic of a [`GenesisProof`] — permanent; never change these bytes.
pub(crate) const GENESIS_PROOF_MAGIC: &[u8; 12] = b"TDBWQCKPG\0\0\0";
/// Framing magic of a [`CheckpointViewProof`] — permanent; never change these bytes.
pub(crate) const CHECKPOINT_VIEW_MAGIC: &[u8; 12] = b"TDBWQCKPV\0\0\0";
/// Wire-format versions (accepted exactly; bump on any layout change).
/// The artifact is v2 since its dictionary columns became segment lists.
pub(crate) const CHECKPOINT_ARTIFACT_VERSION: u32 = 2;
pub(crate) const CHECKPOINT_PROOF_VERSION: u32 = 1;
pub(crate) const GENESIS_PROOF_VERSION: u32 = 1;
pub(crate) const CHECKPOINT_VIEW_VERSION: u32 = 1;

impl Wire for [u64; 3] {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        for v in self {
            v.put(out)?;
        }
        Ok(())
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok([u64::get(reader)?, u64::get(reader)?, u64::get(reader)?])
    }
}

impl Wire for [ArkGT; 3] {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        for v in self {
            v.put(out)?;
        }
        Ok(())
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok([ArkGT::get(reader)?, ArkGT::get(reader)?, ArkGT::get(reader)?])
    }
}

impl Wire for DictFacts {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        self.to_byte().put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        DictFacts::from_byte(u8::get(reader)?)
    }
}

impl Wire for SmtTerminal {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            SmtTerminal::Empty => 0u8.put(out),
            SmtTerminal::Leaf { key, value } => {
                1u8.put(out)?;
                key.put(out)?;
                value.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(SmtTerminal::Empty),
            1 => Ok(SmtTerminal::Leaf { key: Wire::get(reader)?, value: Wire::get(reader)? }),
            _ => Err(invalid("invalid checkpoint index terminal tag")),
        }
    }
}

impl Wire for SmtSibling {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            SmtSibling::Leaf { key, value } => {
                0u8.put(out)?;
                key.put(out)?;
                value.put(out)
            }
            SmtSibling::Node { left, right } => {
                1u8.put(out)?;
                left.put(out)?;
                right.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(SmtSibling::Leaf { key: Wire::get(reader)?, value: Wire::get(reader)? }),
            1 => Ok(SmtSibling::Node { left: Wire::get(reader)?, right: Wire::get(reader)? }),
            _ => Err(invalid("invalid checkpoint index sibling tag")),
        }
    }
}

wire_struct!(SmtPath { siblings, terminal });
wire_struct!(SmtDeleteProof { path, sibling });
wire_struct!(DeltaUpdateProof { additions, removals, removal_paths, addition_paths });
wire_struct!(PaddingProof { sumcheck, opening });
wire_struct!(CheckpointProof {
    deltas,
    nv_tree_frontier,
    pred_tree_frontier,
    nv_index_paths,
    pred_index_paths,
    working_log,
    link,
    prev_present_evals,
    prev_present_opening,
    next_present_evals,
    next_present_opening,
    padding,
    sorted,
});
wire_struct!(CheckpointViewProof {
    nv_commitments,
    pred_commitments,
    nv_evals,
    pred_evals,
    nv_opening,
    pred_opening,
    fixed_opening,
});
wire_struct!(CheckpointArtifact {
    head_root,
    head_commitment_root,
    prev_checkpoint_root,
    present_count,
    node_value_count,
    predicate_count,
    present_commitments,
    nv_dict_segments,
    pred_dict_segments,
    nv_term_tree_root,
    pred_term_tree_root,
    nv_term_index_root,
    pred_term_index_root,
    triple_index_root,
    facts,
    checkpoint_root,
});

fn frame(magic: &[u8; 12], version: u32, body: impl FnOnce(&mut Vec<u8>) -> io::Result<()>) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(magic);
    out.extend_from_slice(&version.to_le_bytes());
    body(&mut out)?;
    if out.len().checked_add(CHECKSUM_BYTES).map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES) {
        return Err(invalid("checkpoint wire object too large"));
    }
    let checksum = Sha256::digest(&out);
    out.extend_from_slice(&checksum);
    Ok(out)
}

fn unframe<T>(
    bytes: &[u8],
    magic: &[u8; 12],
    version: u32,
    what: &str,
    body: impl FnOnce(&mut Reader<'_>) -> io::Result<T>,
) -> io::Result<T> {
    if bytes.len() < magic.len() + 4 + CHECKSUM_BYTES || bytes.len() > MAX_QUERY_ENVELOPE_BYTES {
        return Err(invalid(format!("invalid {what} size")));
    }
    let (payload, checksum) = bytes.split_at(bytes.len() - CHECKSUM_BYTES);
    if Sha256::digest(payload).as_slice() != checksum {
        return Err(invalid(format!("{what} checksum mismatch")));
    }
    let mut reader = Reader { bytes: payload, pos: 0, decoded_items: 0 };
    if reader.take(magic.len())? != magic {
        return Err(invalid(format!("unknown {what} magic")));
    }
    if u32::from_le_bytes(reader.take(4)?.try_into().unwrap()) != version {
        return Err(invalid(format!("unsupported {what} version")));
    }
    let value = body(&mut reader)?;
    reader.finish()?;
    Ok(value)
}

impl CheckpointArtifact {
    /// Frame the artifact for shipping.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        frame(CHECKPOINT_ARTIFACT_MAGIC, CHECKPOINT_ARTIFACT_VERSION, |out| self.put(out))
    }

    /// Parse a framed artifact (no trust checks; `validate` / the verifiers check it).
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        unframe(bytes, CHECKPOINT_ARTIFACT_MAGIC, CHECKPOINT_ARTIFACT_VERSION, "checkpoint artifact", |r| Self::get(r))
    }
}

impl CheckpointProof {
    /// Frame the proof for shipping.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        frame(CHECKPOINT_PROOF_MAGIC, CHECKPOINT_PROOF_VERSION, |out| self.put(out))
    }

    /// Parse a framed proof (no trust checks).
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        unframe(bytes, CHECKPOINT_PROOF_MAGIC, CHECKPOINT_PROOF_VERSION, "checkpoint proof", |r| Self::get(r))
    }
}

impl CheckpointViewProof {
    /// Frame the view proof for shipping.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        frame(CHECKPOINT_VIEW_MAGIC, CHECKPOINT_VIEW_VERSION, |out| self.put(out))
    }

    /// Parse a framed view proof (no trust checks; `verify_checkpoint_view` checks it).
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        unframe(bytes, CHECKPOINT_VIEW_MAGIC, CHECKPOINT_VIEW_VERSION, "checkpoint view proof", |r| Self::get(r))
    }
}

impl GenesisProof {
    /// Frame the genesis proof: the present triples as a counted run of 24-byte records (outside
    /// the per-item budget, bounded by [`MAX_CHECKPOINT_TRIPLES`] and the byte cap instead).
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        if self.present.len() as u64 > MAX_CHECKPOINT_TRIPLES {
            return Err(invalid("genesis proof exceeds the record cap"));
        }
        frame(GENESIS_PROOF_MAGIC, GENESIS_PROOF_VERSION, |out| {
            (self.present.len() as u64).put(out)?;
            for triple in &self.present {
                triple.put(out)?;
            }
            Ok(())
        })
    }

    /// Parse a framed genesis proof (no trust checks).
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        unframe(bytes, GENESIS_PROOF_MAGIC, GENESIS_PROOF_VERSION, "genesis proof", |reader| {
            let count = u64::get(reader)?;
            let remaining = (reader.bytes.len() - reader.pos) as u64;
            if count > MAX_CHECKPOINT_TRIPLES || count.checked_mul(24).map_or(true, |b| b != remaining) {
                return Err(invalid("genesis proof record count disagrees with its size"));
            }
            let mut present: Vec<IdTriple> = Vec::with_capacity(count as usize);
            for _ in 0..count {
                present.push(Wire::get(reader)?);
            }
            Ok(GenesisProof { present })
        })
    }
}
