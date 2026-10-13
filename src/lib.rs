//! loom-swarm: content-addressed model distribution for diskless GPU nodes.
//!
//! Three layers, each usable on its own:
//!
//! - [`manifest`]: fixed-size BLAKE3 pieces and the manifest that is the artifact's identity (plus optional
//!   per-rank shard ranges and virtual "layout" artifacts). Every transport verifies against it.
//! - [`gguf`]: the one format-specific piece, a minimal GGUF header reader that yields tensor extents so a pull
//!   can be expressed in tensor names. Other formats plug in by providing the same shape of extent list.
//! - [`net`]: the piece transport (GET piece ranges, HAVE bitmaps, work stealing across sources) and the
//!   seeder that serves what it holds while still pulling.
//!
//! The `loom-swarm` binary in this crate wires the three into verbs; see `docs/ENGINES.md` for what an
//! inference engine has to provide to use each tier.

pub mod gguf;
pub mod manifest;
pub mod net;
