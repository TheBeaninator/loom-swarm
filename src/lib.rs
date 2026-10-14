//! loom-swarm: content-addressed model distribution for diskless GPU nodes.
//!
//! Three layers, each usable on its own:
//!
//! - [`manifest`]: fixed-size BLAKE3 pieces and the manifest that is the artifact's identity (plus optional
//!   per-rank shard ranges and virtual "layout" artifacts). Every transport verifies against it.
//! - format readers, one per feature: [`gguf`] and [`safetensors`] each read a header into tensor extents
//!   (`manifest::TensorExtent`) so a pull can be expressed in tensor names. Another format plugs in by providing
//!   the same shape of reader and a line in `manifest::detect_format`.
//! - [`net`]: the piece transport (GET piece ranges, HAVE bitmaps, work stealing across sources) and the
//!   seeder that serves what it holds while still pulling.
//!
//! The `loom-swarm` binary in this crate wires the three into verbs; see `docs/ENGINES.md` for what an
//! inference engine has to provide to use each tier.

#[cfg(feature = "gguf")]
pub mod gguf;
pub mod manifest;
pub mod net;
#[cfg(feature = "safetensors")]
pub mod safetensors;
