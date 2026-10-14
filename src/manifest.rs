//! The loom-swarm manifest: fixed-size BLAKE3 pieces plus an optional tensor-extent
//! sidecar (GGUF or safetensors; formats are feature-gated and plug in through `TensorExtent`). The manifest is the unit of trust — every transport
//! (torrent, HTTP, future RDMA) verifies pieces against it, and the semantic
//! layer maps tensors onto piece indices for partial pulls.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::OnceLock;

pub const FORMAT: &str = "loom-swarm/1";
pub const DEFAULT_PIECE_MIB: u64 = 16;

#[derive(Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub file_name: String,
    pub file_size: u64,
    pub piece_size: u64,
    /// blake3 over the concatenated piece hashes — the artifact's swarm identity.
    pub root: String,
    pub pieces: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gguf: Option<GgufMeta>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safetensors: Option<SafetensorsMeta>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tensors: Vec<TensorEntry>,
    /// Explicit tensor-parallel shards for formats without a GGUF sidecar (e.g. strixite's .strixw): "r/n" -> the
    /// [offset, length] byte ranges rank r of n reads, recorded from the engine's own loader. Not part of the root:
    /// adding shards never changes an artifact's swarm identity, and older binaries ignore the field.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub shards: BTreeMap<String, Vec<[u64; 2]>>,
    /// Virtual artifact: the [offset, length] ranges of the file on disk (`disk_size` bytes) that, concatenated in
    /// order, make the byte stream the pieces cover (`file_size` = the sum of the lengths). A tensor-parallel rank's
    /// slice of a weights file whose ranks interleave finely (strixite .strixw: 6.4 KB runs) is one artifact of exactly
    /// its bytes; a receiver writes each piece back at its real offsets into a sparse file of `disk_size`. Empty: the
    /// pieces cover the file itself.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layout: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub disk_size: u64,
    #[serde(skip)]
    prefix: OnceLock<Vec<u64>>,
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

#[derive(Serialize, Deserialize)]
pub struct GgufMeta {
    pub version: u32,
    pub alignment: u64,
    pub data_offset: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
}

/// One tensor's byte range, as a format reader reports it. This is the whole contract a format module has to
/// meet: a `parse(path)` that yields these, or `None` when the file is not that format.
pub struct TensorExtent {
    pub name: String,
    /// Absolute file offset.
    pub offset: u64,
    /// Bytes to read to materialize the tensor. GGUF reports padded offset deltas (alignment slack included);
    /// safetensors reports exact sizes.
    pub size: u64,
}

#[derive(Serialize, Deserialize)]
pub struct SafetensorsMeta {
    pub header_size: u64,
    pub data_offset: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub offset: u64,
    /// Padded extent (includes alignment slack up to the next tensor).
    pub size: u64,
    pub piece_first: u64,
    pub piece_last: u64,
}

impl Manifest {
    /// Where the data section starts when the artifact has a tensor sidecar: everything before it is header, and
    /// every tensor-parallel rank holds it.
    pub fn data_offset(&self) -> Option<u64> {
        if let Some(g) = &self.gguf {
            return Some(g.data_offset);
        }
        self.safetensors.as_ref().map(|s| s.data_offset)
    }

    pub fn n_pieces(&self) -> u64 {
        self.file_size.div_ceil(self.piece_size)
    }

    /// Size of the file on disk: `file_size`, or `disk_size` for a virtual (layout) artifact.
    pub fn on_disk_size(&self) -> u64 {
        if self.layout.is_empty() { self.file_size } else { self.disk_size }
    }

    pub fn piece_len(&self, i: u64) -> u64 {
        (self.file_size - i * self.piece_size).min(self.piece_size)
    }

    /// The (file offset, length) segments holding virtual bytes [voff, voff + len).
    pub fn segments(&self, voff: u64, len: u64) -> Vec<(u64, u64)> {
        if self.layout.is_empty() {
            return vec![(voff, len)];
        }
        let pre = self.prefix.get_or_init(|| {
            let mut p = Vec::with_capacity(self.layout.len() + 1);
            let mut at = 0u64;
            p.push(0);
            for r in &self.layout {
                at += r[1];
                p.push(at);
            }
            p
        });
        let mut out = Vec::new();
        let (mut v, end) = (voff, voff + len);
        let mut k = pre.partition_point(|&x| x <= v).saturating_sub(1);
        while v < end && k < self.layout.len() {
            let [off, l] = self.layout[k];
            let into = v - pre[k];
            let take = (l - into).min(end - v);
            out.push((off + into, take));
            v += take;
            k += 1;
        }
        out
    }

    /// Piece i's bytes from `f` (the file on disk) into buf; returns the piece length.
    pub fn read_piece(&self, f: &File, i: u64, buf: &mut [u8]) -> std::io::Result<usize> {
        let len = self.piece_len(i) as usize;
        let mut at = 0usize;
        for (off, l) in self.segments(i * self.piece_size, len as u64) {
            f.read_exact_at(&mut buf[at..at + l as usize], off)?;
            at += l as usize;
        }
        Ok(len)
    }

    /// Piece i's bytes back to their places in `f`.
    pub fn write_piece(&self, f: &File, i: u64, data: &[u8]) -> std::io::Result<()> {
        let mut at = 0usize;
        for (off, l) in self.segments(i * self.piece_size, data.len() as u64) {
            f.write_all_at(&data[at..at + l as usize], off)?;
            at += l as usize;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Manifest> {
        let m: Manifest = serde_json::from_reader(std::io::BufReader::new(
            File::open(path).with_context(|| format!("open {}", path.display()))?,
        ))?;
        if m.format != FORMAT {
            bail!("manifest format {} (expected {})", m.format, FORMAT);
        }
        if m.pieces.len() as u64 != m.n_pieces() {
            bail!(
                "manifest carries {} piece hashes but geometry implies {}",
                m.pieces.len(),
                m.n_pieces()
            );
        }
        Ok(m)
    }
}

/// Hash every piece of `path` with `threads` workers over contiguous ranges.
/// Plain positional reads — no mmap — so the crate stays free of unsafe code
/// and cold-cache behaviour matches how a transport actually reads pieces.
pub fn hash_pieces(path: &Path, piece_size: u64, threads: usize) -> Result<Vec<[u8; 32]>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_size = file.metadata()?.len();
    if file_size == 0 {
        bail!("{}: empty file", path.display());
    }
    let n_pieces = file_size.div_ceil(piece_size) as usize;
    let mut hashes = vec![[0u8; 32]; n_pieces];
    let per = n_pieces.div_ceil(threads.max(1));

    std::thread::scope(|s| -> Result<()> {
        let mut handles = Vec::new();
        for (w, chunk) in hashes.chunks_mut(per).enumerate() {
            let file = &file;
            handles.push(s.spawn(move || -> Result<()> {
                let mut buf = vec![0u8; piece_size as usize];
                for (i, out) in chunk.iter_mut().enumerate() {
                    let idx = (w * per + i) as u64;
                    let off = idx * piece_size;
                    let len = (file_size - off).min(piece_size) as usize;
                    file.read_exact_at(&mut buf[..len], off)
                        .with_context(|| format!("read piece {idx}"))?;
                    *out = *blake3::hash(&buf[..len]).as_bytes();
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join().expect("hasher thread panicked")?;
        }
        Ok(())
    })?;
    Ok(hashes)
}

pub fn root_of(hashes: &[[u8; 32]]) -> String {
    let mut h = blake3::Hasher::new();
    for p in hashes {
        h.update(p);
    }
    h.finalize().to_hex().to_string()
}

pub fn build(path: &Path, piece_size: u64, threads: usize) -> Result<Manifest> {
    let file_size = std::fs::metadata(path)?.len();
    let hashes = hash_pieces(path, piece_size, threads)?;
    let root = root_of(&hashes);

    let (gguf, safetensors, extents) = detect_format(path)?;
    let tensors = extents
        .iter()
        .map(|t| TensorEntry {
            name: t.name.clone(),
            offset: t.offset,
            size: t.size,
            piece_first: t.offset / piece_size,
            piece_last: if t.size == 0 { t.offset / piece_size } else { (t.offset + t.size - 1) / piece_size },
        })
        .collect();

    Ok(Manifest {
        format: FORMAT.to_string(),
        file_name: path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        file_size,
        piece_size,
        root,
        pieces: hashes.iter().map(hex::encode).collect(),
        gguf,
        safetensors,
        tensors,
        shards: BTreeMap::new(),
        layout: Vec::new(),
        disk_size: 0,
        prefix: OnceLock::new(),
    })
}

/// Try each enabled format reader in turn; the first that recognizes the file supplies the sidecar.
#[allow(unreachable_code, unused_variables)]
fn detect_format(path: &Path) -> Result<(Option<GgufMeta>, Option<SafetensorsMeta>, Vec<TensorExtent>)> {
    #[cfg(feature = "gguf")]
    if let Some(info) = crate::gguf::parse(path)? {
        let meta = GgufMeta {
            version: info.version,
            alignment: info.alignment,
            data_offset: info.data_offset,
            architecture: info.architecture,
            model_name: info.model_name,
        };
        return Ok((Some(meta), None, info.tensors));
    }
    #[cfg(feature = "safetensors")]
    if let Some(info) = crate::safetensors::parse(path)? {
        let meta = SafetensorsMeta { header_size: info.header_size, data_offset: info.data_offset, metadata: info.metadata };
        return Ok((None, Some(meta), info.tensors));
    }
    Ok((None, None, Vec::new()))
}

/// A virtual artifact over `ranges` of `path` (coalesced; see `Manifest::layout`), hashed piece by piece.
pub fn build_layout(path: &Path, ranges: Vec<[u64; 2]>, piece_size: u64, threads: usize) -> Result<Manifest> {
    let disk_size = std::fs::metadata(path)?.len();
    let layout = coalesce(ranges);
    if layout.is_empty() {
        bail!("{}: empty layout", path.display());
    }
    if let Some(r) = layout.iter().find(|r| r[0] + r[1] > disk_size) {
        bail!("{}: layout range [{}, +{}) past the end ({disk_size} bytes)", path.display(), r[0], r[1]);
    }
    let mut m = Manifest {
        format: FORMAT.to_string(),
        file_name: path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
        file_size: layout.iter().map(|r| r[1]).sum(),
        piece_size,
        root: String::new(),
        pieces: Vec::new(),
        gguf: None,
        safetensors: None,
        tensors: Vec::new(),
        shards: BTreeMap::new(),
        layout,
        disk_size,
        prefix: OnceLock::new(),
    };
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let n = m.n_pieces() as usize;
    let mut hashes = vec![[0u8; 32]; n];
    let per = n.div_ceil(threads.max(1));
    std::thread::scope(|s| -> Result<()> {
        let mut hs = Vec::new();
        for (w, chunk) in hashes.chunks_mut(per).enumerate() {
            let (f, m) = (&f, &m);
            hs.push(s.spawn(move || -> Result<()> {
                let mut buf = vec![0u8; m.piece_size as usize];
                for (j, out) in chunk.iter_mut().enumerate() {
                    let i = (w * per + j) as u64;
                    let len = m.read_piece(f, i, &mut buf).with_context(|| format!("read piece {i}"))?;
                    *out = *blake3::hash(&buf[..len]).as_bytes();
                }
                Ok(())
            }));
        }
        for h in hs {
            h.join().expect("hasher thread panicked")?;
        }
        Ok(())
    })?;
    m.root = root_of(&hashes);
    m.pieces = hashes.iter().map(hex::encode).collect();
    Ok(m)
}

/// Sort and coalesce [offset, length] ranges (overlapping or touching ranges merge), dropping empty ones.
pub fn coalesce(mut r: Vec<[u64; 2]>) -> Vec<[u64; 2]> {
    r.retain(|x| x[1] > 0);
    r.sort_unstable();
    let mut out: Vec<[u64; 2]> = Vec::with_capacity(r.len());
    for [off, len] in r {
        if let Some(last) = out.last_mut() {
            if off <= last[0] + last[1] {
                last[1] = last[1].max(off + len - last[0]);
                continue;
            }
        }
        out.push([off, len]);
    }
    out
}

/// Compress a sorted piece-index list into "a-b,c,d-e" form for humans.
pub fn ranges(mut idx: Vec<u64>) -> String {
    idx.sort_unstable();
    idx.dedup();
    let mut out = String::new();
    let mut i = 0;
    while i < idx.len() {
        let start = idx[i];
        let mut end = start;
        while i + 1 < idx.len() && idx[i + 1] == end + 1 {
            i += 1;
            end = idx[i];
        }
        if !out.is_empty() {
            out.push(',');
        }
        if start == end {
            out.push_str(&start.to_string());
        } else {
            out.push_str(&format!("{start}-{end}"));
        }
        i += 1;
    }
    out
}

/// Default tensor-name pattern for `--shard`: the routed-expert tensors that a
/// tensor-parallel engine splits by expert (ds4 TP: rank r of n holds experts
/// [r*E/n, (r+1)*E/n), stored contiguously, so the slice is a byte range).
pub const DEFAULT_SHARD_MATCH: &str = "_exps";

/// Slack added around each shard boundary. Extents are padded offset deltas,
/// so the true expert boundary can sit a few alignment bytes off size*r/n;
/// widening by this much makes the boundary piece land on both sides.
const SHARD_SLACK: u64 = 1 << 16;

/// Pieces a tensor-parallel rank `rank` of `n` must hold: the whole file header,
/// every tensor not matching `pats` in full, and only the rank's contiguous
/// 1/n byte slice of each matching tensor. Measured against ds4 TP=2 page-cache
/// residency on DeepSeek-V4-Flash IQ2XXS (2026-10-03): this reproduces what each
/// rank touches.
pub fn shard_pieces(m: &Manifest, rank: u64, n: u64, pats: &[&str]) -> Result<Vec<u64>> {
    if n == 0 || rank >= n {
        bail!("shard {rank}/{n} out of range");
    }
    if let Some(rs) = m.shards.get(&format!("{rank}/{n}")) {
        // explicit byte ranges (non-GGUF formats): exactly the pieces they touch
        let ps = m.piece_size;
        let mut p: Vec<u64> = Vec::new();
        for &[off, len] in rs {
            if len == 0 {
                continue;
            }
            if off + len > m.file_size {
                bail!("shard {rank}/{n}: range [{off}, +{len}) past the end of the file ({})", m.file_size);
            }
            p.extend(off / ps..=(off + len - 1) / ps);
        }
        p.sort_unstable();
        p.dedup();
        return Ok(p);
    }
    let Some(data_offset) = m.data_offset() else {
        bail!("--shard needs a tensor sidecar (GGUF or safetensors) and this manifest has none");
    };
    let ps = m.piece_size;
    let mut p: Vec<u64> = (0..=data_offset.saturating_sub(1) / ps).collect();
    let mut matched = 0usize;
    for t in &m.tensors {
        if t.size == 0 {
            continue;
        }
        if pats.iter().any(|q| t.name.contains(q)) {
            matched += 1;
            let lo = (t.offset + t.size * rank / n).saturating_sub(SHARD_SLACK).max(t.offset);
            let hi = (t.offset + t.size * (rank + 1) / n + SHARD_SLACK).min(t.offset + t.size);
            p.extend(lo / ps..=(hi - 1) / ps);
        } else {
            p.extend(t.piece_first..=t.piece_last);
        }
    }
    if matched == 0 {
        bail!("--shard: no tensors match {pats:?}");
    }
    p.retain(|&i| i < m.n_pieces());
    p.sort_unstable();
    p.dedup();
    Ok(p)
}
