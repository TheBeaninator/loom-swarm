//! Minimal safetensors header reader — just enough to emit tensor extents.
//!
//! Layout: a little-endian u64 header length N, N bytes of JSON, then the data section. Each tensor entry
//! carries `data_offsets: [begin, end)` relative to the data section, so extents are byte-exact: unlike GGUF
//! there is no alignment slack to fold in. `__metadata__` is the one non-tensor key and is carried through
//! as strings. Multi-file checkpoints (`model-00001-of-00009.safetensors`) are one artifact per file.

use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub use crate::manifest::TensorExtent;

pub struct SafetensorsInfo {
    /// Length of the JSON header in bytes (the leading u64).
    pub header_size: u64,
    /// Absolute file offset of the data section: 8 + header_size.
    pub data_offset: u64,
    /// The `__metadata__` map, string values only.
    pub metadata: BTreeMap<String, String>,
    /// Sorted by offset. Byte-exact.
    pub tensors: Vec<TensorExtent>,
}

// The reference implementation refuses headers over 100 MB; so do we, and anything larger is simply "not this format".
const MAX_HEADER: u64 = 100 << 20;

/// Returns Ok(None) when the file is not safetensors (first word not a plausible header length, or the header is
/// not a JSON object) — the manifest layer treats that as "no semantic sidecar", not an error.
pub fn parse(path: &Path) -> Result<Option<SafetensorsInfo>> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_size = f.metadata()?.len();
    let mut lenb = [0u8; 8];
    if file_size < 8 || f.read_exact(&mut lenb).is_err() {
        return Ok(None);
    }
    let n = u64::from_le_bytes(lenb);
    if n == 0 || n > MAX_HEADER || 8 + n > file_size {
        return Ok(None);
    }
    let mut hdr = vec![0u8; n as usize];
    f.read_exact(&mut hdr)?;
    let Ok(text) = std::str::from_utf8(&hdr) else { return Ok(None) };
    if !text.trim_start().starts_with('{') {
        return Ok(None);
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else { return Ok(None) };
    let Some(obj) = v.as_object() else { return Ok(None) };

    let data_offset = 8 + n;
    let data_size = file_size - data_offset;
    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::with_capacity(obj.len());
    for (name, ent) in obj {
        if name == "__metadata__" {
            if let Some(m) = ent.as_object() {
                for (k, v) in m {
                    if let Some(s) = v.as_str() {
                        metadata.insert(k.clone(), s.to_owned());
                    }
                }
            }
            continue;
        }
        let offs = ent
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .with_context(|| format!("safetensors tensor {name}: no data_offsets"))?;
        let (Some(b), Some(e)) = (offs.first().and_then(|x| x.as_u64()), offs.get(1).and_then(|x| x.as_u64())) else {
            bail!("safetensors tensor {name}: data_offsets must be two integers");
        };
        if e < b || e > data_size {
            bail!("safetensors tensor {name}: data_offsets [{b}, {e}) outside the data section ({data_size} bytes)");
        }
        tensors.push(TensorExtent { name: name.clone(), offset: data_offset + b, size: e - b });
    }
    tensors.sort_by(|a, b| a.offset.cmp(&b.offset).then_with(|| a.name.cmp(&b.name)));
    Ok(Some(SafetensorsInfo { header_size: n, data_offset, metadata, tensors }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a safetensors file: `tensors` = (name, byte length), laid out in order with no gaps.
    fn write_fixture(path: &Path, tensors: &[(&str, u64)]) -> Result<u64> {
        let mut entries = Vec::new();
        let mut off = 0u64;
        for (name, len) in tensors {
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"F16\",\"shape\":[{}],\"data_offsets\":[{off},{}]}}",
                len / 2,
                off + len
            ));
            off += len;
        }
        entries.push("\"__metadata__\":{\"format\":\"pt\",\"n\":\"2\"}".to_string());
        let mut hdr = format!("{{{}}}", entries.join(",")).into_bytes();
        while hdr.len() % 8 != 0 {
            hdr.push(b' '); // the reference writer pads the header to 8 bytes with spaces
        }
        let mut out = (hdr.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hdr);
        let data_offset = out.len() as u64;
        out.extend((0..off).map(|i| (i * 2654435761u64 >> 16) as u8));
        std::fs::write(path, out)?;
        Ok(data_offset)
    }

    #[test]
    fn parses_header_and_sorts_extents() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("loom-swarm-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let f = dir.join("m.safetensors");
        let data_offset =
            write_fixture(&f, &[("model.layers.0.w", 4096), ("model.embed", 1024), ("model.layers.1._exps", 8192)])?;
        let info = parse(&f)?.expect("is safetensors");
        assert_eq!(info.data_offset, data_offset);
        assert_eq!(info.header_size, data_offset - 8);
        assert_eq!(info.metadata.get("format").map(String::as_str), Some("pt"));
        let names: Vec<&str> = info.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["model.layers.0.w", "model.embed", "model.layers.1._exps"]);
        assert_eq!(info.tensors[1].offset, data_offset + 4096);
        assert_eq!(info.tensors[1].size, 1024);
        assert_eq!(info.tensors[2].offset + info.tensors[2].size, std::fs::metadata(&f)?.len());

        // not this format: a GGUF-looking first word, and a small first word over non-JSON bytes
        let g = dir.join("x.gguf");
        std::fs::write(&g, b"GGUF\x03\x00\x00\x00rest of a header")?;
        assert!(parse(&g)?.is_none());
        let r = dir.join("x.bin");
        let mut junk = 16u64.to_le_bytes().to_vec();
        junk.extend_from_slice(&[0xffu8; 64]);
        std::fs::write(&r, junk)?;
        assert!(parse(&r)?.is_none());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn manifest_sidecar_and_shards() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("loom-swarm-st-m-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let f = dir.join("m.safetensors");
        let mib = 1u64 << 20;
        // header piece, a 1 MiB dense tensor, then 4 MiB of experts: TP 0/2 must take the header, the dense tensor
        // and the first half of the experts (plus boundary slack), never the far end.
        let data_offset = write_fixture(&f, &[("blk.0.attn", mib), ("blk.0.ffn_exps", 4 * mib)])?;
        let m = crate::manifest::build(&f, mib, 2)?;
        assert!(m.gguf.is_none());
        let st = m.safetensors.as_ref().expect("safetensors sidecar");
        assert_eq!(st.data_offset, data_offset);
        assert_eq!(m.tensors.len(), 2);
        assert_eq!(m.data_offset(), Some(data_offset));
        let n = m.n_pieces();
        let p0 = crate::manifest::shard_pieces(&m, 0, 2, &["_exps"])?;
        let p1 = crate::manifest::shard_pieces(&m, 1, 2, &["_exps"])?;
        assert!(p0.contains(&0) && p1.contains(&0), "both ranks hold the header piece");
        assert!(!p0.contains(&(n - 1)), "rank 0 does not fetch the last expert piece: {p0:?}");
        assert!(p1.contains(&(n - 1)));
        // JSON round trip keeps the sidecar
        let j = serde_json::to_string(&m)?;
        let m2: crate::manifest::Manifest = serde_json::from_str(&j)?;
        assert_eq!(m2.data_offset(), Some(data_offset));
        assert_eq!(crate::manifest::shard_pieces(&m2, 1, 2, &["_exps"])?, p1);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
