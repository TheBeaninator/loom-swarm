//! Minimal GGUF header reader — just enough to emit tensor extents.
//!
//! Extent sizes are derived from offset deltas (sorted), not from a dtype size
//! table. That folds each tensor's alignment padding into its extent, which is
//! exactly what a fetch plan wants: the padded range is what you must read to
//! materialize the tensor. Byte-exact tensor sizes are NOT this module's job.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

pub use crate::manifest::TensorExtent;

pub struct GgufInfo {
    pub version: u32,
    pub alignment: u64,
    pub data_offset: u64,
    pub architecture: Option<String>,
    pub model_name: Option<String>,
    pub tensors: Vec<TensorExtent>,
}

const MAGIC: &[u8; 4] = b"GGUF";
// A hostile or corrupt header must not make us allocate the moon.
const MAX_STRING: u64 = 1 << 24;
const MAX_COUNT: u64 = 1 << 24;

/// Returns Ok(None) when the file is not GGUF (wrong magic) — the manifest
/// layer treats that as "no semantic sidecar", not an error.
pub fn parse(path: &std::path::Path) -> Result<Option<GgufInfo>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_size = file.metadata()?.len();
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    if r.read_exact(&mut magic).is_err() || &magic != MAGIC {
        return Ok(None);
    }
    let version = read_u32(&mut r)?;
    if version < 2 {
        bail!("gguf version {version} predates 64-bit counts; not supported");
    }
    let tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;
    if tensor_count > MAX_COUNT || kv_count > MAX_COUNT {
        bail!("implausible gguf header: {tensor_count} tensors / {kv_count} kvs");
    }

    let mut alignment: u64 = 32;
    let mut architecture = None;
    let mut model_name = None;
    for _ in 0..kv_count {
        let key = read_string(&mut r)?;
        let ty = read_u32(&mut r)?;
        let captured = read_value(&mut r, ty)?;
        match (key.as_str(), captured) {
            ("general.alignment", Value::Int(n)) if n > 0 => alignment = n,
            ("general.architecture", Value::Str(s)) => architecture = Some(s),
            ("general.name", Value::Str(s)) => model_name = Some(s),
            _ => {}
        }
    }

    let mut raw: Vec<(String, u64)> = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut r)?;
        let n_dims = read_u32(&mut r)?;
        if n_dims > 8 {
            bail!("tensor {name}: implausible n_dims {n_dims}");
        }
        for _ in 0..n_dims {
            read_u64(&mut r)?;
        }
        let _dtype = read_u32(&mut r)?;
        let rel_offset = read_u64(&mut r)?;
        raw.push((name, rel_offset));
    }

    let header_end = r.stream_position()?;
    let data_offset = header_end.div_ceil(alignment) * alignment;

    raw.sort_by_key(|(_, off)| *off);
    let mut tensors = Vec::with_capacity(raw.len());
    for i in 0..raw.len() {
        let (name, rel) = &raw[i];
        let abs = data_offset + rel;
        let end = match raw.get(i + 1) {
            Some((_, next_rel)) => data_offset + next_rel,
            None => file_size,
        };
        if end < abs {
            bail!("tensor {name}: extent end {end} precedes offset {abs}");
        }
        tensors.push(TensorExtent { name: name.clone(), offset: abs, size: end - abs });
    }

    // Seek back cleanliness is irrelevant — the reader is dropped here.
    let _ = r.seek(SeekFrom::Start(0));
    Ok(Some(GgufInfo { version, alignment, data_offset, architecture, model_name, tensors }))
}

enum Value {
    Int(u64),
    Str(String),
    Other,
}

fn read_value<R: Read>(r: &mut R, ty: u32) -> Result<Value> {
    Ok(match ty {
        0 | 1 | 7 => Value::Int(read_bytes_as_u64(r, 1)?),
        2 | 3 => Value::Int(read_bytes_as_u64(r, 2)?),
        4 | 5 => Value::Int(read_bytes_as_u64(r, 4)?),
        6 => {
            skip(r, 4)?;
            Value::Other
        }
        8 => Value::Str(read_string(r)?),
        9 => {
            let elem_ty = read_u32(r)?;
            let count = read_u64(r)?;
            if count > MAX_COUNT {
                bail!("implausible gguf array length {count}");
            }
            for _ in 0..count {
                read_value(r, elem_ty)?;
            }
            Value::Other
        }
        10 | 11 => Value::Int(read_bytes_as_u64(r, 8)?),
        12 => {
            skip(r, 8)?;
            Value::Other
        }
        other => bail!("unknown gguf kv type {other}"),
    })
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_bytes_as_u64<R: Read>(r: &mut R, n: usize) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b[..n])?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)?;
    if len > MAX_STRING {
        bail!("implausible gguf string length {len}");
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip<R: Read>(r: &mut R, n: u64) -> Result<()> {
    std::io::copy(&mut r.take(n), &mut std::io::sink())?;
    Ok(())
}
