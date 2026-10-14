#![forbid(unsafe_code)]
//! `loom-swarm` — content-addressed model distribution (card afe883b66d).
//!
//! Verbs:
//!   loom-swarm manifest <file> [--piece-mib N] [--threads N] [--layout ranges] [-o out.json]
//!   loom-swarm verify   <file> --manifest m.json [--threads N]
//!   loom-swarm extents  --manifest m.json --match <substring> [--match ...] | --shard r/n
//!   loom-swarm shards   --manifest m.json --add r/n=<ranges file> [--add ...]   (explicit TP shards, formats without a sidecar)
//!   loom-swarm chunk    <file> --manifest m.json --store <dir>
//!   loom-swarm rehydrate --manifest m.json --store <dir> -o <file>
//!   loom-swarm serve    --listen addr:port --manifest m.json [--manifest ...]
//!   loom-swarm pull     --manifest m.json --from origin [--from peer ...] [--shard r/n] -o <file>
//!   loom-swarm seed     --want m.json=out[@r/n] [--want ...] --from origin [--from peer ...]
//!                       --listen addr:port [--ready <file>]   (serve while pulling, stay up)

use loom_swarm::manifest;
use loom_swarm::net;

use anyhow::{bail, Context, Result};
use manifest::Manifest;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }
    match args[1].as_str() {
        "manifest" => cmd_manifest(&args[2..]),
        "verify" => cmd_verify(&args[2..]),
        "extents" => cmd_extents(&args[2..]),
        "shards" => cmd_shards(&args[2..]),
        "chunk" => cmd_chunk(&args[2..]),
        "rehydrate" => cmd_rehydrate(&args[2..]),
        "serve" => cmd_serve(&args[2..]),
        "pull" => cmd_pull(&args[2..]),
        "seed" => cmd_seed(&args[2..]),
        other => {
            eprintln!("unknown verb: {other}");
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!(
        "loom-swarm — content-addressed model distribution
  manifest <file> [--piece-mib N] [--threads N] [--layout <file of \"offset length\" lines>] [-o out.json]
           --layout: a virtual artifact of just those byte ranges (e.g. one TP rank's reads of a weights file);
           a puller writes them back at their real offsets in a sparse file of the full size
  verify   <file> --manifest m.json [--threads N]
  extents  --manifest m.json --match <substring> [--match ...] | --shard r/n [--shard-match s]
  shards   --manifest m.json --add r/n=<file of \"offset length\" lines> [--add ...]
           record a rank's byte ranges (e.g. an engine's read log) so --shard / @r/n work on files without a tensor sidecar
  chunk    <file> --manifest m.json --store <dir>
  rehydrate --manifest m.json --store <dir> -o <file>
  serve    --listen <addr:port> --manifest m.json [--manifest ...] [--file <path>]
  pull     --manifest m.json --from <origin> [--from <peer> ...] [--conns K]
           [--match s ...] [--shard r/n [--shard-match s ...]] -o <file>
  seed     --want m.json=<out>[@r/n] [--want ...] --from <origin> [--from <peer> ...]
           --listen <addr:port> [--conns K] [--shard-match s ...] [--ready <file>]
           serves every held piece while pulling the rest; stays up as a seeder.
           The first --from is the origin (holds everything); peers may be partial.
           An existing <out> is resumed: pieces that already verify are kept."
    );
}

struct Opts {
    positional: Vec<String>,
    flags: Vec<(String, String)>,
}

fn parse(args: &[String], flags_taking_value: &[&str]) -> Result<Opts> {
    let mut positional = Vec::new();
    let mut flags = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some(name) = a.strip_prefix("--") {
            if !flags_taking_value.contains(&name) {
                bail!("unknown flag --{name}");
            }
            let val = args.get(i + 1).with_context(|| format!("--{name} needs a value"))?;
            flags.push((name.to_string(), val.clone()));
            i += 2;
        } else if a == "-o" {
            let val = args.get(i + 1).context("-o needs a value")?;
            flags.push(("o".to_string(), val.clone()));
            i += 2;
        } else {
            positional.push(a.clone());
            i += 1;
        }
    }
    Ok(Opts { positional, flags })
}

impl Opts {
    fn flag(&self, name: &str) -> Option<&str> {
        self.flags.iter().rev().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }
    fn flag_all(&self, name: &str) -> Vec<&str> {
        self.flags.iter().filter(|(n, _)| n == name).map(|(_, v)| v.as_str()).collect()
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(16)
}

fn gib_per_s(bytes: u64, secs: f64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64 / secs
}

fn cmd_manifest(args: &[String]) -> Result<()> {
    let o = parse(args, &["piece-mib", "threads", "layout"])?;
    let [file] = o.positional.as_slice() else { bail!("usage: manifest <file> ...") };
    let path = Path::new(file);
    let piece_size = o.flag("piece-mib").map(str::parse).transpose()?.unwrap_or(manifest::DEFAULT_PIECE_MIB) << 20;
    let threads = o.flag("threads").map(str::parse).transpose()?.unwrap_or_else(default_threads);

    let t = Instant::now();
    let m = match o.flag("layout") {
        Some(lf) => manifest::build_layout(path, read_ranges(lf)?, piece_size, threads)?,
        None => manifest::build(path, piece_size, threads)?,
    };
    let secs = t.elapsed().as_secs_f64();

    let out: PathBuf = match o.flag("o") {
        Some(p) => p.into(),
        None => PathBuf::from(format!("{file}.swarm.json")),
    };
    serde_json::to_writer_pretty(std::io::BufWriter::new(File::create(&out)?), &m)?;
    eprintln!(
        "{}: {} pieces of {} MiB, {} tensors, root {}",
        m.file_name,
        m.n_pieces(),
        piece_size >> 20,
        m.tensors.len(),
        &m.root[..16]
    );
    eprintln!(
        "hashed {:.2} GiB in {:.2}s = {:.2} GiB/s ({} threads) -> {}",
        m.file_size as f64 / (1u64 << 30) as f64,
        secs,
        gib_per_s(m.file_size, secs),
        threads,
        out.display()
    );
    Ok(())
}

fn cmd_verify(args: &[String]) -> Result<()> {
    let o = parse(args, &["manifest", "threads"])?;
    let [file] = o.positional.as_slice() else { bail!("usage: verify <file> --manifest m.json") };
    let m = Manifest::load(Path::new(o.flag("manifest").context("--manifest required")?))?;
    let threads = o.flag("threads").map(str::parse).transpose()?.unwrap_or_else(default_threads);

    let size = std::fs::metadata(file)?.len();
    if size != m.on_disk_size() {
        bail!("size mismatch: file {} vs manifest {}", size, m.on_disk_size());
    }
    let t = Instant::now();
    let hashes = if m.layout.is_empty() {
        manifest::hash_pieces(Path::new(file), m.piece_size, threads)?
    } else {
        // a virtual artifact: hash its pieces through the layout (one reader is plenty for a check)
        let f = File::open(file)?;
        let mut buf = vec![0u8; m.piece_size as usize];
        let mut hs = Vec::with_capacity(m.n_pieces() as usize);
        for i in 0..m.n_pieces() {
            let len = m.read_piece(&f, i, &mut buf)?;
            hs.push(*blake3::hash(&buf[..len]).as_bytes());
        }
        hs
    };
    let secs = t.elapsed().as_secs_f64();

    let bad: Vec<u64> = hashes
        .iter()
        .enumerate()
        .filter(|(i, h)| hex::encode(h) != m.pieces[*i])
        .map(|(i, _)| i as u64)
        .collect();
    eprintln!("verified {:.2} GiB in {:.2}s = {:.2} GiB/s", size as f64 / (1u64 << 30) as f64, secs, gib_per_s(size, secs));
    if bad.is_empty() {
        let root = manifest::root_of(&hashes);
        if root != m.root {
            bail!("pieces match but root differs — manifest is self-inconsistent");
        }
        eprintln!("OK: all {} pieces match, root {}", hashes.len(), &root[..16]);
        Ok(())
    } else {
        bail!("{} corrupt pieces: {}", bad.len(), manifest::ranges(bad));
    }
}

fn cmd_extents(args: &[String]) -> Result<()> {
    let o = parse(args, &["manifest", "match", "shard", "shard-match"])?;
    let m = Manifest::load(Path::new(o.flag("manifest").context("--manifest required")?))?;
    if let Some(sh) = o.flag("shard") {
        let p = select_pieces(&m, &[], Some(sh), &o.flag_all("shard-match"))?;
        let bytes: u64 = p.iter().map(|&i| (m.file_size - i * m.piece_size).min(m.piece_size)).sum();
        eprintln!("shard {sh}: {} of {} pieces, {:.2} GiB", p.len(), m.n_pieces(), bytes as f64 / (1u64 << 30) as f64);
        println!("{}", manifest::ranges(p));
        return Ok(());
    }
    let pats = o.flag_all("match");
    if pats.is_empty() {
        bail!("at least one --match <substring> required");
    }
    if m.tensors.is_empty() {
        bail!("manifest has no tensor sidecar (artifact was neither GGUF nor safetensors)");
    }
    let mut pieces = Vec::new();
    let mut bytes = 0u64;
    let mut count = 0usize;
    for t in &m.tensors {
        if pats.iter().any(|p| t.name.contains(p)) {
            println!("{}\toffset {}\tsize {}\tpieces {}-{}", t.name, t.offset, t.size, t.piece_first, t.piece_last);
            pieces.extend(t.piece_first..=t.piece_last);
            bytes += t.size;
            count += 1;
        }
    }
    if count == 0 {
        bail!("no tensors matched");
    }
    pieces.sort_unstable();
    pieces.dedup();
    let fetch_bytes = pieces.len() as u64 * m.piece_size;
    eprintln!(
        "\n{} tensors, {:.3} GiB of extents -> {} pieces ({:.3} GiB fetched, {:.1}% overfetch)\npieces: {}",
        count,
        bytes as f64 / (1u64 << 30) as f64,
        pieces.len(),
        fetch_bytes as f64 / (1u64 << 30) as f64,
        (fetch_bytes.saturating_sub(bytes)) as f64 * 100.0 / bytes.max(1) as f64,
        manifest::ranges(pieces)
    );
    Ok(())
}

/// "offset length" lines (an engine's read log), blank lines skipped.
fn read_ranges(file: &str) -> Result<Vec<[u64; 2]>> {
    let mut rs = Vec::new();
    for (i, line) in std::fs::read_to_string(file).with_context(|| format!("read {file}"))?.lines().enumerate() {
        let mut it = line.split_whitespace();
        match (it.next(), it.next(), it.next()) {
            (None, ..) => continue,
            (Some(off), Some(len), None) => rs.push([off.parse()?, len.parse()?]),
            _ => bail!("{file}:{}: expected \"offset length\", got {line:?}", i + 1),
        }
    }
    Ok(rs)
}

fn cmd_shards(args: &[String]) -> Result<()> {
    let o = parse(args, &["manifest", "add"])?;
    let mpath = Path::new(o.flag("manifest").context("--manifest required")?);
    let mut m = Manifest::load(mpath)?;
    let adds = o.flag_all("add");
    if adds.is_empty() {
        for (k, rs) in &m.shards {
            let bytes: u64 = rs.iter().map(|r| r[1]).sum();
            let p = manifest::shard_pieces(&m, k.split('/').next().unwrap_or("0").parse()?, k.split('/').nth(1).unwrap_or("1").parse()?, &[])?;
            println!("{k}: {} ranges, {:.2} GiB read, {} pieces", rs.len(), bytes as f64 / (1u64 << 30) as f64, p.len());
        }
        return Ok(());
    }
    for a in adds {
        let (key, file) = a.split_once('=').context("--add wants r/n=<file>")?;
        let (r, n) = key.split_once('/').context("--add wants r/n=<file>")?;
        let (r, n): (u64, u64) = (r.parse()?, n.parse()?);
        if n == 0 || r >= n {
            bail!("--add {key}: rank out of range");
        }
        let mut rs: Vec<[u64; 2]> = m.shards.remove(key).unwrap_or_default();
        for (i, line) in std::fs::read_to_string(file).with_context(|| format!("read {file}"))?.lines().enumerate() {
            let mut it = line.split_whitespace();
            let (Some(off), Some(len), None) = (it.next(), it.next(), it.next()) else {
                if line.trim().is_empty() {
                    continue;
                }
                bail!("{file}:{}: expected \"offset length\", got {line:?}", i + 1);
            };
            let (off, len): (u64, u64) = (off.parse()?, len.parse()?);
            if off + len > m.file_size {
                bail!("{file}:{}: range [{off}, +{len}) past the end of {} ({} bytes)", i + 1, m.file_name, m.file_size);
            }
            rs.push([off, len]);
        }
        let rs = manifest::coalesce(rs);
        let bytes: u64 = rs.iter().map(|x| x[1]).sum();
        eprintln!("{key}: {} ranges, {:.2} GiB", rs.len(), bytes as f64 / (1u64 << 30) as f64);
        m.shards.insert(key.to_string(), rs);
    }
    let tmp = mpath.with_extension("json.tmp");
    serde_json::to_writer(std::io::BufWriter::new(File::create(&tmp)?), &m)?;
    std::fs::rename(&tmp, mpath)?;
    Ok(())
}

fn cmd_chunk(args: &[String]) -> Result<()> {
    let o = parse(args, &["manifest", "store"])?;
    let [file] = o.positional.as_slice() else { bail!("usage: chunk <file> --manifest m.json --store <dir>") };
    let m = Manifest::load(Path::new(o.flag("manifest").context("--manifest required")?))?;
    if !m.layout.is_empty() {
        bail!("chunk: virtual (layout) manifests are pulled with seed / pull, not chunked");
    }
    let store = Path::new(o.flag("store").context("--store required")?).join(&m.root);
    std::fs::create_dir_all(&store)?;

    let src = File::open(file)?;
    if src.metadata()?.len() != m.file_size {
        bail!("size mismatch vs manifest");
    }
    let t = Instant::now();
    let mut buf = vec![0u8; m.piece_size as usize];
    for idx in 0..m.n_pieces() {
        let off = idx * m.piece_size;
        let len = (m.file_size - off).min(m.piece_size) as usize;
        src.read_exact_at(&mut buf[..len], off)?;
        if hex::encode(blake3::hash(&buf[..len]).as_bytes()) != m.pieces[idx as usize] {
            bail!("piece {idx} does not match manifest — refusing to chunk a corrupt source");
        }
        std::fs::write(store.join(format!("{idx}.piece")), &buf[..len])?;
    }
    let secs = t.elapsed().as_secs_f64();
    eprintln!(
        "chunked {} pieces into {} in {:.2}s = {:.2} GiB/s",
        m.n_pieces(),
        store.display(),
        secs,
        gib_per_s(m.file_size, secs)
    );
    Ok(())
}

fn cmd_rehydrate(args: &[String]) -> Result<()> {
    let o = parse(args, &["manifest", "store"])?;
    let m = Manifest::load(Path::new(o.flag("manifest").context("--manifest required")?))?;
    if !m.layout.is_empty() {
        bail!("rehydrate: virtual (layout) manifests are pulled with seed / pull");
    }
    let store = Path::new(o.flag("store").context("--store required")?).join(&m.root);
    let out_path = o.flag("o").context("-o <file> required")?;

    let t = Instant::now();
    let mut out = std::io::BufWriter::new(File::create(out_path)?);
    for idx in 0..m.n_pieces() {
        let piece = store.join(format!("{idx}.piece"));
        let data = std::fs::read(&piece).with_context(|| format!("missing {}", piece.display()))?;
        if hex::encode(blake3::hash(&data).as_bytes()) != m.pieces[idx as usize] {
            bail!("piece {idx} corrupt in store");
        }
        out.write_all(&data)?;
    }
    out.flush()?;
    let secs = t.elapsed().as_secs_f64();
    eprintln!(
        "rehydrated {:.2} GiB in {:.2}s = {:.2} GiB/s -> {}",
        m.file_size as f64 / (1u64 << 30) as f64,
        secs,
        gib_per_s(m.file_size, secs),
        out_path
    );
    Ok(())
}

fn cmd_serve(args: &[String]) -> Result<()> {
    let o = parse(args, &["listen", "manifest", "file"])?;
    let listen = o.flag("listen").unwrap_or("0.0.0.0:7150");
    let manifests = o.flag_all("manifest");
    if manifests.is_empty() {
        bail!("at least one --manifest required");
    }
    let file_override = o.flag("file");
    if file_override.is_some() && manifests.len() > 1 {
        bail!("--file only pairs with a single --manifest");
    }
    let mut artifacts = Vec::new();
    for m in manifests {
        let a = net::resolve_artifact(Path::new(m), file_override)?;
        eprintln!("  {} root {} ({} pieces)", a.path.display(), &a.manifest.root[..16], a.manifest.n_pieces());
        artifacts.push(a);
    }
    let listener = std::net::TcpListener::bind(listen).with_context(|| format!("bind {listen}"))?;
    net::serve(listener, std::sync::Arc::new(artifacts))
}

/// Which pieces to fetch: everything, tensors matching --match, or one
/// tensor-parallel rank's shard (`r/n`, expert tensors split by --shard-match).
fn select_pieces(m: &Manifest, pats: &[&str], shard: Option<&str>, shard_pats: &[&str]) -> Result<Vec<u64>> {
    if let Some(sh) = shard {
        let (r, n) = sh.split_once('/').context("--shard wants r/n, e.g. 0/2")?;
        let pats: Vec<&str> = if shard_pats.is_empty() { vec![manifest::DEFAULT_SHARD_MATCH] } else { shard_pats.to_vec() };
        return manifest::shard_pieces(m, r.parse()?, n.parse()?, &pats);
    }
    if pats.is_empty() {
        return Ok((0..m.n_pieces()).collect());
    }
    if m.tensors.is_empty() {
        bail!("--match needs a tensor sidecar and this manifest has none");
    }
    let mut p: Vec<u64> = m
        .tensors
        .iter()
        .filter(|t| pats.iter().any(|q| t.name.contains(q)))
        .flat_map(|t| t.piece_first..=t.piece_last)
        .collect();
    if p.is_empty() {
        bail!("no tensors matched");
    }
    p.sort_unstable();
    p.dedup();
    Ok(p)
}

fn report(what: &str, n: usize, r: &net::PullReport, out: &str) {
    let per: Vec<String> = r
        .per_source
        .iter()
        .map(|(s, b)| format!("{s} {:.2} GiB", *b as f64 / (1u64 << 30) as f64))
        .collect();
    eprintln!(
        "{what}: {n} pieces wanted, {} already held; fetched {:.2} GiB in {:.2}s = {:.2} GiB/s [{}] -> {out}",
        r.resumed_pieces,
        r.bytes as f64 / (1u64 << 30) as f64,
        r.secs,
        gib_per_s(r.bytes, r.secs.max(1e-9)),
        per.join(", ")
    );
}

fn cmd_pull(args: &[String]) -> Result<()> {
    let o = parse(args, &["manifest", "from", "conns", "match", "sink", "no-verify", "shard", "shard-match"])?;
    let mode = match (o.flag("sink"), o.flag("no-verify")) {
        (Some(_), _) => net::PullMode::Sink,
        (_, Some(_)) => net::PullMode::NoVerify,
        _ => net::PullMode::Full,
    };
    let m = Manifest::load(Path::new(o.flag("manifest").context("--manifest required")?))?;
    let sources: Vec<String> = o.flag_all("from").iter().map(|s| s.to_string()).collect();
    if sources.is_empty() {
        bail!("at least one --from <addr:port> required");
    }
    let conns: usize = o.flag("conns").map(str::parse).transpose()?.unwrap_or(4);
    let out = o.flag("o").context("-o <file> required")?;
    let pieces = select_pieces(&m, &o.flag_all("match"), o.flag("shard"), &o.flag_all("shard-match"))?;
    let n = pieces.len();
    let have = std::sync::Arc::new(net::HaveMap::empty(m.n_pieces()));
    let r = net::pull_swarm(&m, &sources, conns, pieces, Path::new(out), mode, have)?;
    report("pull", n, &r, out);
    Ok(())
}

/// Pull one or more artifacts while serving every piece already held, then
/// keep serving: the node becomes a seeder for whatever it loaded. `--ready`
/// is touched once every artifact is complete, for launchers to wait on.
fn cmd_seed(args: &[String]) -> Result<()> {
    let o = parse(args, &["want", "from", "listen", "conns", "shard-match", "ready"])?;
    let wants = o.flag_all("want");
    if wants.is_empty() {
        bail!("at least one --want m.json=<out>[@r/n] required");
    }
    let sources: Vec<String> = o.flag_all("from").iter().map(|s| s.to_string()).collect();
    if sources.is_empty() {
        bail!("at least one --from <origin addr:port> required");
    }
    let listen = o.flag("listen").context("--listen <addr:port> required")?;
    let conns: usize = o.flag("conns").map(str::parse).transpose()?.unwrap_or(4);
    let shard_pats = o.flag_all("shard-match");

    struct Want {
        out: String,
        pieces: Vec<u64>,
    }
    let mut served = Vec::new();
    let mut plan = Vec::new();
    for w in wants {
        let (mpath, rest) = w.split_once('=').context("--want wants m.json=<out>[@r/n]")?;
        let (out, shard) = match rest.rsplit_once('@') {
            Some((o, s)) if s.contains('/') => (o, Some(s)),
            _ => (rest, None),
        };
        let m = Manifest::load(Path::new(mpath))?;
        let pieces = select_pieces(&m, &[], shard, &shard_pats)?;
        let have = std::sync::Arc::new(net::HaveMap::empty(m.n_pieces()));
        served.push(net::Served { manifest: m, path: PathBuf::from(out), have });
        plan.push(Want { out: out.to_string(), pieces });
    }
    let served = std::sync::Arc::new(served);

    // Serve first, so peers see our have-map grow from the first piece.
    let listener = std::net::TcpListener::bind(listen).with_context(|| format!("bind {listen}"))?;
    {
        let served = served.clone();
        std::thread::spawn(move || {
            if let Err(e) = net::serve(listener, served) {
                eprintln!("serve: {e:#}");
                std::process::exit(2);
            }
        });
    }

    // The output files must exist at full size before the server can open
    // them; pull_swarm creates/resumes them, so pull in sequence.
    let t = Instant::now();
    for (a, w) in served.iter().zip(&plan) {
        let n = w.pieces.len();
        let r = net::pull_swarm(&a.manifest, &sources, conns, w.pieces.clone(), Path::new(&w.out), net::PullMode::Full, a.have.clone())?;
        report(&a.manifest.file_name, n, &r, &w.out);
    }
    eprintln!("all artifacts complete in {:.1}s; seeding on {listen}", t.elapsed().as_secs_f64());
    if let Some(ready) = o.flag("ready") {
        std::fs::write(ready, format!("{}\n", t.elapsed().as_secs_f64()))?;
    }
    loop {
        std::thread::park();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_non_gguf() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("loom-swarm-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let f = dir.join("blob.bin");
        // 2.5 pieces at 1 MiB piece size, deterministic bytes.
        let data: Vec<u8> = (0..(5 << 19)).map(|i| (i * 2654435761u64 >> 16) as u8).collect();
        std::fs::write(&f, &data)?;

        let m = manifest::build(&f, 1 << 20, 4)?;
        assert_eq!(m.n_pieces(), 3);
        assert!(m.gguf.is_none());
        assert_eq!(m.pieces.len(), 3);

        // Corrupt one byte and confirm exactly one piece flags.
        let mut bad = data.clone();
        bad[1 << 20] ^= 1;
        let f2 = dir.join("blob2.bin");
        std::fs::write(&f2, &bad)?;
        let h2 = manifest::hash_pieces(&f2, 1 << 20, 4)?;
        let mismatches: Vec<usize> =
            (0..3).filter(|&i| hex::encode(h2[i]) != m.pieces[i]).collect();
        assert_eq!(mismatches, vec![1]);

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn explicit_shards_non_gguf() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("loom-swarm-shards-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let f = dir.join("w.strixw");
        std::fs::write(&f, vec![7u8; 10 << 20])?; // 10 pieces of 1 MiB
        let mut m = manifest::build(&f, 1 << 20, 2)?;
        let root = m.root.clone();
        // no sidecar and no explicit shard: --shard refuses
        assert!(manifest::shard_pieces(&m, 0, 2, &[]).is_err());
        // rank 1 of 2 reads [0, 100) (header), [3 MiB + 5, +2 MiB) and a touching range that coalesces
        let mib = 1u64 << 20;
        m.shards.insert("1/2".into(), manifest::coalesce(vec![[3 * mib + 5, 2 * mib], [0, 100], [5 * mib + 5, 10]]));
        assert_eq!(m.shards["1/2"], vec![[0, 100], [3 * mib + 5, 2 * mib + 10]]);
        assert_eq!(manifest::shard_pieces(&m, 1, 2, &[])?, vec![0, 3, 4, 5]);
        assert!(manifest::shard_pieces(&m, 0, 2, &[]).is_err()); // rank 0 has none recorded
        // round trip through JSON keeps shards and the root
        let j = serde_json::to_string(&m)?;
        let m2: Manifest = serde_json::from_str(&j)?;
        assert_eq!(m2.root, root);
        assert_eq!(manifest::shard_pieces(&m2, 1, 2, &[])?, vec![0, 3, 4, 5]);
        // a range past the end is refused
        m.shards.insert("0/2".into(), vec![[9 * mib, 2 * mib]]);
        assert!(manifest::shard_pieces(&m, 0, 2, &[]).is_err());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn ranges_compact() {
        assert_eq!(manifest::ranges(vec![0, 1, 2, 5, 7, 8]), "0-2,5,7-8");
    }
}
