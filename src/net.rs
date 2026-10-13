//! Rung-2 transport: piece ranges over plain TCP, with swarm semantics.
//!
//! Rung 1 (libtorrent) measured 0.38 GiB/s loopback tuned — 18x short of the
//! ~6.9 GB/s fabric NIC — while raw TCP sendfile did 10.65 GiB/s on the same
//! box (card afe883b66d, 2026-08-15). So the transport is the simplest thing
//! that rides that path: a request for piece ranges, streamed back
//! back-to-back, verified against the manifest on the pulling side. The
//! server never hashes; the puller always does.
//!
//! Swarm (v2, 2026-10-03): every artifact carries a have-map. A seeder serves
//! the pieces it already holds while it is still pulling the rest, so N nodes
//! loading the same model stop all queueing on one origin NIC. Pullers ask each
//! source what it has (HAVE) and steal work piece-batch by piece-batch, so a
//! fast source (a Thunderbolt peer) naturally takes more than a slow one.
//!
//! Wire format (all integers LE):
//!   v1 request: b"LSWM" u8 1  [u8;32] root  u32 n  n x (u64 start, u64 count)
//!               one GET per connection (kept for old pullers).
//!   v2 request: b"LSWM" u8 2  u8 op  [u8;32] root  <op body>; repeat until EOF.
//!     op 0 GET  body: u32 n  n x (u64 start, u64 count)
//!     op 1 HAVE body: none
//!   response: u8 status (0 ok, 1 unknown root, 2 bad range, 3 piece not held)
//!     GET  ok: raw piece bytes in request order (lengths implied by geometry)
//!     HAVE ok: u64 n_pieces  then ceil(n/64) LE u64 words, bit i = piece i held

use crate::manifest::Manifest;
use anyhow::{bail, Context, Result};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAGIC: &[u8; 4] = b"LSWM";
const V1: u8 = 1;
const V2: u8 = 2;
const OP_GET: u8 = 0;
const OP_HAVE: u8 = 1;
const ST_OK: u8 = 0;
const ST_UNKNOWN_ROOT: u8 = 1;
const ST_BAD_RANGE: u8 = 2;
const ST_NOT_HELD: u8 = 3;

/// Pieces per GET. 4 x 16 MiB keeps the tail short and the stealing fair
/// while each request still streams 64 MiB back-to-back.
const BATCH: usize = 4;
/// How stale a source's have-map may get before it is re-asked.
const HAVE_TTL: Duration = Duration::from_millis(500);
/// A source that refuses connections this many times in a row is dropped.
const MAX_CONNECT_FAILS: u32 = 40;
const IO_TIMEOUT: Duration = Duration::from_secs(60);
/// How long the origin waits at start for every live peer's first have-map.
const PEER_GRACE: Duration = Duration::from_secs(2);

/// Lock-free bitset of held pieces, shared between a seeder's server and puller.
pub struct HaveMap {
    words: Vec<AtomicU64>,
    n: u64,
}

impl HaveMap {
    pub fn empty(n: u64) -> HaveMap {
        HaveMap { words: (0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(), n }
    }
    pub fn full(n: u64) -> HaveMap {
        let h = HaveMap::empty(n);
        for i in 0..n {
            h.set(i);
        }
        h
    }
    pub fn set(&self, i: u64) {
        self.words[(i / 64) as usize].fetch_or(1 << (i % 64), Ordering::Release);
    }
    pub fn get(&self, i: u64) -> bool {
        i < self.n && self.words[(i / 64) as usize].load(Ordering::Acquire) & (1 << (i % 64)) != 0
    }
    #[allow(dead_code)]
    pub fn count(&self) -> u64 {
        self.words.iter().map(|w| w.load(Ordering::Relaxed).count_ones() as u64).sum()
    }
    fn to_bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.load(Ordering::Acquire).to_le_bytes()).collect()
    }
}

/// A peer's have-map as last reported.
struct Snapshot {
    words: Vec<u64>,
}

impl Snapshot {
    fn get(&self, i: u64) -> bool {
        self.words.get((i / 64) as usize).is_some_and(|w| w & (1 << (i % 64)) != 0)
    }
}

pub struct Served {
    pub manifest: Manifest,
    pub path: PathBuf,
    pub have: Arc<HaveMap>,
}

/// One artifact per manifest; the file defaults to `file_name` next to the
/// manifest, matching how manifests are laid down beside what they describe.
/// A served-from-origin artifact is trusted complete (have = all pieces).
pub fn resolve_artifact(manifest_path: &Path, file_override: Option<&str>) -> Result<Served> {
    let manifest = Manifest::load(manifest_path)?;
    let path = match file_override {
        Some(p) => PathBuf::from(p),
        None => manifest_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(&manifest.file_name),
    };
    let size = std::fs::metadata(&path)
        .with_context(|| format!("artifact {}", path.display()))?
        .len();
    if size != manifest.on_disk_size() {
        bail!(
            "{}: size {} does not match manifest {}",
            path.display(),
            size,
            manifest.on_disk_size()
        );
    }
    let have = Arc::new(HaveMap::full(manifest.n_pieces()));
    Ok(Served { manifest, path, have })
}

pub fn serve(listener: TcpListener, artifacts: Arc<Vec<Served>>) -> Result<()> {
    eprintln!(
        "serving {} artifact(s) on {}",
        artifacts.len(),
        listener.local_addr()?
    );
    for conn in listener.incoming() {
        let conn = match conn {
            Ok(c) => c,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        let artifacts = artifacts.clone();
        std::thread::spawn(move || {
            let peer = conn.peer_addr().map(|a| a.to_string()).unwrap_or_default();
            if let Err(e) = handle(conn, &artifacts) {
                eprintln!("[{peer}] {e:#}");
            }
        });
    }
    Ok(())
}

fn handle(mut conn: TcpStream, artifacts: &[Served]) -> Result<()> {
    conn.set_nodelay(true)?;
    loop {
        let mut head = [0u8; 5];
        match conn.read_exact(&mut head) {
            Ok(()) => {}
            // A clean close between requests is how v2 clients say goodbye.
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }
        if &head[..4] != MAGIC || !(head[4] == V1 || head[4] == V2) {
            bail!("bad handshake");
        }
        let version = head[4];
        let op = if version == V1 { OP_GET } else { read_u8(&mut conn)? };
        let mut root = [0u8; 32];
        conn.read_exact(&mut root)?;
        let root_hex = hex::encode(root);
        let art = artifacts.iter().find(|a| a.manifest.root == root_hex);

        match op {
            OP_HAVE => {
                let Some(a) = art else {
                    conn.write_all(&[ST_UNKNOWN_ROOT])?;
                    continue;
                };
                let mut out = vec![ST_OK];
                out.extend_from_slice(&a.manifest.n_pieces().to_le_bytes());
                out.extend_from_slice(&a.have.to_bytes());
                conn.write_all(&out)?;
            }
            OP_GET => {
                let n_ranges = read_u32(&mut conn)?;
                if n_ranges > 1 << 20 {
                    bail!("implausible range count {n_ranges}");
                }
                let mut ranges = Vec::with_capacity(n_ranges as usize);
                for _ in 0..n_ranges {
                    ranges.push((read_u64(&mut conn)?, read_u64(&mut conn)?));
                }
                let Some(a) = art else {
                    conn.write_all(&[ST_UNKNOWN_ROOT])?;
                    if version == V1 {
                        bail!("unknown root {root_hex}");
                    }
                    continue;
                };
                let m = &a.manifest;
                let n_pieces = m.n_pieces();
                if ranges.iter().any(|&(s, c)| c == 0 || s.checked_add(c).is_none_or(|e| e > n_pieces)) {
                    conn.write_all(&[ST_BAD_RANGE])?;
                    bail!("bad range in request for {n_pieces} pieces");
                }
                if !ranges.iter().all(|&(s, c)| (s..s + c).all(|i| a.have.get(i))) {
                    conn.write_all(&[ST_NOT_HELD])?;
                    if version == V1 {
                        return Ok(());
                    }
                    continue;
                }
                conn.write_all(&[ST_OK])?;
                // io::copy(File::take -> socket) rides the kernel sendfile path
                // on Linux; the server never buffers or hashes piece data.
                let mut file = File::open(&a.path)?;
                for (start, count) in ranges {
                    let voff = start * m.piece_size;
                    let vlen = ((start + count) * m.piece_size).min(m.file_size) - voff;
                    // a virtual (layout) artifact sends its pieces' segments in order: same bytes on the wire
                    for (off, len) in m.segments(voff, vlen) {
                        file.seek(SeekFrom::Start(off))?;
                        let sent = std::io::copy(&mut Read::by_ref(&mut file).take(len), &mut conn)?;
                        if sent != len {
                            bail!("short send: {sent} of {len}");
                        }
                    }
                }
            }
            other => bail!("unknown op {other}"),
        }
        if version == V1 {
            return Ok(());
        }
    }
}

/// Stage ablation for benchmarking: skip the write, the hash, or neither.
/// `NoVerify` is for measuring only — a real pull always verifies.
#[derive(Clone, Copy, PartialEq)]
pub enum PullMode {
    Full,
    Sink,
    NoVerify,
}

const NEEDED: u8 = 1;
const CLAIMED: u8 = 2;
const DONE: u8 = 3;

pub struct PullReport {
    pub bytes: u64,
    pub secs: f64,
    pub resumed_pieces: u64,
    /// Bytes fetched per source, in `sources` order.
    pub per_source: Vec<(String, u64)>,
}

/// Open (or create) `out` at full size and mark every wanted piece that is
/// already present and verifies as held — a restarted seeder resumes instead
/// of refetching, and a hole reads as zeros so it never falsely verifies.
fn open_resume(m: &Manifest, want: &[[u8; 32]], out: &Path, pieces: &[u64], have: &HaveMap) -> Result<(File, u64)> {
    let existed = out.exists();
    let f = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(out)
        .with_context(|| format!("open {}", out.display()))?;
    if f.metadata()?.len() != m.on_disk_size() {
        f.set_len(0)?;
        f.set_len(m.on_disk_size())?;
        return Ok((f, 0));
    }
    if !existed {
        return Ok((f, 0));
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(16);
    let per = pieces.len().div_ceil(threads).max(1);
    let found = AtomicU64::new(0);
    std::thread::scope(|s| {
        for chunk in pieces.chunks(per) {
            let (f, found) = (&f, &found);
            s.spawn(move || {
                let mut buf = vec![0u8; m.piece_size as usize];
                for &i in chunk {
                    let len = m.piece_len(i) as usize;
                    if m.read_piece(f, i, &mut buf).is_ok()
                        && blake3::hash(&buf[..len]).as_bytes() == &want[i as usize]
                    {
                        have.set(i);
                        found.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        }
    });
    Ok((f, found.into_inner()))
}

fn decode_hashes(m: &Manifest) -> Result<(Vec<[u8; 32]>, [u8; 32])> {
    let want = m
        .pieces
        .iter()
        .map(|h| {
            let v = hex::decode(h)?;
            <[u8; 32]>::try_from(v.as_slice()).map_err(|_| anyhow::anyhow!("bad hash length"))
        })
        .collect::<Result<_>>()?;
    let root = <[u8; 32]>::try_from(hex::decode(&m.root)?.as_slice())
        .map_err(|_| anyhow::anyhow!("bad root length"))?;
    Ok((want, root))
}

/// Shared state of one pull.
struct Job<'a> {
    m: &'a Manifest,
    want: Vec<[u8; 32]>,
    root: [u8; 32],
    out: Option<File>,
    have: Arc<HaveMap>,
    mode: PullMode,
    /// Pull order: the wanted pieces, shuffled in runs so concurrent pullers
    /// of the same shard hit the origin for different pieces first.
    order: Vec<u64>,
    state: Vec<AtomicU8>,
    remaining: AtomicUsize,
    /// Latest have-map per source (index 0 = origin).
    snaps: Vec<Mutex<Option<Arc<Snapshot>>>>,
    alive: Vec<AtomicUsize>,
    fetched: Vec<AtomicU64>,
    abort: AtomicBool,
    t0: Instant,
}

impl Job<'_> {
    /// A peer (any source but the origin) currently holds piece i.
    /// Live peers' latest have-maps; None while a live peer has not reported
    /// yet (the origin holds off briefly so it doesn't race the peers' first answer).
    fn peer_snaps(&self) -> Option<Vec<Arc<Snapshot>>> {
        let mut v = Vec::new();
        for s in 1..self.snaps.len() {
            if self.alive[s].load(Ordering::Relaxed) == 0 {
                continue;
            }
            match self.snaps[s].lock().unwrap().as_ref() {
                Some(sn) => v.push(sn.clone()),
                None if self.t0.elapsed() < PEER_GRACE => return None,
                None => {}
            }
        }
        Some(v)
    }

    /// Claim up to BATCH pieces, contiguous where possible, that `src` holds.
    /// The origin leaves pieces a live peer already has to that peer, so its
    /// one NIC is spent only on bytes nobody else can supply yet.
    fn claim(&self, src: usize, snap: &Snapshot, cursor: &mut usize) -> Vec<u64> {
        let n = self.order.len();
        let mut got = Vec::new();
        let peers = if src == 0 {
            match self.peer_snaps() {
                Some(p) => p,
                None => return got,
            }
        } else {
            Vec::new()
        };
        for k in 0..n {
            let pos = (*cursor + k) % n;
            let i = self.order[pos];
            if self.state[i as usize].load(Ordering::Relaxed) != NEEDED || !snap.get(i) {
                continue;
            }
            if peers.iter().any(|p| p.get(i)) {
                continue;
            }
            if self.state[i as usize]
                .compare_exchange(NEEDED, CLAIMED, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            got.push(i);
            // Extend with the following pieces in file order while they qualify.
            let mut j = i + 1;
            while got.len() < BATCH
                && (j as usize) < self.state.len()
                && snap.get(j)
                && self.state[j as usize]
                    .compare_exchange(NEEDED, CLAIMED, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
            {
                got.push(j);
                j += 1;
            }
            *cursor = (pos + 1) % n;
            break;
        }
        got
    }

    fn release(&self, pieces: &[u64]) {
        for &i in pieces {
            let _ = self.state[i as usize].compare_exchange(CLAIMED, NEEDED, Ordering::AcqRel, Ordering::Relaxed);
        }
    }
}

/// Pull `pieces` from `sources` (the first is the origin, the rest are peers
/// that may hold only part of the artifact), `conns` connections per source,
/// verifying every piece against the manifest and writing it at its offset in
/// `out`. Pieces already present and valid in `out` are kept (resume). Subset
/// pulls leave holes — the file is full-size, sparse elsewhere. `have` is
/// updated as pieces land, so a server sharing it seeds them immediately.
pub fn pull_swarm(
    m: &Manifest,
    sources: &[String],
    conns: usize,
    pieces: Vec<u64>,
    out: &Path,
    mode: PullMode,
    have: Arc<HaveMap>,
) -> Result<PullReport> {
    if pieces.is_empty() {
        bail!("nothing to pull");
    }
    if sources.is_empty() {
        bail!("no sources");
    }
    let (want, root) = decode_hashes(m)?;
    let (out_file, resumed) = if mode == PullMode::Sink {
        (None, 0)
    } else {
        let (f, r) = open_resume(m, &want, out, &pieces, &have)?;
        (Some(f), r)
    };

    let state: Vec<AtomicU8> = (0..m.n_pieces()).map(|_| AtomicU8::new(0)).collect();
    let mut remaining = 0usize;
    for &i in &pieces {
        if !have.get(i) {
            state[i as usize].store(NEEDED, Ordering::Relaxed);
            remaining += 1;
        } else {
            state[i as usize].store(DONE, Ordering::Relaxed);
        }
    }
    let order = shuffled_runs(&pieces, 8);
    let job = Job {
        m,
        want,
        root,
        out: out_file,
        have,
        mode,
        order,
        state,
        remaining: AtomicUsize::new(remaining),
        snaps: sources.iter().map(|_| Mutex::new(None)).collect(),
        alive: sources.iter().map(|_| AtomicUsize::new(conns.max(1))).collect(),
        fetched: sources.iter().map(|_| AtomicU64::new(0)).collect(),
        abort: AtomicBool::new(false),
        t0: Instant::now(),
    };

    let t0 = Instant::now();
    let conns = conns.max(1);
    let nthreads = sources.len() * conns;
    std::thread::scope(|s| {
        for (si, src) in sources.iter().enumerate() {
            for c in 0..conns {
                let job = &job;
                // Spread each thread's starting point over the order.
                let start = (si * conns + c) * job.order.len() / nthreads;
                s.spawn(move || source_worker(job, si, src, start));
            }
        }
    });
    let left = job.remaining.load(Ordering::Acquire);
    if left > 0 {
        bail!("{left} pieces could not be fetched from any source");
    }
    let bytes: u64 = job.fetched.iter().map(|b| b.load(Ordering::Relaxed)).sum();
    Ok(PullReport {
        bytes,
        secs: t0.elapsed().as_secs_f64(),
        resumed_pieces: resumed,
        per_source: sources
            .iter()
            .zip(&job.fetched)
            .map(|(s, b)| (s.clone(), b.load(Ordering::Relaxed)))
            .collect(),
    })
}

/// Plain single-shot pull (tests and library callers); no shared have-map.
#[allow(dead_code)]
pub fn pull(
    m: &Manifest,
    sources: &[String],
    conns: usize,
    pieces: Vec<u64>,
    out: &Path,
    mode: PullMode,
) -> Result<(u64, f64)> {
    let have = Arc::new(HaveMap::empty(m.n_pieces()));
    let r = pull_swarm(m, sources, conns, pieces, out, mode, have)?;
    Ok((r.bytes, r.secs))
}

fn source_worker(job: &Job, si: usize, src: &str, start: usize) {
    let mut conn: Option<TcpStream> = None;
    let mut cursor = start;
    let mut fails = 0u32;
    let mut last_have = Instant::now() - HAVE_TTL * 2;
    let mut snap: Option<Arc<Snapshot>> = None;
    let mut buf = vec![0u8; job.m.piece_size as usize * BATCH];

    let quit = |job: &Job| {
        job.alive[si].fetch_sub(1, Ordering::Relaxed);
    };
    loop {
        if job.remaining.load(Ordering::Acquire) == 0 || job.abort.load(Ordering::Relaxed) {
            return quit(job);
        }
        if conn.is_none() {
            match connect(src) {
                Ok(c) => {
                    conn = Some(c);
                    fails = 0;
                }
                Err(_) => {
                    fails += 1;
                    if fails >= MAX_CONNECT_FAILS {
                        eprintln!("source {src}: unreachable, giving up on it");
                        return quit(job);
                    }
                    std::thread::sleep(Duration::from_millis(500));
                    continue;
                }
            }
        }
        let c = conn.as_mut().unwrap();

        if snap.is_none() || last_have.elapsed() > HAVE_TTL {
            match request_have(c, &job.root, job.m.n_pieces()) {
                Ok(Some(s)) => {
                    let s = Arc::new(s);
                    *job.snaps[si].lock().unwrap() = Some(s.clone());
                    snap = Some(s);
                    last_have = Instant::now();
                }
                Ok(None) => {
                    // Source does not carry this artifact (yet): poll slowly.
                    snap = Some(Arc::new(Snapshot { words: Vec::new() }));
                    last_have = Instant::now();
                }
                Err(_) => {
                    conn = None;
                    fails += 1;
                    if fails >= MAX_CONNECT_FAILS {
                        return quit(job);
                    }
                    std::thread::sleep(Duration::from_millis(500));
                    continue;
                }
            }
        }

        let batch = job.claim(si, snap.as_ref().unwrap(), &mut cursor);
        if batch.is_empty() {
            // Nothing this source can supply right now; it may later (a peer
            // still pulling), or another thread's claim may be released.
            std::thread::sleep(Duration::from_millis(100));
            last_have = Instant::now() - HAVE_TTL * 2;
            continue;
        }
        match fetch_batch(job, si, c, &batch, &mut buf) {
            Ok(true) => {}
            Ok(false) => {
                // Source no longer claims these (or never did): refresh view.
                job.release(&batch);
                last_have = Instant::now() - HAVE_TTL * 2;
            }
            Err(e) => {
                job.release(&batch);
                eprintln!("source {src}: {e:#}");
                conn = None;
                fails += 1;
                if fails >= MAX_CONNECT_FAILS {
                    return quit(job);
                }
            }
        }
    }
}

fn connect(src: &str) -> Result<TcpStream> {
    let addr = src
        .to_socket_addrs()?
        .next()
        .with_context(|| format!("resolve {src}"))?;
    let c = TcpStream::connect_timeout(&addr, Duration::from_secs(3))?;
    c.set_nodelay(true)?;
    c.set_read_timeout(Some(IO_TIMEOUT))?;
    c.set_write_timeout(Some(IO_TIMEOUT))?;
    Ok(c)
}

fn request_have(c: &mut TcpStream, root: &[u8; 32], n: u64) -> Result<Option<Snapshot>> {
    let mut req = Vec::with_capacity(38);
    req.extend_from_slice(MAGIC);
    req.push(V2);
    req.push(OP_HAVE);
    req.extend_from_slice(root);
    c.write_all(&req)?;
    match read_u8(c)? {
        ST_OK => {}
        ST_UNKNOWN_ROOT => return Ok(None),
        s => bail!("HAVE refused: status {s}"),
    }
    let got = read_u64(c)?;
    if got != n {
        bail!("source reports {got} pieces, manifest has {n}");
    }
    let mut words = Vec::with_capacity(n.div_ceil(64) as usize);
    for _ in 0..n.div_ceil(64) {
        words.push(read_u64(c)?);
    }
    Ok(Some(Snapshot { words }))
}

/// GET one batch (contiguous runs). Ok(false) = source says it doesn't hold them.
fn fetch_batch(job: &Job, si: usize, c: &mut TcpStream, batch: &[u64], buf: &mut [u8]) -> Result<bool> {
    let m = job.m;
    let runs = to_runs(batch);
    let mut req = Vec::with_capacity(42 + runs.len() * 16);
    req.extend_from_slice(MAGIC);
    req.push(V2);
    req.push(OP_GET);
    req.extend_from_slice(&job.root);
    req.extend_from_slice(&(runs.len() as u32).to_le_bytes());
    for &(s, n) in &runs {
        req.extend_from_slice(&s.to_le_bytes());
        req.extend_from_slice(&n.to_le_bytes());
    }
    c.write_all(&req)?;
    match read_u8(c)? {
        ST_OK => {}
        ST_NOT_HELD | ST_UNKNOWN_ROOT => return Ok(false),
        s => bail!("GET refused: status {s}"),
    }
    // Read the whole batch before verifying any of it: the stream must be
    // drained in full to stay framed even if one piece turns out bad.
    let mut lens = Vec::with_capacity(batch.len());
    let mut at = 0usize;
    for &i in batch {
        let off = i * m.piece_size;
        let len = (m.file_size - off).min(m.piece_size) as usize;
        c.read_exact(&mut buf[at..at + len]).with_context(|| format!("piece {i}"))?;
        lens.push((i, off, at, len));
        at += len;
    }
    let mut bad = Vec::new();
    for &(i, off, at, len) in &lens {
        let data = &buf[at..at + len];
        if job.mode != PullMode::NoVerify && blake3::hash(data).as_bytes() != &job.want[i as usize] {
            bad.push(i);
            continue;
        }
        if let Some(f) = &job.out {
            if m.layout.is_empty() {
                f.write_all_at(data, off)?;
            } else {
                m.write_piece(f, i, data)?;
            }
        }
        job.have.set(i);
        job.state[i as usize].store(DONE, Ordering::Release);
        job.remaining.fetch_sub(1, Ordering::AcqRel);
        job.fetched[si].fetch_add(len as u64, Ordering::Relaxed);
    }
    if !bad.is_empty() {
        job.release(&bad);
        bail!("pieces {bad:?} failed verification — corrupt or wrong source");
    }
    Ok(true)
}

/// Pieces in file order, grouped into runs of `run` and the runs shuffled with
/// a per-process seed (pid ^ time): distinct pullers start in distinct places.
fn shuffled_runs(pieces: &[u64], run: usize) -> Vec<u64> {
    let mut chunks: Vec<&[u64]> = pieces.chunks(run).collect();
    let mut x = (std::process::id() as u64) << 32
        ^ std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1)
        | 1;
    for i in (1..chunks.len()).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        chunks.swap(i, (x % (i as u64 + 1)) as usize);
    }
    chunks.concat()
}

fn to_runs(pieces: &[u64]) -> Vec<(u64, u64)> {
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for &p in pieces {
        match runs.last_mut() {
            Some((start, count)) if *start + *count == p => *count += 1,
            _ => runs.push((p, 1)),
        }
    }
    runs
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
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

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(dir: &Path, name: &str, pieces_x2: u64) -> (PathBuf, Vec<u8>) {
        let src = dir.join(name);
        let data: Vec<u8> = (0..(pieces_x2 << 19)).map(|i| (i * 40503u64 >> 8) as u8).collect();
        std::fs::write(&src, &data).unwrap();
        (src, data)
    }

    fn spawn_server(arts: Vec<Served>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let arts = Arc::new(arts);
        std::thread::spawn(move || {
            let _ = serve(listener, arts);
        });
        addr
    }

    #[test]
    fn runs_from_pieces() {
        assert_eq!(to_runs(&[0, 1, 2, 5, 7, 8]), vec![(0, 3), (5, 1), (7, 2)]);
    }

    #[test]
    fn serve_pull_roundtrip() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("loom-swarm-net-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let (src, data) = blob(&dir, "blob.bin", 5);

        let m = crate::manifest::build(&src, 1 << 20, 2)?;
        let n = m.n_pieces();
        let addr = spawn_server(vec![Served {
            manifest: crate::manifest::build(&src, 1 << 20, 2)?,
            path: src.clone(),
            have: Arc::new(HaveMap::full(n)),
        }]);

        let out = dir.join("out.bin");
        let (bytes, _) = pull(&m, &[addr.clone()], 2, (0..n).collect(), &out, PullMode::Full)?;
        assert_eq!(bytes, data.len() as u64);
        assert_eq!(std::fs::read(&out)?, data);

        // Resume: re-pulling into a complete file fetches nothing.
        let have = Arc::new(HaveMap::empty(n));
        let r = pull_swarm(&m, &[addr.clone()], 2, (0..n).collect(), &out, PullMode::Full, have)?;
        assert_eq!((r.bytes, r.resumed_pieces), (0, n));

        // Subset pull: piece 1 only — full-size sparse file, that range correct.
        let out2 = dir.join("out2.bin");
        pull(&m, &[addr], 1, vec![1], &out2, PullMode::Full)?;
        let got = std::fs::read(&out2)?;
        assert_eq!(got.len(), data.len());
        assert_eq!(&got[1 << 20..2 << 20], &data[1 << 20..2 << 20]);

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// A partial peer serves only what it holds; a puller with origin + peer
    /// takes the peer's pieces from the peer and the rest from the origin.
    #[test]
    fn layout_artifact_roundtrip() -> Result<()> {
        // a "rank" reading interleaved 6,400-byte runs of a 5 MiB file: its virtual artifact is just those bytes
        let dir = std::env::temp_dir().join(format!("loom-swarm-layout-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let (src, data) = blob(&dir, "w.bin", 10);
        let mut ranges: Vec<[u64; 2]> = (0..data.len() as u64 / 25_600).map(|k| [k * 25_600 + 6_400, 6_400]).collect();
        ranges.push([0, 100]); // header
        let m = crate::manifest::build_layout(&src, ranges.clone(), 1 << 20, 2)?;
        assert_eq!(m.disk_size, data.len() as u64);
        assert_eq!(m.file_size, 100 + (data.len() as u64 / 25_600) * 6_400);
        let n = m.n_pieces();
        let addr = spawn_server(vec![Served {
            manifest: crate::manifest::build_layout(&src, ranges.clone(), 1 << 20, 2)?,
            path: src.clone(),
            have: Arc::new(HaveMap::full(n)),
        }]);
        let out = dir.join("out.bin");
        let (bytes, _) = pull(&m, &[addr.clone()], 2, (0..n).collect(), &out, PullMode::Full)?;
        assert_eq!(bytes, m.file_size); // only the rank's bytes crossed the wire
        let got = std::fs::read(&out)?;
        assert_eq!(got.len(), data.len()); // full-size (sparse) file
        for [o, l] in crate::manifest::coalesce(ranges) {
            assert_eq!(&got[o as usize..(o + l) as usize], &data[o as usize..(o + l) as usize]);
        }
        assert!(got[100..6_400].iter().all(|&b| b == 0)); // not this rank's: never written
        // resume over the written file: everything verifies through the layout, nothing refetched
        let r = pull_swarm(&m, &[addr], 2, (0..n).collect(), &out, PullMode::Full, Arc::new(HaveMap::empty(n)))?;
        assert_eq!((r.bytes, r.resumed_pieces), (0, n));
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn swarm_prefers_peer_for_held_pieces() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("loom-swarm-swarm-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let (src, data) = blob(&dir, "blob.bin", 16); // 8 pieces of 1 MiB
        let m = crate::manifest::build(&src, 1 << 20, 2)?;
        let n = m.n_pieces();
        let origin = spawn_server(vec![Served {
            manifest: crate::manifest::build(&src, 1 << 20, 2)?,
            path: src.clone(),
            have: Arc::new(HaveMap::full(n)),
        }]);

        // Peer: pull pieces 0..4 from origin, then serve with its partial map.
        let peer_file = dir.join("peer.bin");
        let peer_have = Arc::new(HaveMap::empty(n));
        pull_swarm(&m, &[origin.clone()], 1, (0..4).collect(), &peer_file, PullMode::Full, peer_have.clone())?;
        assert_eq!(peer_have.count(), 4);
        let peer = spawn_server(vec![Served {
            manifest: crate::manifest::build(&src, 1 << 20, 2)?,
            path: peer_file,
            have: peer_have,
        }]);

        let out = dir.join("out.bin");
        let r = pull_swarm(&m, &[origin, peer], 1, (0..n).collect(), &out, PullMode::Full, Arc::new(HaveMap::empty(n)))?;
        assert_eq!(std::fs::read(&out)?, data);
        assert_eq!(r.per_source[1].1, 4 << 20, "peer should supply exactly what it holds");
        assert_eq!(r.per_source[0].1, 4 << 20);

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
