# Using loom-swarm from an inference engine

loom-swarm moves model weights to the nodes that need them and verifies every byte. It does not care what the engine
is. What the engine provides decides how much of the crate it can use. Four tiers, each a superset of the one before.

## Tier 0: a file at a path

The engine changes nothing. An operator runs `loom-swarm manifest <weights>` once on the origin, `loom-swarm serve`
there, and `loom-swarm seed --want <manifest>=<path> ...` on each node. When the ready file appears the engine opens
`<path>` as it always did.

What you get: peer-to-peer loading (N nodes stop queueing on one origin link), BLAKE3 verification of every piece,
resume after interruption, and a warm restart in seconds when the seeded copy is kept.

Engine obligation: none. Any file format.

## Tier 1: say which bytes a rank reads

For tensor-parallel or pipeline-parallel engines, a rank needs only part of the file. Two ways to tell loom-swarm:

- **Static shard ranges.** `loom-swarm shards --manifest m.json --add 0/2=<ranges> --add 1/2=<ranges>` records
  explicit `[offset, length]` ranges per rank in the manifest. The engine (or its converter) emits those ranges.
- **A recorded read log.** Run one rank once with reads logged (`offset length` per line), then
  `loom-swarm manifest <weights> --layout <log> -o <weights>.r0of2.swarm.json`. The result is a *virtual artifact*:
  exactly that rank's bytes, pulled and written at their real offsets into a sparse full-size file, so the engine's
  paths and offsets do not change.

What you get: a rank pulls only what it reads. Measured: a 4-way tensor-parallel rank fetched 36.7 GiB instead of 67.6.

Engine obligation: deterministic reads for a given rank (so a log is reusable), or a converter that can state the
per-rank byte ranges. Reads must be at stable offsets; mmap is fine (the log can come from `strace`, an LD_PRELOAD
read logger, or the engine's own loader instrumentation).

## Tier 2: expose tensor structure

If the file format can be read into a list of `(tensor name, offset, length)`, loom-swarm maps tensors onto pieces and
pulls can be expressed by name: `--match _exps --shard 1/4` pulls the expert tensors that belong to rank 1 of 4.

GGUF has this built in (`src/gguf.rs`, 175 lines: a header reader whose extents are offset deltas, so alignment padding
is included and the padded range is what you must read). A second format needs the same shape of reader:

```rust
pub struct TensorExtent { pub name: String, pub offset: u64, pub size: u64 }
pub fn parse(path: &Path) -> anyhow::Result<Option<Vec<TensorExtent>>>   // None when the file is not this format
```

Formats are feature-gated; `gguf` is on by default. safetensors is the obvious next one (its header is JSON with
`data_offsets` per tensor).

What you get: pulls by tensor pattern, the expert-shard rule (`_exps` tensors split by rank), and the basis for paging
experts in and out by name.

Engine obligation: a readable header with tensor offsets, or a sidecar that provides one.

## Tier 3: link the library

`loom-swarm` is also a library: `manifest`, `gguf` and `net` are public modules. An engine that wants to start
before every byte has landed, or to order pieces by its own layer schedule, uses:

- `manifest::Manifest::load`, `segments`, `read_piece`, `write_piece`: the geometry of pieces, including virtual layouts.
- `manifest::shard_pieces`: the piece list for a rank under the expert rule or explicit shards.
- `net::HaveMap`: what is held locally, shareable with the seeder.
- `net` pullers and the seeder: hand them a piece priority list instead of "everything".

What you get: lazy loading on first touch, layer-ordered arrival, serving pieces to peers from inside the engine's
own process.

Engine obligation: Rust, or a C ABI shim (not provided yet; the surface is small enough that one is a short job).

## Performance options and where they live

| Option | Tier | Where |
|---|---|---|
| Multiple sources, work stealing in 4-piece batches | 0 | `net` (every puller) |
| Fast interconnect first, LAN second | 0 | list `--from` addresses in that order |
| Local cache disk instead of tmpfs | 0 | operator: point the output path at the cache |
| Per-rank virtual layout (no overfetch) | 1 | `manifest --layout` |
| Expert-shard split by rank | 2 | `extents --shard r/n`, `pull --shard r/n` |
| Piece priority by layer order | 3 | library caller |

## What loom-swarm will not do

Create torrents (the manifest is the source of truth; any torrent client can carry the same pieces), convert or
quantize weights, or know anything about tensor contents beyond their byte ranges.
