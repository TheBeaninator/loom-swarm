# loom-swarm

Content-addressed model distribution for a rack of diskless GPU nodes. One binary, no daemon dependencies.

Two layers, deliberately separate:

- **Wire.** A file is cut into fixed-size pieces (16 MiB by default) and each piece gets a BLAKE3 hash; the manifest
  (`<file>.swarm.json`) is the artifact's identity. Any transport that moves pieces and verifies them against the
  manifest works: this crate's own HTTP range puller, a torrent client, or an RDMA path later. It works on any file.
- **Semantic.** When the artifact parses as GGUF or safetensors (each behind a default-on Cargo feature), a
  tensor-extent sidecar maps tensor name to byte range to piece indices. Expert-shard paging and partial layer pulls
  consume it: a tensor-parallel shard map is just a piece priority list. For any other format, `shards` records
  explicit per-rank byte ranges, and `manifest --layout` turns one rank's recorded read log into a per-rank layout
  artifact that is exactly that rank's bytes.

Measured on a 4-node Strix Halo pod (diskless workers, 124 GB RAM each, 100 GbE RoCE between workers, one 2.5 GbE link
to the origin), 2026-10-07, loading a 4-way tensor-parallel model from two peers that held a copy on local cache disks:

| | per rank |
|---|---|
| Bytes pulled with a per-rank layout artifact | 36.71 GiB (67.6 GiB with whole-file pieces) |
| Pull time, origin plus peers over the RoCE rail | 9.3 s (3.95 GiB/s) |
| Engine load from the seeded copy, all four ranks | 5.2 to 6.0 s |
| Same load straight over NFS | 274.5 s |
| Pieces verified against the manifest | every one; 113,337 tensor hashes compared against the NFS-loaded copy, 0 mismatches |
| Where the bytes came from (one rank) | 29.8 GiB from its TP partner over the rail, 17.3 and 15.5 GiB from the two cache peers, about 1 GiB each over the LAN, 0 from the origin |

The origin served nothing in that run: once two nodes held copies, the whole load came from peers, and the fastest
link took the most work without any configuration beyond listing addresses in preference order. A full 4-rank
tensor-parallel model went from cold to serving in about 15 s, where NFS had taken four and a half minutes.

Without the rail (2.5 GbE only, 2026-10-03): two tensor-parallel pairs cold-loaded an 81 GiB model in 356 s from origin
plus peers, against 790 s for one pair over NFS; a restart with the seeded copy kept took 15 s.

## Verbs

```
loom-swarm manifest  <file> [--piece-mib N] [--threads N] [--layout <ranges>] [-o out.json]
loom-swarm verify    <file> --manifest m.json [--threads N]
loom-swarm extents   --manifest m.json --match <substring> [--match ...] | --shard r/n
loom-swarm shards    --manifest m.json --add r/n=<ranges file> [--add ...]      # explicit TP shards, non-GGUF
loom-swarm chunk     <file> --manifest m.json --store <dir>
loom-swarm rehydrate --manifest m.json --store <dir> -o <file>
loom-swarm serve     --listen addr:port --manifest m.json [--manifest ...]
loom-swarm pull      --manifest m.json --from origin [--from peer ...] [--shard r/n] -o <file>
loom-swarm seed      --want m.json=out[@r/n] [--want ...] --from origin [--from peer ...] --listen addr:port [--ready <file>]
```

`seed` is the worker-side verb: it pulls what it wants from the origin and from every peer that already holds pieces,
verifies each piece, writes a per-rank pull at its real offsets into a sparse full-size file, and keeps serving so the
next node loads from peers instead of queueing on the origin's link.

## Typical deployment

1. Origin (the node with the weights on disk): `loom-swarm manifest model.gguf` once per file, then
   `loom-swarm serve --listen 0.0.0.0:7150 --manifest model.gguf.swarm.json` as a service.
2. Each worker: `loom-swarm seed --want model.gguf.swarm.json=/srv/swarm/model.gguf@0/2 --from http://origin:7150
   --from http://peer-a:7150 --listen 0.0.0.0:7150 --ready /run/swarm.ready`, pointing the inference engine at
   `/srv/swarm/model.gguf` (a tmpfs, or a local cache disk when the node has one) once the ready file appears.
3. Prefer a fast interconnect for peers: list a peer's RDMA or 100 GbE address first and its LAN address second.

Torrent creation is intentionally not here. The manifest is the source of truth; libtorrent or any other client can
move the same pieces.

## Status

Extracted from the Loom workspace (crate `loom-swarm`, 2026-10) where it has been loading tensor-parallel shards on a
production pod since 2026-10-03. Development continues there; `tools/sync-from-loom.sh` brings `src/` across one way
and records the Loom commit in `.loom-sync`. Rust 2021, dependencies: anyhow, serde, serde_json, hex, blake3.
Features: `gguf`, `safetensors` (both default); `--no-default-features` leaves the wire layer only.

## License

MIT. The crate is also a library: `manifest`, `gguf` and `net` are public modules. See docs/ENGINES.md for what an
inference engine provides to use each tier.
