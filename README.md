# loom-swarm

Content-addressed model distribution for a rack of diskless GPU nodes. One binary, no daemon dependencies.

Two layers, deliberately separate:

- **Wire.** A file is cut into fixed-size pieces (16 MiB by default) and each piece gets a BLAKE3 hash; the manifest
  (`<file>.swarm.json`) is the artifact's identity. Any transport that moves pieces and verifies them against the
  manifest works: this crate's own HTTP range puller, a torrent client, or an RDMA path later. It works on any file.
- **Semantic.** When the artifact parses as GGUF, a tensor-extent sidecar maps tensor name to byte range to piece
  indices. Expert-shard paging and partial layer pulls consume it: a tensor-parallel shard map is just a piece
  priority list. For engines that are not GGUF, `shards` records explicit per-rank byte ranges, and `manifest --layout`
  turns one rank's recorded read log into a per-rank layout artifact that is exactly that rank's bytes.

Measured on a 4-node Strix Halo pod (diskless workers, 124 GB RAM each, one 2.5 GbE link to the origin): two
tensor-parallel pairs cold-loading an 81 GiB model took 356 s from origin plus peers against 790 s for one pair over
NFS; a restart with the RAM disk kept took 15 s. A TP4 rank pulling its layout artifact fetched 36.7 GiB instead of 67.6.

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
production pod since 2026-10-03. Rust 2021, dependencies: anyhow, serde, serde_json, hex, blake3.
