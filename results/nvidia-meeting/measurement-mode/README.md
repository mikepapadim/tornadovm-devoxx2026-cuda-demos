# ncu and nsys disagree on the *ratio*, not just the absolute time

A methodological result that constrains every comparison in this bundle, and a
correction to how task B2's headline figure was phrased.

## The observation

Demo 15's three kernels, **launch geometry verified identical on both sides**
(block=256, grid=16384, confirmed from `launch__block_size` / `launch__grid_size`
rather than from source comments), `n = 4194304`:

| Kernel | nsys, steady state (20 exec) | ncu, cold single launch |
|---|---|---|
| `elementwise` | 13,944 / 10,621 ns → **1.31** | 20,128 / 19,744 ns → **1.02** |
| `stencil` | 14,322 / 11,546 ns → **1.24** | 20,256 / 19,424 ns → **1.04** |
| `polynomial` | 35,236 / 39,926 ns → **0.88** | 41,312 / 44,928 ns → **0.92** |

(TornadoVM / hand-written CUDA.)

**The memory-bound ratios collapse from 1.31 and 1.24 under `nsys` to 1.02 and
1.04 under `ncu`.** The compute-bound ratio barely moves.

## Why

`ncu` serialises launches, flushes caches between them and does not allow clock
boost. Under those conditions both implementations become DRAM-latency-bound in
the same way, and the extra L1↔L2 sector traffic from the 16-byte payload offset
is largely hidden. Under `nsys` in steady state the data is L2-resident and the
clocks are boosted, so sector count is much closer to being the limiter and the
penalty shows.

The compute-bound kernel is insensitive because its bottleneck is the FMA
dependency chain either way.

## Consequence — and a correction

The bundle already said "`ncu` kernel times are not comparable to `nsys` kernel
times". That was too weak. **Ratios computed under one mode are not comparable
to ratios computed under the other**, even with geometry, problem size and
correctness all controlled.

Task B2's decomposition — 1.075 × 1.131 = 1.216 — was measured entirely under
`ncu` at `n = 16777216`. It is internally consistent and its *structural*
conclusions hold:

- sectors/request is 5.00 vs 4.00 at both block sizes, so the alignment penalty
  is real and geometry-independent;
- the 256→1024 block penalty is 1.131 on TornadoVM and 1.159 on hand-written
  CUDA, so it is not a TornadoVM property;
- the two effects multiply to the uncontrolled ratio exactly.

But **its 1.075 figure is an `ncu`-condition number and must not be quoted as
"TornadoVM is 1.075x slower"**, nor presented as superseding demo 15's 1.31 and
1.24. Those are `nsys` steady-state measurements at matched geometry and are the
figures that describe realistic operation.

An earlier revision of this bundle, and a summary given alongside it, framed
1.075 as "the defensible number" replacing 1.20–1.31. That was wrong on two
counts: it mixed measurement modes, and it swept in demo 15's figures, which
were never geometry-confounded in the first place. Only **demo 12's** 1.20–1.23
was confounded, and that retraction stands.

## What to quote

| Question | Figure | Source |
|---|---|---|
| How does generated code perform in realistic steady state, matched geometry? | **1.31 / 1.24 memory-bound; 0.88 compute-bound** | demo 15 under `nsys` |
| Is the memory gap layout or code generation? | layout — offset sweep on one binary reproduces it | task B |
| How much of an uncontrolled ratio is block size? | 1.131 of 1.216 | task B2, `ncu` |
| Does the alignment penalty persist across block sizes? | yes, 5.00 vs 4.00 at both | task B2, `ncu` |

## Files

`demo15-launch-geometry-and-ncu-times.csv` — the geometry verification and `ncu`
times behind the right-hand column above.
