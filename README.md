![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg)

# Mini tensor core v2 (Tiny Tapeout, GlobalFoundries 180 nm)

The GlobalFoundries 180 nm version of our [mini tensor core](https://github.com/tonystark101-debug/tt-minitensor-sky130). It computes **C += A × B** on 4×4 int8 matrices, with int32 totals.

## Changes from v1
- **C sits in a rotating ring**, so the adder always reads the front entry and no 16-way selector is needed. A full product rotates the ring 16 times, which puts every entry back in place.
- **A new command, COMPUTE_NEW (C = n·A×B), replaces CLEAR.** A, B and C are no longer reset.
- **Clock target is 25 MHz**, because GlobalFoundries 180 nm cells (3.3 V) are slower than SkyWater 130 nm.

## Status (local LibreLane run, same flow as this repo's GDS action)

| Check | Result |
|---|---|
| Layout rules (DRC), layout vs netlist (LVS), antenna | 0 errors |
| cocotb tests, RTL and gate-level | 6 / 6 pass |
| Timing at 25 MHz, typical corner (25 °C, 3.3 V) | +8.6 ns slack |
| Timing at 25 MHz, slow corner (125 °C, 3.0 V) | −16.4 ns (about 18 MHz worst case) |
| Size | 6 tiles (3×2), 1.08 × 0.33 mm, 55% utilization |

The 7 Dec 2026 GF shuttle (TTGF26c) was full when we checked. This design waits for a free slot.

Part of our open GPU research: [gpu-research](https://github.com/tonystark101-debug/gpu-research). License: Apache-2.0.
