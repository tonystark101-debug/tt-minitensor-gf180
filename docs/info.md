<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This is a tiny version of the AI unit (a "tensor core") inside a GPU. It multiplies two 4×4 matrices of signed 8-bit numbers (int8) and adds the result into a 4×4 matrix of signed 32-bit totals (int32):

**C += A × B**

- **One element per clock.** Four multipliers work together as one 4-wide dot-product unit, the building block of GPU tensor cores. Each clock it finishes one element of C, which is one row of A times one column of B.
- **19 clocks per product.** A full 4×4×4 product takes 16 clocks, plus 3 to drain the pipeline.
- **C rotates.** C is kept in a 16-entry ring that rotates one step per element, so the adder always works on the entry at the front. After each full product, every entry is back in its own place.
- **Repeat to test top speed.** A command can repeat the product up to 256 times (n × A × B). That keeps the multipliers busy for about 4,100 clocks, long enough to find the fastest clock at which the chip still gets the right answer.

### Pins

| Pin | Meaning |
|---|---|
| ui_in[7:0] | DATA for WRITE. Repeat count for COMPUTE (n = DATA + 1). Read address (bits 6:0) the rest of the time. |
| uio_in[4:0] | ADDR for WRITE: 0–15 = A[i][k] at i·4+k, 16–31 = B[k][j] at 16+k·4+j |
| uio_in[6:5] | OP: 0 = none, 1 = WRITE, 2 = COMPUTE (C += n·A×B), 3 = COMPUTE_NEW (C = n·A×B) |
| uio_in[7] | STROBE: the OP runs once on each rising edge |
| uo_out[7:0] | The byte at the read address, one clock later |

All uio pins are inputs.

### Read addresses

| Address | Contents |
|---|---|
| 0x00–0x3F | C. Element e = addr[5:2] (row-major, i·4+j), byte addr[1:0] (little-endian) |
| 0x40–0x4F | A bytes |
| 0x50–0x5F | B bytes |
| 0x7E | Status: bit 7 = busy, bits 3:0 = completed COMPUTE commands (mod 16) |
| 0x7F | ID = 0x4D |

### Sending a command

1. Set DATA, ADDR and OP with STROBE low.
2. Raise STROBE. Keep DATA, ADDR and OP steady for at least 3 clocks, because STROBE passes through a 2-flop synchroniser.
3. Lower STROBE.

While the unit is busy:
- WRITE and COMPUTE are ignored.
- C reads as it rotates, so read C only when the unit is idle.

Reset doesn't clear A, B or C. Write A and B, and use COMPUTE_NEW for the first product.

## How to test

1. Reset the design and read address 0x7F. It should return 0x4D.
2. WRITE the 16 bytes of A (ADDR 0–15) and the 16 bytes of B (ADDR 16–31).
3. Send COMPUTE_NEW with DATA = 0.
4. Read address 0x7E until bit 7 is 0.
5. Read C from addresses 0x00–0x3F and compare it with A × B computed on the RP2040.

To find the top clock speed:

1. Load A and B at a slow clock.
2. Raise the clock and send COMPUTE_NEW with DATA = 255 (256 passes).
3. Lower the clock again, read C, and compare it with 256 × A × B.
4. Repeat at a higher clock each time, until the results go wrong.

## External hardware

None. The demo board's RP2040 drives every input and reads every output.
