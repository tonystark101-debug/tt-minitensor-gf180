/*
 * Copyright (c) 2026 The minitensor authors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Mini tensor core: computes C += A x B for 4x4 matrices.
 *   A, B: signed 8-bit (int8).   C: signed 32-bit accumulators (int32).
 * Four multipliers form one 4-wide dot-product unit (the building block of
 * GPU tensor cores). It produces one element of C per clock through a
 * 3-stage pipeline, so a full 4x4x4 product takes 16 clocks plus 3 to drain.
 *
 * C lives in a 16-entry ring that rotates once per element, so the adder
 * always reads entry 0 and no 16-way selector is needed. A pass rotates the
 * ring 16 times, which leaves every entry back in its home slot.
 *
 * Host interface (all pins synchronous to clk):
 *   ui_in[7:0]   DATA: write data / read address (bits 6:0) / repeat count
 *   uio_in[4:0]  ADDR: write address. 0-15 = A[i][k] at i*4+k,
 *                                     16-31 = B[k][j] at 16+k*4+j
 *   uio_in[6:5]  OP:   0 = none, 1 = WRITE,
 *                      2 = COMPUTE      (C += n * A x B)
 *                      3 = COMPUTE_NEW  (C  = n * A x B)
 *                      where n = DATA + 1 passes (1-256)
 *   uio_in[7]    STROBE: OP runs once per rising edge (2-flop synchronised;
 *                keep DATA/ADDR/OP stable for 3 clocks after raising it)
 *   uo_out[7:0]  read data, registered: the byte at read address ui_in[6:0]
 *     0x00-0x3F  C[e] byte b, e = addr[5:2] (row-major i*4+j), b = addr[1:0]
 *                (little-endian)
 *     0x40-0x4F  A bytes     0x50-0x5F  B bytes
 *     0x7E       status = {busy, 3'b0, done_count[3:0]}
 *     0x7F       ID = 0x4D
 * WRITE and COMPUTE are ignored while busy. A, B and C are not reset:
 * write A and B, and use COMPUTE_NEW for the first product.
 */

`default_nettype none

module tt_um_tonystark101_minitensor (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

  localparam integer ACC_W = 32;
  localparam [7:0] ID_BYTE = 8'h4D;
  localparam [1:0] OP_WRITE = 2'd1, OP_COMPUTE = 2'd2, OP_COMPUTE_NEW = 2'd3;

  // All bidirectional pins are inputs.
  assign uio_out = 8'h00;
  assign uio_oe  = 8'h00;

  // ---------------------------------------------------------------- strobe
  reg [2:0] strobe_sync;
  always @(posedge clk) begin
    if (!rst_n) strobe_sync <= 3'b000;
    else        strobe_sync <= {strobe_sync[1:0], uio_in[7]};
  end
  wire       fire = strobe_sync[1] & ~strobe_sync[2];
  wire [1:0] op   = uio_in[6:5];
  wire [4:0] addr = uio_in[4:0];

  // --------------------------------------------------------------- control
  reg        issuing;      // stage 1 is producing elements
  reg  [3:0] idx;          // element issued this cycle: i = idx[3:2], j = idx[1:0]
  reg  [7:0] passes_left;  // extra passes after the current one
  reg        first_pass;   // the current pass overwrites C (COMPUTE_NEW)
  reg        v1, v2;       // stage 2 / stage 3 hold valid data
  reg        ow1, ow2;     // ... and that data overwrites instead of adding
  reg  [3:0] done_count;   // COMPUTE commands completed (mod 16)
  wire       busy = issuing | v1 | v2;

  wire do_write   = fire && (op == OP_WRITE) && !busy;
  wire do_compute = fire && (op == OP_COMPUTE || op == OP_COMPUTE_NEW) && !busy;

  always @(posedge clk) begin
    if (!rst_n) begin
      issuing     <= 1'b0;
      idx         <= 4'd0;
      passes_left <= 8'd0;
      first_pass  <= 1'b0;
    end else if (do_compute) begin
      issuing     <= 1'b1;
      idx         <= 4'd0;
      passes_left <= ui_in;
      first_pass  <= (op == OP_COMPUTE_NEW);
    end else if (issuing) begin
      idx <= idx + 4'd1;
      if (idx == 4'd15) begin
        first_pass <= 1'b0;
        if (passes_left == 8'd0) issuing <= 1'b0;
        else                     passes_left <= passes_left - 8'd1;
      end
    end
  end

  always @(posedge clk) begin
    if (!rst_n) begin
      v1  <= 1'b0;
      v2  <= 1'b0;
      ow1 <= 1'b0;
      ow2 <= 1'b0;
    end else begin
      v1  <= issuing;
      ow1 <= issuing & first_pass;
      v2  <= v1;
      ow2 <= ow1;
    end
  end

  // The last element of the last pass leaves stage 3 with nothing behind it.
  always @(posedge clk) begin
    if (!rst_n)                     done_count <= 4'd0;
    else if (v2 && !v1 && !issuing) done_count <= done_count + 4'd1;
  end

  // ------------------------------------------------------ operand registers
  wire [127:0] a_flat;  // A[i][k] at bits (i*4+k)*8
  wire [127:0] b_flat;  // B[k][j] at bits (k*4+j)*8

  genvar e;
  generate
    for (e = 0; e < 16; e = e + 1) begin : g_ab
      localparam [4:0] A_ADDR = e;
      localparam [4:0] B_ADDR = e + 16;
      reg [7:0] a_q;
      reg [7:0] b_q;
      always @(posedge clk) begin
        if (do_write && addr == A_ADDR) a_q <= ui_in;
        if (do_write && addr == B_ADDR) b_q <= ui_in;
      end
      assign a_flat[e*8 +: 8] = a_q;
      assign b_flat[e*8 +: 8] = b_q;
    end
  endgenerate

  // ---------------------------------------- stage 1: select row/col, multiply
  wire [1:0] ri = idx[3:2];
  wire [1:0] cj = idx[1:0];

  wire signed [7:0] a0 = a_flat[{ri, 2'd0, 3'd0} +: 8];
  wire signed [7:0] a1 = a_flat[{ri, 2'd1, 3'd0} +: 8];
  wire signed [7:0] a2 = a_flat[{ri, 2'd2, 3'd0} +: 8];
  wire signed [7:0] a3 = a_flat[{ri, 2'd3, 3'd0} +: 8];
  wire signed [7:0] b0 = b_flat[{2'd0, cj, 3'd0} +: 8];
  wire signed [7:0] b1 = b_flat[{2'd1, cj, 3'd0} +: 8];
  wire signed [7:0] b2 = b_flat[{2'd2, cj, 3'd0} +: 8];
  wire signed [7:0] b3 = b_flat[{2'd3, cj, 3'd0} +: 8];

  reg signed [15:0] p0, p1, p2, p3;
  always @(posedge clk) begin
    p0 <= a0 * b0;
    p1 <= a1 * b1;
    p2 <= a2 * b2;
    p3 <= a3 * b3;
  end

  // ------------------------------------------------ stage 2: add the products
  wire signed [17:0] p0x = {{2{p0[15]}}, p0};
  wire signed [17:0] p1x = {{2{p1[15]}}, p1};
  wire signed [17:0] p2x = {{2{p2[15]}}, p2};
  wire signed [17:0] p3x = {{2{p3[15]}}, p3};
  reg  signed [17:0] sum2;
  always @(posedge clk) begin
    sum2 <= (p0x + p1x) + (p2x + p3x);
  end

  // ------------------------------------- stage 3: accumulate into the C ring
  wire [16*ACC_W-1:0] c_flat;  // slot s at bits s*ACC_W
  wire [ACC_W-1:0]    c_head = c_flat[0 +: ACC_W];
  wire [ACC_W-1:0]    c_base = ow2 ? {ACC_W{1'b0}} : c_head;
  wire [ACC_W-1:0]    c_new  = c_base + {{(ACC_W-18){sum2[17]}}, sum2};

  generate
    for (e = 0; e < 16; e = e + 1) begin : g_c
      reg  [ACC_W-1:0] c_q;
      wire [ACC_W-1:0] c_next;
      if (e == 15) begin : g_tail
        assign c_next = c_new;                         // updated head re-enters at the tail
      end else begin : g_body
        assign c_next = c_flat[(e+1)*ACC_W +: ACC_W];  // everyone else moves down one slot
      end
      always @(posedge clk) begin
        if (v2) c_q <= c_next;
      end
      assign c_flat[e*ACC_W +: ACC_W] = c_q;
    end
  endgenerate

  // ------------------------------------------------------------- read port
  wire [6:0] raddr = ui_in[6:0];
  reg  [7:0] rdata;
  always @(posedge clk) begin
    if (!rst_n) begin
      rdata <= 8'h00;
    end else begin
      casez (raddr)
        7'b0??_????: rdata <= c_flat[{raddr[5:0], 3'd0} +: 8];
        7'b100_????: rdata <= a_flat[{raddr[3:0], 3'd0} +: 8];
        7'b101_????: rdata <= b_flat[{raddr[3:0], 3'd0} +: 8];
        7'h7E:       rdata <= {busy, 3'b000, done_count};
        7'h7F:       rdata <= ID_BYTE;
        default:     rdata <= 8'h00;
      endcase
    end
  end
  assign uo_out = rdata;

  // List all unused inputs to prevent warnings
  wire _unused = &{ena, 1'b0};

endmodule
