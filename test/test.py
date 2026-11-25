# SPDX-FileCopyrightText: © 2026 The minitensor authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the mini tensor core: C += A x B on 4x4 int8 matrices, int32 C."""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

OP_NONE, OP_WRITE, OP_COMPUTE, OP_COMPUTE_NEW = 0, 1, 2, 3
ADDR_STATUS, ADDR_ID = 0x7E, 0x7F
ID_BYTE = 0x4D
MASK32 = (1 << 32) - 1


def s32(v):
    v &= MASK32
    return v - (1 << 32) if v & 0x80000000 else v


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def scaled(m, n):
    return [[s32(n * x) for x in row] for row in m]


def add(m1, m2):
    return [[s32(x + y) for x, y in zip(r1, r2)] for r1, r2 in zip(m1, m2)]


async def start(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())  # 50 MHz
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def command(dut, op, addr=0, data=0):
    """Set DATA/ADDR/OP, pulse STROBE, hold the inputs while it is synchronised."""
    ctrl = (op << 5) | (addr & 0x1F)
    dut.ui_in.value = data & 0xFF
    dut.uio_in.value = ctrl
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0x80 | ctrl
    await ClockCycles(dut.clk, 4)
    dut.uio_in.value = ctrl
    await ClockCycles(dut.clk, 3)
    dut.uio_in.value = 0


async def read_byte(dut, raddr):
    dut.uio_in.value = 0
    dut.ui_in.value = raddr & 0x7F
    await ClockCycles(dut.clk, 2)
    return dut.uo_out.value.to_unsigned()


async def wait_idle(dut, max_polls=5000):
    for _ in range(max_polls):
        status = await read_byte(dut, ADDR_STATUS)
        if not status & 0x80:
            return status
    raise AssertionError("still busy")


async def load(dut, a, b):
    for i in range(4):
        for k in range(4):
            await command(dut, OP_WRITE, i * 4 + k, a[i][k])
    for k in range(4):
        for j in range(4):
            await command(dut, OP_WRITE, 16 + k * 4 + j, b[k][j])


async def read_c(dut):
    c = [[0] * 4 for _ in range(4)]
    for e in range(16):
        v = 0
        for byte in range(4):
            v |= (await read_byte(dut, e * 4 + byte)) << (8 * byte)
        c[e // 4][e % 4] = s32(v)
    return c


async def compute(dut, passes=1, new=False):
    await command(dut, OP_COMPUTE_NEW if new else OP_COMPUTE, data=passes - 1)
    return await wait_idle(dut)


def rand_matrix(rng):
    return [[rng.randint(-128, 127) for _ in range(4)] for _ in range(4)]


@cocotb.test()
async def test_id_and_status_after_reset(dut):
    await start(dut)
    assert await read_byte(dut, ADDR_ID) == ID_BYTE
    assert await read_byte(dut, ADDR_STATUS) == 0
    assert await read_byte(dut, 0x60) == 0
    assert await read_byte(dut, 0x7D) == 0


@cocotb.test()
async def test_single_product_and_readback(dut):
    await start(dut)
    rng = random.Random(1)
    a, b = rand_matrix(rng), rand_matrix(rng)
    await load(dut, a, b)
    for i in range(4):
        for k in range(4):
            assert await read_byte(dut, 0x40 + i * 4 + k) == a[i][k] & 0xFF
            assert await read_byte(dut, 0x50 + i * 4 + k) == b[i][k] & 0xFF
    status = await compute(dut, new=True)
    assert status & 0x0F == 1, "done_count should count one COMPUTE"
    assert await read_c(dut) == matmul(a, b)


@cocotb.test()
async def test_accumulate_repeat_and_overwrite(dut):
    await start(dut)
    rng = random.Random(2)
    a, b = rand_matrix(rng), rand_matrix(rng)
    ref = matmul(a, b)
    await load(dut, a, b)
    await compute(dut, passes=3, new=True)
    assert await read_c(dut) == scaled(ref, 3)
    status = await compute(dut, passes=1)
    assert status & 0x0F == 2
    assert await read_c(dut) == scaled(ref, 4)
    await compute(dut, passes=256, new=True)
    assert await read_c(dut) == scaled(ref, 256)
    # accumulate a different product on top
    a2, b2 = rand_matrix(rng), rand_matrix(rng)
    await load(dut, a2, b2)
    await compute(dut, passes=2)
    assert await read_c(dut) == add(scaled(ref, 256), scaled(matmul(a2, b2), 2))


@cocotb.test()
async def test_extremes(dut):
    await start(dut)
    for av, bv in [(-128, -128), (-128, 127), (127, 127), (127, -1)]:
        a = [[av] * 4 for _ in range(4)]
        b = [[bv] * 4 for _ in range(4)]
        await load(dut, a, b)
        await compute(dut, passes=256, new=True)
        assert await read_c(dut) == scaled(matmul(a, b), 256), (av, bv)


@cocotb.test()
async def test_random_products(dut):
    await start(dut)
    rng = random.Random(3)
    for _ in range(6):
        a, b = rand_matrix(rng), rand_matrix(rng)
        passes = rng.randint(1, 8)
        await load(dut, a, b)
        await compute(dut, passes=passes, new=True)
        assert await read_c(dut) == scaled(matmul(a, b), passes)


@cocotb.test()
async def test_ignores_commands_while_busy(dut):
    await start(dut)
    rng = random.Random(4)
    a, b = rand_matrix(rng), rand_matrix(rng)
    await load(dut, a, b)
    await command(dut, OP_COMPUTE_NEW, data=255)  # 256 passes, about 4,100 clocks
    assert await read_byte(dut, ADDR_STATUS) & 0x80, "should be busy"
    await command(dut, OP_WRITE, 0, (a[0][0] + 1) & 0xFF)  # must be ignored
    await command(dut, OP_COMPUTE_NEW, data=0)  # must be ignored
    await wait_idle(dut)
    assert await read_byte(dut, 0x40) == a[0][0] & 0xFF
    assert await read_c(dut) == scaled(matmul(a, b), 256)
