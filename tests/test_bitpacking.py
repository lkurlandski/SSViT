"""
Tests.
"""

import math
from typing import Callable
from typing import Protocol

import pytest
import numpy as np
import torch
from torch import Tensor
from torch import BoolTensor
from torch import ByteTensor

from src.binanal import get_ranges_numpy
from src.bitpacking import pack_bool_tensor
from src.bitpacking import unpack_bit_tensor
from src.bitpacking import smallest_unsigned_integer_dtype
from src.bitpacking import packbits
from src.bitpacking import unpackbits
from src.bitpacking import slice_bitpacked_tensor
from src.bitpacking import _slice_bitpacked_tensor_from_mask_general
from src.bitpacking import _slice_bitpacked_tensor_from_mask_bigchunks


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("batch_size", list(range(10)))
@pytest.mark.parametrize("num_channels", list(range(1, 65)))
def test_pack_bool_tensor(ndim: int, batch_size: int, num_channels: int) -> None:
    size = [batch_size, num_channels]
    for k in range(2, ndim):
        size.insert(0, k)
    size = tuple(size)
    b = torch.randint(0, 2, size, dtype=torch.bool)
    x = pack_bool_tensor(b)
    assert x.shape == size[:-1]
    assert x.dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64)
    assert x.device == b.device


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("batch_size", list(range(10)))
@pytest.mark.parametrize("num_channels", list(range(1, 65)))
@pytest.mark.parametrize("otype", [torch.bool, torch.int8, torch.uint8, torch.uint16, torch.int16, torch.int32, torch.float16, torch.float32])
def test_unpack_bit_tensor(ndim: int, batch_size: int, num_channels: int, otype: torch.dtype) -> None:
    dtype = smallest_unsigned_integer_dtype(num_channels)
    size = [batch_size]
    for k in range(2, ndim):
        size.insert(0, k)
    size = tuple(size)
    x = torch.from_numpy(np.random.randint(0, torch.iinfo(dtype).max, size=size, dtype=np.uint64)).to(dtype)
    b = unpack_bit_tensor(x, num_channels)
    assert b.shape == size + (num_channels,)
    assert b.dtype == torch.bool
    assert b.device == x.device
    y = unpack_bit_tensor(x, num_channels, dtype=otype)
    assert y.shape == size + (num_channels,)
    assert y.dtype == otype
    assert y.device == x.device
    assert torch.all(y == b.to(otype))


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("batch_size", list(range(10)))
@pytest.mark.parametrize("num_channels", list(range(1, 65)))
def test_pack_then_unpack_tensor(ndim: int, batch_size: int, num_channels: int) -> None:
    size = [batch_size, num_channels]
    for k in range(2, ndim):
        size.insert(0, k)
    size = tuple(size)
    b_1 = torch.randint(0, 2, size, dtype=torch.bool)
    x = pack_bool_tensor(b_1)
    b_2 = unpack_bit_tensor(x, num_channels)
    assert b_1.shape == b_2.shape
    assert b_1.dtype == b_2.dtype
    assert b_1.device == b_2.device
    assert torch.all(b_1 == b_2)


@pytest.mark.parametrize("a", [1, 2, 7, 8, 9])
@pytest.mark.parametrize("b", [1, 2, 15, 16, 17])
@pytest.mark.parametrize("c", [1, 2, 31, 32, 33])
@pytest.mark.parametrize("d", [1, 2, 63, 64, 65])
@pytest.mark.parametrize("axis", [0, 1, 2, 3, -1])
def test_packunpackbits(a: int, b: int, c: int, d: int, axis: int) -> None:
    print(f"{a=} {b=} {c=} {d=} {axis=}")
    if all(t == 1 for t in (a, b, c, d)):
        return

    x = torch.randint(0, 2, (a, b, c, d), dtype=torch.bool).squeeze()
    axis = min(axis, x.ndim - 1)
    axis_ = axis if axis >= 0 else x.ndim + axis

    y = packbits(x, axis)
    assert y.dtype == torch.uint8
    assert y.shape == x.shape[:axis_] + (x.shape[axis] // 8 + (x.shape[axis] % 8 > 0),) + x.shape[axis_ + 1:]

    z = unpackbits(y, x.shape[axis], axis=axis)
    assert z.dtype == torch.bool
    assert z.shape == x.shape
    assert torch.all(x == z)

    w = unpackbits(y, axis=axis)
    assert w.dtype == torch.bool
    assert w.shape == x.shape[:axis_] + (math.ceil(x.shape[axis] / 8) * 8,) + x.shape[axis_ + 1:]
    w_ = w.movedim(axis, 0)[0:x.shape[axis]].movedim(0, axis)
    assert torch.all(x == w_)


class TestSliceBitpackedTensor:

    @pytest.mark.parametrize("mode", ["mask", "ranges"])  # NOTE: "idx" mode not implemented yet.
    @pytest.mark.parametrize("bigchunks", [False, True])
    @pytest.mark.parametrize("axis", [0, 1, -1])
    @pytest.mark.parametrize("length", [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129])
    @pytest.mark.parametrize("ndim", [1, 2, 3, 4])
    def test_random_selection(self, mode: str, bigchunks: bool, axis: int, length: int, ndim: int) -> None:
        torch.manual_seed(0)
        if ndim == 1 and axis > 0:
            return

        size = [torch.randint(1, length + 1, size=(1,)).item() for _ in range(ndim)]
        size[axis] = length
        size = tuple(size)
        mask: Tensor = torch.randint(0, 2, size=(size[axis],), dtype=torch.bool)
        num_selected = int(mask.sum())

        original = torch.randint(0, 2, size=size, dtype=torch.bool)
        poriginal = packbits(original, axis=axis)
        reference: Tensor = original.movedim(axis, -1)[..., mask].movedim(-1, axis)
        if num_selected > 0:
            preference = packbits(reference, axis=axis)
        else:
            with pytest.raises(ValueError):
                preference = packbits(reference, axis=axis)
            axis_ = axis if axis >= 0 else original.ndim + axis
            preference = torch.empty(size[:axis_] + (0,) + size[axis_ + 1:], dtype=torch.uint8)

        if mode == "mask":
            idx = None
            ranges = None
        if mode == "idx":
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            ranges = None
            mask = None
        if mode == "ranges":
            lo, hi = get_ranges_numpy(mask.numpy())
            ranges = list(zip(lo.tolist(), hi.tolist()))
            idx = None
            mask = None

        psliced = slice_bitpacked_tensor(poriginal, mask=mask, idx=idx, ranges=ranges, bigchunks=bigchunks, axis=axis)

        assert psliced.dtype == torch.uint8
        assert psliced.device == poriginal.device
        for i in range(psliced.ndim):
            if i == (axis if axis >= 0 else psliced.ndim + axis):
                assert psliced.shape[i] <= preference.shape[i]
            else:
                assert psliced.shape[i] == poriginal.shape[i]

        if num_selected > 0:
            sliced = unpackbits(psliced, reference.shape[axis], axis=axis)
        else:
            with pytest.raises(ValueError):
                sliced = unpackbits(psliced, reference.shape[axis], axis=axis)
            axis_ = axis if axis >= 0 else original.ndim + axis
            sliced = torch.empty(size[:axis_] + (0,) + size[axis_ + 1:], dtype=torch.bool)
        assert sliced.dtype == reference.dtype
        assert sliced.shape == reference.shape
        assert torch.all(sliced == reference)


MASKED_SELECT_PACKED = [_slice_bitpacked_tensor_from_mask_general, _slice_bitpacked_tensor_from_mask_bigchunks]


class _MaskSelectPacked(Protocol):

    def __call__(self, packed: ByteTensor, mask: BoolTensor, axis: int) -> ByteTensor:
        ...


class TestSliceBitpackedTensorLegacy:

    @pytest.mark.parametrize(
        "shape,axis",
        [
            ((25,), 0),
            ((4, 25), -1),
            ((4, 25), 1),
            ((25, 4), 0),
            ((2, 3, 25), -1),
            ((2, 25, 3), 1),
            ((25, 2, 3), 0),
        ],
    )
    @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    def test_matches_reference_random(self, shape: tuple[int], axis: int, mask_select_packed: _MaskSelectPacked) -> None:
        torch.manual_seed(0)
        # create random bool data and mask
        d = shape[axis if axis >= 0 else len(shape) + axis]
        x = (torch.rand(shape) < 0.5)
        mask = (torch.rand(d) < 0.6)  # some true, some false

        packed = packbits(x, axis=axis)
        got = mask_select_packed(packed, mask, axis=axis)

        # Reference: select on the original bool axis, then pack
        x_sel = x.movedim(axis, -1)[..., mask].movedim(-1, axis)
        ref = packbits(x_sel, axis=axis)

        assert got.dtype == torch.uint8
        assert torch.equal(got, ref)

    @pytest.mark.parametrize("shape,axis", [((17,), 0), ((3, 17), -1), ((3, 17), 1)])
    @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    def test_full_selection_is_noop(self, shape: tuple[int], axis: int, mask_select_packed: _MaskSelectPacked) -> None:
        d = shape[axis if axis >= 0 else len(shape) + axis]
        x = (torch.arange(int(torch.prod(torch.tensor(shape)))) % 2 == 0).reshape(shape)
        x = x.to(torch.bool)
        mask = torch.ones(d, dtype=torch.bool)

        packed = packbits(x, axis=axis)
        got = mask_select_packed(packed, mask, axis=axis)

        assert torch.equal(got, packed)  # selecting all bits should be identity

    @pytest.mark.parametrize("shape,axis", [((17,), 0), ((2, 17), 1), ((2, 5, 17), -1)])
    @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    def test_empty_selection_has_zero_bytes(self, shape: tuple[int], axis: int, mask_select_packed: _MaskSelectPacked) -> None:
        d = shape[axis if axis >= 0 else len(shape) + axis]
        x = (torch.rand(shape) < 0.5)
        mask = torch.zeros(d, dtype=torch.bool)  # select nothing

        packed = packbits(x, axis=axis)
        got = mask_select_packed(packed, mask, axis=axis)

        # Expect zero-length along the byte axis
        pos_axis = axis if axis >= 0 else got.ndim + axis
        assert got.shape[pos_axis] == 0
        assert got.dtype == torch.uint8

    @pytest.mark.parametrize("d", [1, 7, 8, 9, 15, 16, 17, 31, 32])
    @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    def test_various_lengths_and_padding(self, d: int, mask_select_packed: _MaskSelectPacked) -> None:
        torch.manual_seed(1)
        x = (torch.rand(d) < 0.5)
        mask = (torch.rand(d) < 0.5)
        packed = packbits(x, axis=0)
        got = mask_select_packed(packed, mask, axis=0)

        ref = packbits(x[mask], axis=0)
        assert torch.equal(got, ref)
        # bytes count matches ceil(k/8)
        k = int(mask.sum())
        assert got.numel() == (k + 7) // 8

    # NOTE: we moved the error checking to the public function slice_bitpacked_tensor,
    # so these tests are commented out because they all fail.
    # @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    # def test_error_wrong_packed_dtype(self, mask_select_packed: _MaskSelectPacked) -> None:
    #     packed = torch.randint(0, 256, (3,), dtype=torch.int16)  # wrong dtype
    #     mask = torch.ones(24, dtype=torch.bool)
    #     with pytest.raises(TypeError):
    #         mask_select_packed(packed, mask, axis=0)

    # @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    # def test_error_mask_not_bool(self, mask_select_packed: _MaskSelectPacked) -> None:
    #     packed = torch.randint(0, 256, (3,), dtype=torch.uint8)
    #     mask = torch.ones(24, dtype=torch.int64)  # not bool
    #     with pytest.raises(TypeError):
    #         mask_select_packed(packed, mask, axis=0)

    # @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    # def test_error_mask_not_1d(self, mask_select_packed: _MaskSelectPacked) -> None:
    #     packed = torch.randint(0, 256, (3,), dtype=torch.uint8)
    #     mask = torch.ones(24, 1, dtype=torch.bool)
    #     with pytest.raises(ValueError):
    #         mask_select_packed(packed, mask, axis=0)

    @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    def test_error_mask_too_long(self, mask_select_packed: _MaskSelectPacked) -> None:
        # 2 bytes => 16 bits available, but provide longer mask
        packed = torch.randint(0, 256, (2,), dtype=torch.uint8)
        mask = torch.ones(17, dtype=torch.bool)
        with pytest.raises(ValueError):
            mask_select_packed(packed, mask, axis=0)

    @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    def test_axis_out_of_range(self, mask_select_packed: _MaskSelectPacked) -> None:
        packed = torch.randint(0, 256, (2, 2), dtype=torch.uint8)
        mask = torch.ones(16, dtype=torch.bool)
        with pytest.raises(IndexError):
            mask_select_packed(packed, mask, axis=3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    def test_cuda_parity(self, mask_select_packed: _MaskSelectPacked) -> None:
        torch.manual_seed(123)
        x = (torch.rand(2, 25) < 0.5).cuda()
        mask = (torch.rand(25) < 0.5).cuda()

        packed = packbits(x, axis=-1)
        got = mask_select_packed(packed, mask, axis=-1)
        ref = packbits(x[..., mask], axis=-1)

        assert got.is_cuda and ref.is_cuda
        assert torch.equal(got.cpu(), ref.cpu())

    # Helper: safe reference (unpacked -> mask -> repack), works even when mask.sum()==0
    def _ref(self, x: torch.Tensor, mask: torch.BoolTensor, axis: int) -> torch.Tensor:
        if not bool(mask.any()):
            # produce a correctly-shaped empty packed tensor by slicing an existing packed
            base = packbits(x, axis=axis)
            pos_axis = axis if axis >= 0 else base.ndim + axis
            sl = [slice(None)] * base.ndim
            sl[pos_axis] = slice(0, 0)
            return base[tuple(sl)]
        x_sel = x.movedim(axis, -1)[..., mask].movedim(-1, axis)
        return packbits(x_sel, axis=axis)

    @pytest.mark.parametrize("B", [1, 2, 3, 4, 8, 64, 256, 512])
    @pytest.mark.parametrize("axis", [0, -1])
    @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    def test_trailing_zero_mask_bytes_multiple_of_8_no_oob(self, B: int, axis: int, mask_select_packed: _MaskSelectPacked) -> None:
        """
        Select exactly 8*m bits (multiple of 8) with m < B, leaving trailing zero mask bytes.
        Used to trigger OOB in scatter_add_.
        """
        d = 8 * B
        for m in [0, 1, max(0, B - 2), B - 1]:
            k = 8 * m
            shape = (d,) if axis == 0 else (3, d)          # d is the axis length
            x = (torch.arange(int(torch.prod(torch.tensor(shape)))) % 2 == 0).reshape(shape).to(torch.bool)
            mask = torch.zeros(d, dtype=torch.bool)
            mask[:k] = True

            packed = packbits(x, axis=axis)
            got = mask_select_packed(packed, mask, axis=axis)
            ref = self._ref(x, mask, axis)

            assert got.dtype == torch.uint8
            assert torch.equal(got, ref), f"Mismatch for B={B}, m={m}, axis={axis}"

    @pytest.mark.parametrize("B", [1, 2, 3, 4, 8, 64, 256, 512])
    @pytest.mark.parametrize("axis", [0, -1])
    @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    def test_trailing_zero_mask_bytes_non_multiple_of_8(self, B: int, axis: int, mask_select_packed: _MaskSelectPacked) -> None:
        """
        Select k = 8*m + r bits (r in {1,3,7}), ensuring at least one trailing zero mask byte.
        """
        d = 8 * B
        for m in [0, 1, max(0, B - 2)]:
            for r in [1, 3, 7]:
                k = 8 * m + r
                if k > d:
                    continue
                shape = (d,) if axis == 0 else (2, 3, d)     # put d on the last axis when axis=-1
                x = (torch.rand(shape) < 0.5)
                mask = torch.zeros(d, dtype=torch.bool)
                mask[:k] = True

                packed = packbits(x, axis=axis)
                got = mask_select_packed(packed, mask, axis=axis)
                ref = self._ref(x, mask, axis)

                assert torch.equal(got, ref), f"Mismatch for B={B}, m={m}, r={r}, axis={axis}"

    @pytest.mark.parametrize(
        "shape,axis",
        [
            ((8 * 5,), 0),         # 5 bytes on axis 0
            ((3, 8 * 7), -1),      # last axis has d
            ((8 * 9, 2), 0),       # axis 0 has d, extra dim
            ((2, 8 * 11, 3), 1),   # middle axis has d
        ],
    )
    @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    def test_random_sparse_masks_with_long_zero_runs(self, shape: tuple[int, ...], axis: int, mask_select_packed: _MaskSelectPacked) -> None:
        torch.manual_seed(1234)
        d = shape[axis if axis >= 0 else len(shape) + axis]
        x = (torch.rand(shape) < 0.5)

        mask = (torch.rand(d) < 0.3)
        if d >= 16:
            mask[-16:] = 0  # zero tail of 2 bytes
        if d >= 24:
            mid = d // 2
            mask[mid:mid + 8] = 0  # zero a middle byte

        packed = packbits(x, axis=axis)
        got = mask_select_packed(packed, mask, axis=axis)
        ref = self._ref(x, mask, axis)

        assert torch.equal(got, ref)

    @pytest.mark.parametrize("B", [32, 64, 128])
    @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    def test_large_dim_many_patterns_cpu(self, B: int, mask_select_packed: _MaskSelectPacked) -> None:
        """
        Fuzz k across byte boundaries; includes k==0 without calling packbits on empty.
        """
        torch.manual_seed(0)
        d = 8 * B
        x = (torch.rand(d) < 0.5)
        packed = packbits(x, axis=0)

        ks = list(range(0, d + 1, max(1, d // 17))) + [1, 7, 8, 9, d - 9, d - 8, d - 1, d]
        ks = sorted(set(k for k in ks if 0 <= k <= d))
        for k in ks:
            mask = torch.zeros(d, dtype=torch.bool)
            mask[:k] = True
            got = mask_select_packed(packed, mask, axis=0)
            ref = self._ref(x, mask, axis=0)
            assert torch.equal(got, ref), f"Mismatch for B={B}, k={k}"

    @pytest.mark.parametrize("mask_select_packed", MASKED_SELECT_PACKED)
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_trailing_zero_bytes_boundary(self, mask_select_packed: _MaskSelectPacked) -> None:
        """
        Specifically target the prior OOB on GPU (e.g., 512 bytes; select 8*511 bits).
        """
        B = 512
        d = 8 * B
        x = (torch.rand(d, device="cuda") < 0.5)
        mask = torch.zeros(d, dtype=torch.bool)
        mask[: 8 * (B - 1)] = True  # last byte all zeros

        packed = packbits(x, axis=0)
        got = mask_select_packed(packed, mask.cuda(), axis=0)
        ref = self._ref(x, mask, axis=0)
        assert torch.equal(got.cpu(), ref.cpu())
