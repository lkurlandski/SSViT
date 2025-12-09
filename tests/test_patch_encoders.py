# tests/test_patch_encoders.py

import math

import pytest
import torch

from src.architectures import PatchEncoderBase
from src.architectures import PatchEncoder
from src.architectures import ConvPatchEncoder
from src.architectures import HierarchicalConvPatchEncoder
from src.architectures import PatchEncoderLowMem


@pytest.mark.parametrize("T", [1, 2, 63, 64, 65, 512, 513])
@pytest.mark.parametrize("P", [1, 4, 16, 64, 256])
def test_compute_patch_dims_patch_size_mode(T: int, P: int) -> None:
    """In patch_size mode, P is fixed and N = ceil(T / P)."""
    patch_size = P
    num_patches = None

    P_res, N_res = PatchEncoderBase.compute_patch_dims(
        seq_length=T,
        patch_size=patch_size,
        num_patches=num_patches,
    )

    assert P_res == patch_size
    assert N_res == math.ceil(T / patch_size)
    assert P_res > 0
    assert N_res > 0
    assert P_res * N_res >= T


@pytest.mark.parametrize("T", [1, 2, 63, 64, 65, 512, 513, 4096])
@pytest.mark.parametrize("N", [1, 2, 4, 8, 16, 32])
def test_compute_patch_dims_num_patches_mode(T: int, N: int) -> None:
    """In num_patches mode, N is fixed and P = ceil(T / N)."""
    patch_size = None
    num_patches = N

    P_res, N_res = PatchEncoderBase.compute_patch_dims(
        seq_length=T,
        patch_size=patch_size,
        num_patches=num_patches,
    )

    assert N_res == num_patches
    assert P_res == math.ceil(T / num_patches)
    assert P_res > 0
    assert P_res * N_res >= T


def test_patchencoder_patch_size_mode_shapes() -> None:
    """PatchEncoder with fixed patch_size should output N = ceil(T / P) patches."""
    B, T, E, C = 3, 1000, 8, 16
    patch_size = 64
    kernel_size = 64  # must be <= patch_size
    stride = 64

    enc = PatchEncoder(
        in_channels=E,
        out_channels=C,
        num_patches=None,
        patch_size=patch_size,
        kernel_size=kernel_size,
        stride=stride,
    )

    z = torch.randn(B, T, E)
    out = enc.forward_embeddings(z)
    P_res, N_res = enc.resolve_patch_dims(T)

    assert P_res == patch_size
    assert N_res == math.ceil(T / patch_size)
    assert out.shape == (B, N_res, C)


def test_patchencoder_num_patches_mode_shapes_and_min_length() -> None:
    """
    PatchEncoder with fixed num_patches should:
      - output exactly num_patches patches for T >= min_length,
      - raise for T < min_length.
    """
    B, E, C = 2, 8, 16
    num_patches = 32
    kernel_size = 64
    stride = 64

    enc = PatchEncoder(
        in_channels=E,
        out_channels=C,
        num_patches=num_patches,
        patch_size=None,
        kernel_size=kernel_size,
        stride=stride,
    )

    # min_length = num_patches * kernel_size
    assert enc.min_length == num_patches * kernel_size

    # T exactly at min_length
    T_valid = enc.min_length
    z = torch.randn(B, T_valid, E)
    out = enc.forward_embeddings(z)
    assert out.shape == (B, num_patches, C)

    # T slightly above min_length
    T_valid2 = T_valid + 123
    z2 = torch.randn(B, T_valid2, E)
    out2 = enc.forward_embeddings(z2)
    assert out2.shape == (B, num_patches, C)

    # T below min_length should trigger the min_length guard in forward()
    T_too_short = enc.min_length - 1
    z_short = torch.randn(B, T_too_short, E)
    with pytest.raises(RuntimeError):
        _ = enc.forward(z_short)


def test_convpatchencoder_patch_size_mode_shapes() -> None:
    """ConvPatchEncoder should match the patch_size mode geometry."""
    B, T, E, C = 4, 1000, 8, 16
    patch_size = 64

    enc = ConvPatchEncoder(
        in_channels=E,
        out_channels=C,
        num_patches=None,
        patch_size=patch_size,
    )

    z = torch.randn(B, T, E)
    out = enc.forward_embeddings(z)
    P_res, N_res = enc.resolve_patch_dims(T)

    assert P_res == patch_size
    assert N_res == math.ceil(T / patch_size)
    assert out.shape == (B, N_res, C)


def test_hierarchicalconvpatchencoder_patch_size_mode_shapes() -> None:
    """HierarchicalConvPatchEncoder should follow the same patch geometry as ConvPatchEncoder."""
    B, T, E, C = 2, 1500, 8, 32
    patch_size = 64  # nice composite number for factoring

    enc = HierarchicalConvPatchEncoder(
        in_channels=E,
        out_channels=C,
        num_patches=None,
        patch_size=patch_size,
    )

    z = torch.randn(B, T, E)
    out = enc.forward_embeddings(z)
    P_res, N_res = enc.resolve_patch_dims(T)

    assert P_res == patch_size
    assert N_res == math.ceil(T / patch_size)
    assert out.shape == (B, N_res, C)


def test_patchencoderlowmem_num_patches_mode_shapes_and_min_length() -> None:
    """
    PatchEncoderLowMem must:
      - output exactly num_patches patches for T >= min_length,
      - raise for T < min_length (via forward()).
    """
    B, E, C = 2, 8, 16
    num_patches = 32
    kernel_size = 64
    stride = 64

    enc = PatchEncoderLowMem(
        in_channels=E,
        out_channels=C,
        num_patches=num_patches,
        patch_size=None,
        kernel_size=kernel_size,
        stride=stride,
    )

    assert enc.min_length == num_patches * kernel_size

    # T exactly at min_length
    T_valid = enc.min_length
    z = torch.randn(B, T_valid, E)
    out = enc.forward_embeddings(z)
    assert out.shape == (B, num_patches, C)

    # T slightly above min_length
    T_valid2 = T_valid + 123
    z2 = torch.randn(B, T_valid2, E)
    out2 = enc.forward_embeddings(z2)
    assert out2.shape == (B, num_patches, C)

    # T below min_length should be rejected by forward()
    T_too_short = enc.min_length - 1
    z_short = torch.randn(B, T_too_short, E)
    with pytest.raises(RuntimeError):
        _ = enc.forward(z_short)


def test_patchencoder_and_lowmem_agree_on_num_patches_geometry() -> None:
    """
    For the same (T, num_patches, kernel_size/stride), PatchEncoder and PatchEncoderLowMem
    should agree on:
      - N = num_patches
      - output shape (B, N, C)
    """
    B, E, C = 2, 8, 16
    num_patches = 16
    kernel_size = 64
    stride = 64

    T_valid = num_patches * kernel_size + 123  # some T >= min_length

    z = torch.randn(B, T_valid, E)

    enc_std = PatchEncoder(
        in_channels=E,
        out_channels=C,
        num_patches=num_patches,
        patch_size=None,
        kernel_size=kernel_size,
        stride=stride,
    )

    enc_lowmem = PatchEncoderLowMem(
        in_channels=E,
        out_channels=C,
        num_patches=num_patches,
        patch_size=None,
        kernel_size=kernel_size,
        stride=stride,
    )

    out_std = enc_std.forward_embeddings(z)
    out_low = enc_lowmem.forward_embeddings(z)

    assert out_std.shape == (B, num_patches, C)
    assert out_low.shape == (B, num_patches, C)
