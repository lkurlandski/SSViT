"""
Tests.
"""

from typing import Any

import pytest
import torch
from torch import Tensor
import torch.nn.functional as F

from src.architectures import ShardedTokenEmbedding


def _build_expected_maps(
    num_embeddings: int, shard_tokens: dict[int, int],
) -> tuple[Tensor, Tensor, dict[int, tuple[int, int]], int]:
    """
    Reconstruct expected base_map / mask_map from constructor args,
    assuming shard_tokens are sorted by token id.
    """
    base = torch.arange(num_embeddings, dtype=torch.int32)
    mask = torch.zeros(num_embeddings, dtype=torch.int32)

    next_free = num_embeddings
    shard_blocks: dict[int, tuple[int, int]] = {}
    for tok in sorted(shard_tokens.keys()):
        m = shard_tokens[tok]
        shard_blocks[tok] = (next_free, m)
        base[tok] = next_free
        mask[tok] = m - 1
        next_free += m

    return base, mask, shard_blocks, next_free


def _expected_remap(input_ids: torch.Tensor, num_embeddings: int, shard_tokens: dict[int, int], row_hash_stride: int) -> Tensor:
    """
    Compute expected remapped indices exactly from constructor args.
    Uses int64 math (safe for these small tests).
    """
    assert input_ids.dim() == 2
    B, T = input_ids.shape

    base_map, mask_map, _, _ = _build_expected_maps(num_embeddings, shard_tokens)

    # Gather maps by token id
    base = base_map[input_ids.to(torch.long)].to(torch.int64)
    mask = mask_map[input_ids.to(torch.long)].to(torch.int64)

    pos = torch.arange(T, dtype=torch.int64).view(1, T)
    row = (torch.arange(B, dtype=torch.int64) * int(row_hash_stride)).view(B, 1)
    shard = pos + row

    remapped = base + (shard & mask)
    return remapped


def _set_row_id_weights_(emb: ShardedTokenEmbedding) -> None:
    """
    Make row r produce embedding vector [r, r, ..., r].
    Useful for testing exact remap behavior from outputs.
    """
    with torch.no_grad():
        W = emb.embedding.weight
        rows = torch.arange(W.shape[0], device=W.device, dtype=W.dtype).view(-1, 1)
        W.copy_(rows.expand_as(W))


def test_internal_size_and_maps_are_correct() -> None:
    num_embeddings = 10
    embedding_dim = 4
    shard_tokens = {7: 2, 2: 4}  # intentionally unsorted input
    emb = ShardedTokenEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        shard_tokens=shard_tokens,
    )

    exp_base, exp_mask, exp_blocks, exp_internal = _build_expected_maps(
        num_embeddings, shard_tokens
    )

    assert emb.num_embeddings_conceptual == num_embeddings
    assert emb.num_embeddings_internal == exp_internal
    assert emb.embedding.num_embeddings == exp_internal
    assert emb._shard_blocks == exp_blocks

    torch.testing.assert_close(emb.base_map.cpu(), exp_base)
    torch.testing.assert_close(emb.mask_map.cpu(), exp_mask)


@pytest.mark.parametrize("input_dtype", [torch.int64, torch.int32])
def test_forward_remap_exact_indices(input_dtype: torch.dtype) -> None:
    num_embeddings = 8
    embedding_dim = 1  # easier to decode row IDs from output
    row_hash_stride = 5
    shard_tokens = {1: 4, 3: 2}

    emb = ShardedTokenEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        shard_tokens=shard_tokens,
        row_hash_stride=row_hash_stride,
    )

    _set_row_id_weights_(emb)

    # Mix sharded and non-sharded tokens
    x = torch.tensor(
        [
            [0, 1, 1, 3, 4, 1, 3, 7],
            [1, 1, 2, 3, 3, 5, 1, 0],
        ],
        dtype=input_dtype,
    )

    y = emb(x).squeeze(-1)  # shape (B, T), values should equal remapped row index
    expected = _expected_remap(
        x.to(torch.long), num_embeddings=num_embeddings, shard_tokens=shard_tokens, row_hash_stride=row_hash_stride
    ).to(y.dtype)

    torch.testing.assert_close(y.cpu(), expected.cpu())


def test_padding_outputs_zero_for_unsharded_and_sharded_padding() -> None:
    num_embeddings = 12
    embedding_dim = 3
    shard_tokens = {2: 4, 5: 2}
    padding_idx = (4, 5)  # 4 unsharded padding, 5 sharded padding

    emb = ShardedTokenEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        shard_tokens=shard_tokens,
        padding_idx=padding_idx,
        row_hash_stride=7,
    )

    # Make weights nonzero so zero outputs really prove masking works
    with torch.no_grad():
        emb.embedding.weight.normal_(mean=1.0, std=0.1)

    x = torch.tensor(
        [
            [4, 5, 1, 2, 5, 4],
            [5, 4, 3, 5, 4, 6],
        ],
        dtype=torch.int64,
    )
    y = emb(x)  # (B, T, E)

    pad_mask = (x == 4) | (x == 5)
    assert torch.all(y[pad_mask] == 0)

    # padding_idx_internal should include:
    # - 4 (unsharded padding)
    # - 5 original row (sharded padding original row)
    # - shard rows for 5
    # - original row for token 2 (because all sharded originals are frozen)
    frozen = set(emb.padding_idx_internal.cpu().tolist())
    b5, m5 = emb._shard_blocks[5]
    expected_subset = {4, 5, *range(b5, b5 + m5), 2}
    assert expected_subset.issubset(frozen)


def test_backward_zero_grad_for_frozen_rows() -> None:
    num_embeddings = 16
    embedding_dim = 4
    shard_tokens = {2: 4, 9: 2}
    padding_idx = (3, 9)  # include one sharded padding token

    emb = ShardedTokenEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        shard_tokens=shard_tokens,
        padding_idx=padding_idx,
        row_hash_stride=11,
    )

    x = torch.tensor(
        [
            [2, 2, 3, 9, 1, 4, 2],
            [9, 3, 2, 5, 6, 2, 9],
        ],
        dtype=torch.int64,
    )

    out = emb(x)
    loss = out.sum()
    loss.backward()

    grad = emb.embedding.weight.grad
    assert grad is not None

    frozen_rows = emb.padding_idx_internal.cpu()
    assert torch.all(grad[frozen_rows.to(grad.device)] == 0)

    # At least one non-frozen row should get nonzero grad
    nonzero_rows = (grad.abs().sum(dim=1) > 0).nonzero(as_tuple=False).flatten().cpu().tolist()
    frozen_set = set(frozen_rows.tolist())
    assert any(r not in frozen_set for r in nonzero_rows)


def test_zero_frozen_rows_sets_them_to_zero() -> None:
    emb = ShardedTokenEmbedding(
        num_embeddings=10,
        embedding_dim=3,
        shard_tokens={2: 4},
        padding_idx=(1,),
    )

    with torch.no_grad():
        emb.embedding.weight.fill_(7.0)

    emb.zero_frozen_rows_()

    frozen = emb.padding_idx_internal.to(emb.embedding.weight.device)
    assert torch.all(emb.embedding.weight[frozen] == 0)

    # At least one non-frozen row remains nonzero
    all_rows = torch.arange(emb.embedding.weight.shape[0], device=frozen.device)
    mask = torch.ones_like(all_rows, dtype=torch.bool)
    mask[frozen] = False
    if mask.any():
        assert torch.any(emb.embedding.weight[all_rows[mask]] != 0)


def test_tie_shards_averages_each_shard_block() -> None:
    emb = ShardedTokenEmbedding(
        num_embeddings=10,
        embedding_dim=2,
        shard_tokens={2: 4, 7: 2},
    )

    with torch.no_grad():
        # Make shard blocks have distinct rows so tie_shards_ must change them
        for tok, (b, m) in emb._shard_blocks.items():
            vals = torch.arange(m, dtype=emb.embedding.weight.dtype).view(m, 1)
            emb.embedding.weight[b:b+m].copy_(vals.expand(m, emb.embedding_dim))

    # Capture expected means before tying
    expected_means = {}
    with torch.no_grad():
        for tok, (b, m) in emb._shard_blocks.items():
            expected_means[tok] = emb.embedding.weight[b:b+m].mean(dim=0, keepdim=True).clone()

    emb.tie_shards_()

    with torch.no_grad():
        for tok, (b, m) in emb._shard_blocks.items():
            block = emb.embedding.weight[b:b+m]
            exp = expected_means[tok].expand_as(block)
            torch.testing.assert_close(block, exp)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(num_embeddings=8, embedding_dim=4, padding_idx=(8,)), "padding token id"),
        (dict(num_embeddings=8, embedding_dim=4, padding_idx=(1, 1)), "duplicates"),
        (dict(num_embeddings=8, embedding_dim=4, shard_tokens={1: 1}), "must be >= 2"),
        (dict(num_embeddings=8, embedding_dim=4, shard_tokens={1: 3}), "power-of-two"),
        (dict(num_embeddings=8, embedding_dim=4, shard_tokens={8: 2}), "out of range"),
    ],
)
def test_invalid_configs_raise(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ShardedTokenEmbedding(**kwargs)


def test_internal_size_bound_check_raises() -> None:
    # Small conceptual vocab, but enormous shard count forces internal > 2^31 - 1
    with pytest.raises(ValueError, match="<= 2\\^31-1"):
        ShardedTokenEmbedding(
            num_embeddings=4,
            embedding_dim=2,
            shard_tokens={0: 2**31},  # also power-of-two and >=2
        )


def test_max_T_and_max_B_guards() -> None:
    emb = ShardedTokenEmbedding(
        num_embeddings=16,
        embedding_dim=4,
        shard_tokens={2: 4},
        max_T=4,
        max_B=2,
    )

    x_ok = torch.tensor([[1, 2, 3, 4], [2, 2, 2, 2]], dtype=torch.int64)
    _ = emb(x_ok)  # should not raise

    x_too_long = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    with pytest.raises(ValueError, match="max_T"):
        _ = emb(x_too_long)

    x_too_big_batch = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64)
    with pytest.raises(ValueError, match="max_B"):
        _ = emb(x_too_big_batch)


def test_sharded_high_frequency_token_spreads_gradients_across_block() -> None:
    """
    Sanity check that repeated use of a sharded token causes gradients on
    multiple shard rows (the whole point of the method).
    """
    emb = ShardedTokenEmbedding(
        num_embeddings=32,
        embedding_dim=4,
        shard_tokens={5: 8},
        row_hash_stride=13,
    )

    x = torch.full((4, 64), 5, dtype=torch.int64)  # only the sharded token
    out = emb(x)
    loss = out.sum()
    loss.backward()

    grad = emb.embedding.weight.grad
    assert grad is not None

    b, m = emb._shard_blocks[5]
    block_grad_norm = grad[b:b+m].abs().sum(dim=1)
    num_touched = int((block_grad_norm > 0).sum().item())

    # We expect multiple shard rows to be touched (usually all, but keep test robust).
    assert num_touched >= 2
