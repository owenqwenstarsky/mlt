"""The central correctness property: splitting a global batch unevenly across
simulated ranks, with token-weighted reduction, gives the same gradients and the
same optimizer step as one machine processing the whole batch."""

import mlx.core as mx
import numpy as np
import pytest

from mlt.data.batching import make_microbatches
from mlt.data.sampler import rank_slice, split_counts
from mlt.dist.reduce import pack, sum_flat, unpack
from mlt.dist.world import single
from mlt.train.step import StepFunctions, local_gradients, synchronise

from .conftest import tiny_gpt


def _chat_items(n=8, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n):
        L = int(rng.integers(6, 30))
        toks = rng.integers(0, 97, size=L)
        mask = (rng.random(L) < 0.5).astype(np.uint8)
        mask[0] = 0
        mask[-1] = 1
        items.append((toks, mask))
    return items


def _text_items(n=8, L=16, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.integers(0, 97, size=L + 1), None) for _ in range(n)]


def _single_machine(model, items, seq_len):
    fns = StepFunctions(model, compile=False)
    g, ls, n = local_gradients(fns, make_microbatches(items, len(items), seq_len))
    return {k: v / n for k, v in g.items()}, (ls / n).item()


def _simulated(model, items, seq_len, shares, mbs, comm_dtype=None):
    fns = StepFunctions(model, compile=False)
    counts = split_counts(len(items), shares)
    per_rank = []
    for r in range(len(shares)):
        sl = rank_slice(counts, r)
        per_rank.append(local_gradients(fns, make_microbatches(items[sl], mbs[r], seq_len)))
    # Simulated all_sum over packed buckets, exactly what the ring backend does.
    packed = [pack(g, comm_dtype, bucket_bytes=4096) for g, _, _ in per_rank]
    slots = packed[0][1]
    summed = [sum(p[0][i].astype(mx.float32) for p in packed) for i in range(len(packed[0][0]))]
    grads = unpack(summed, slots, mx.float32)
    loss = sum(ls for _, ls, _ in per_rank)
    n = sum(c for _, _, c in per_rank)
    return {k: v / n for k, v in grads.items()}, (loss / n).item()


@pytest.mark.parametrize("kind", ["text", "chat"])
@pytest.mark.parametrize("shares,mbs", [([0.5, 0.5], [4, 4]), ([0.7, 0.3], [2, 1]), ([0.2, 0.5, 0.3], [1, 3, 2])])
def test_uneven_split_equals_single_machine(kind, shares, mbs):
    model = tiny_gpt()
    items = _text_items() if kind == "text" else _chat_items()
    seq_len = 32
    ref, ref_loss = _single_machine(model, items, seq_len)
    got, loss = _simulated(model, items, seq_len, shares, mbs)
    assert abs(loss - ref_loss) < 1e-4
    for k in ref:
        assert mx.allclose(ref[k], got[k], atol=1e-5, rtol=1e-4), k


def test_bf16_communication_is_close():
    model = tiny_gpt()
    items = _text_items()
    ref, _ = _single_machine(model, items, 32)
    got, _ = _simulated(model, items, 32, [0.5, 0.5], [2, 2], comm_dtype=mx.bfloat16)
    for k in ref:
        assert mx.allclose(ref[k], got[k], atol=1e-3, rtol=2e-2), k


def test_accumulation_equals_one_big_batch():
    model = tiny_gpt()
    items = _chat_items()
    fns = StepFunctions(model, compile=False)
    g1, l1, n1 = local_gradients(fns, make_microbatches(items, 8, 32))
    g2, l2, n2 = local_gradients(fns, make_microbatches(items, 1, 32))
    assert n1.item() == n2.item()
    assert abs(l1.item() - l2.item()) < 1e-3
    for k in g1:
        assert mx.allclose(g1[k].astype(mx.float32), g2[k], atol=1e-4, rtol=1e-4)


def test_synchronise_single_rank_normalises_by_tokens():
    g = {"w": mx.array([2.0, 4.0])}
    out, loss, n = synchronise(g, mx.array(6.0), mx.array(2.0), single(), mx.float32)
    assert out["w"].tolist() == [1.0, 2.0] and loss.item() == 6.0 and n.item() == 2.0


def test_pack_unpack_roundtrip_and_buckets():
    flat = {"a": mx.arange(10, dtype=mx.float32).reshape(2, 5), "b": mx.ones((3,), dtype=mx.bfloat16), "c": mx.zeros((1000,))}
    buckets, slots = pack(flat, None, bucket_bytes=64)
    assert len(buckets) >= 3  # dtype change and size limit both split buckets
    back = unpack(buckets, slots)
    for k in flat:
        assert back[k].dtype == flat[k].dtype and mx.array_equal(back[k], flat[k])


def test_sum_flat_with_custom_reduce():
    flat = {"x": mx.array([1.0, 2.0]), "y": mx.array([[3.0]])}
    out = sum_flat(flat, single(), comm_dtype=mx.float32, reduce_fn=lambda b: b * 3)
    assert out["x"].tolist() == [3.0, 6.0] and out["y"].tolist() == [[9.0]]
