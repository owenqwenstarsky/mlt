import math

import mlx.core as mx
import numpy as np

from mlt.config.schema import TrainConfig
from mlt.train.optim import AdamW, clip_by_global_norm, lr_at


def test_schedules():
    c = TrainConfig(steps=100, lr=1.0, warmup_steps=10, min_lr_ratio=0.1)
    assert lr_at(0, 100, c) == 0.1 and lr_at(9, 100, c) == 1.0
    assert lr_at(10, 100, c) == 1.0
    assert math.isclose(lr_at(100, 100, c), 0.1)
    assert math.isclose(lr_at(55, 100, c), 0.55)
    lin = TrainConfig(steps=100, lr=1.0, schedule="linear", min_lr_ratio=0.0)
    assert math.isclose(lr_at(50, 100, lin), 0.5)
    assert lr_at(70, 100, TrainConfig(steps=100, lr=2.0, schedule="constant")) == 2.0


def _numpy_adamw(p, grads, lr, b1, b2, eps, wd):
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    for t, g in enumerate(grads, 1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        p = p * (1 - lr * wd) - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return p


def test_adamw_matches_reference_and_skips_decay_for_vectors():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(3, 4)).astype(np.float32)
    b = rng.normal(size=(4,)).astype(np.float32)
    gw = [rng.normal(size=w.shape).astype(np.float32) for _ in range(3)]
    gb = [rng.normal(size=b.shape).astype(np.float32) for _ in range(3)]
    opt = AdamW(0.9, 0.95, 1e-8, weight_decay=0.1)
    params = {"w": mx.array(w), "norm.weight": mx.array(b)}
    opt.init(params)
    for i in range(3):
        params = opt.update(params, {"w": mx.array(gw[i]), "norm.weight": mx.array(gb[i])}, lr=0.01)
    assert np.allclose(np.array(params["w"]), _numpy_adamw(w, gw, 0.01, 0.9, 0.95, 1e-8, 0.1), atol=1e-5)
    assert np.allclose(np.array(params["norm.weight"]), _numpy_adamw(b, gb, 0.01, 0.9, 0.95, 1e-8, 0.0), atol=1e-5)


def test_bf16_params_keep_fp32_state():
    opt = AdamW(0.9, 0.95, 1e-8, 0.0)
    p = {"w": mx.ones((2, 2), dtype=mx.bfloat16)}
    opt.init(p)
    p = opt.update(p, {"w": mx.ones((2, 2), dtype=mx.bfloat16)}, lr=0.1)
    assert p["w"].dtype == mx.bfloat16 and opt.m["w"].dtype == mx.float32
    state = opt.state()
    opt2 = AdamW(0.9, 0.95, 1e-8, 0.0)
    opt2.load_state(state, step=1)
    assert mx.array_equal(opt2.v["w"], opt.v["w"]) and opt2.step == 1


def test_clip():
    g = {"a": mx.array([3.0]), "b": mx.array([4.0])}
    clipped, norm = clip_by_global_norm(g, 1.0)
    assert math.isclose(norm.item(), 5.0, rel_tol=1e-6)
    assert math.isclose(clipped["a"].item(), 0.6, rel_tol=1e-4)
    same, _ = clip_by_global_norm(g, 10.0)
    assert same["b"].item() == 4.0
