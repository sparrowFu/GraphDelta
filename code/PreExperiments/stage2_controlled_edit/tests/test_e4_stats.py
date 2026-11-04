"""e4 统计口径单元测试：平局记错、C ≡ Δv·Δt、符号翻转精确性、ridge、检索。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from e4.paired import pair_metrics
from e4.stats import RidgeProbe, scene_permutation_pair_acc, sign_flip_test


def _norm(x):
    return x / np.linalg.norm(x)


def test_pair_metrics_tie_counts_wrong():
    S = np.array([[[1.0, 1.0],   # before 图平局 → 记错
                   [0.2, 0.8]]])  # after 图选对（S[1] = [对T0, 对T1]）
    m = pair_metrics(S)
    assert m["pair_acc"] == 0.0
    assert m["i2t_acc"] == 0.5


def test_pair_metrics_perfect():
    S = np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.8, 0.3], [0.2, 0.7]]])
    m = pair_metrics(S)
    assert m["pair_acc"] == 1.0
    assert m["t2i_acc"] == 1.0


def test_C_identity_with_delta_dot_dt():
    rng = np.random.default_rng(0)
    z0, z1 = _norm(rng.standard_normal(16)), _norm(rng.standard_normal(16))
    t0, t1 = _norm(rng.standard_normal(16)), _norm(rng.standard_normal(16))
    S = np.array([[z0 @ t0, z0 @ t1], [z1 @ t0, z1 @ t1]])[None]
    C = float((S[:, 0, 0] + S[:, 1, 1] - S[:, 0, 1] - S[:, 1, 0]).mean())
    dv, dt = z1 - z0, t1 - t0
    assert abs(C - float(dv @ dt)) < 1e-10


def test_sign_flip_exact_enumeration():
    # n=4 全 +1：|null|>=1 仅全同号两种 → p = 2/16
    res = sign_flip_test(np.ones(4), n_perm=1000, seed=0)
    assert res["exact"] and res["n_perm_used"] == 16
    assert abs(res["p"] - 2 / 16) < 1e-12
    # n=12 走抽样：p ∈ [0,1] 有限
    res2 = sign_flip_test(np.random.default_rng(1).standard_normal(12),
                          n_perm=1000, seed=0)
    assert not res2["exact"] and 0.0 <= res2["p"] <= 1.0


def test_scene_permutation_null():
    rng = np.random.default_rng(2)
    mats = []
    for _ in range(3):
        z0, z1 = _norm(rng.standard_normal(8)), _norm(rng.standard_normal(8))
        t0, t1 = 0.9 * z0 + 0.1 * rng.standard_normal(8), 0.9 * z1 + 0.1 * rng.standard_normal(8)
        t0, t1 = _norm(t0), _norm(t1)
        mats.append(np.array([[t0 @ z0, t0 @ z1], [t1 @ z0, t1 @ z1]]))
    res = scene_permutation_pair_acc(mats, n_perm=1000, seed=0)
    assert res["exact"] and res["n_perm_used"] == 8
    assert res["obs"] == 1.0
    assert abs(res["p"] - 1 / 8) < 1e-12  # 只有恒等重排达到 1


def test_ridge_probe_separable():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 5))
    y = np.array(["a"] * 20 + ["b"] * 20)[rng.permutation(40)]
    X[y == "a"] += 3.0
    probe = RidgeProbe([0.01, 1.0, 100.0]).fit(X[:30], y[:30], X[30:40], y[30:40])
    assert (probe.predict(X[30:40]) == y[30:40]).all()


def test_retrieval_alpha0_and_correct():
    from e4.transfer import _retrieval_acc
    rng = np.random.default_rng(4)
    z_b = _norm(rng.standard_normal(16))
    d = rng.standard_normal(16)
    z_a = _norm(z_b + 0.5 * _norm(d))
    z_other = _norm(rng.standard_normal(16))
    r = {"before_id": "b", "after_id": "a"}
    z = {"b": z_b, "a": z_a, "o": z_other}
    pool = [("b", "before", False), ("a", "after", False), ("o", "other", False)]
    ok0, rank0, _ = _retrieval_acc(z, r, pool, d, 0.0, method_name="alpha0")
    assert not ok0 and rank0 != 0  # α=0 检索回原图，作流程参照不作基线
    ok1, rank1, _ = _retrieval_acc(z, r, pool, d, 1.0, override_direction=d,
                                   method_name="correct")
    assert ok1 and rank1 == 0
