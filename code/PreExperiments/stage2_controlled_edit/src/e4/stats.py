"""E4 统计封装：符号翻转置换、场景内匹配置换、ridge 探针（工作方案第 5–7 节）。

默认 1,000 次置换、α=0.05（configs/e4_protocol.yaml）；允许的唯一重排少于
置换次数时全枚举并报告数量，不把重复抽取当新重排（第 7 节）。
"""
import itertools

import numpy as np

from src.metrics.bootstrap import cluster_mean_ci


def sign_flip_test(values, n_perm: int, seed: int) -> dict:
    """场景级符号翻转检验 H0：C（或余量）关于 0 对称。双侧 p。

    2^n <= n_perm 时全枚举（含观测自身），否则有放回抽 n_perm 次。
    """
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    obs = v.mean()
    if n == 0:
        return {"p": float("nan"), "n_perm_used": 0, "exact": False, "obs_mean": float("nan")}
    exact = (2 ** n) <= n_perm
    if exact:
        flips = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        null = flips @ v / n
        n_used = len(null)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        null = (signs * v).mean(axis=1)
        n_used = n_perm
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    return {"p": p, "n_perm_used": int(n_used), "exact": bool(exact),
            "obs_mean": float(obs)}


def scene_pair_acc(S) -> float:
    """单场景 Pair Accuracy（E4-A 主口径，平局记错）。

    先在同一模板内判断两张图是否均选对，再跨模板平均（七模板不增加独立
    样本数）。paired.pair_metrics 与 scene_permutation_pair_acc 共用本计分，
    保证报告的观测值与置换值是同一个指标（审查问题 2：不得先平均相似度
    再判断）。
    """
    S = np.asarray(S, dtype=np.float64)
    c00 = S[:, 0, 0] > S[:, 0, 1]   # before 图在 (T0,T1) 中选对
    c11 = S[:, 1, 1] > S[:, 1, 0]   # after 图选对
    return float((c00 & c11).mean())


def scene_permutation_pair_acc(S_by_scene: list, n_perm: int, seed: int) -> dict:
    """E4-A 参考分布（第 5 节）：场景内随机交换两文本标签后的 Pair Accuracy。

    S_by_scene：每场景的 (T,2,2) 模板相似度矩阵（S[t,i,j]=sim(img_i,text_j)）。
    交换文本标签 = 该场景全部模板的文本列一起交换（模板不分开换）；
    每个置换值与观测值同用 scene_pair_acc 计算（先模板内判断、再跨模板
    平均）。2^n <= n_perm 时全枚举（含观测自身）。返回零分布均值 / 标准差
    与 p（单侧上尾：真实配对应高于随机配对）。
    """

    def _pair_acc(mats, swaps):
        acc = []
        for m, s in zip(mats, swaps):
            mm = m if s == 0 else m[:, :, [1, 0]]
            acc.append(scene_pair_acc(mm))
        return float(np.mean(acc))

    mats = [np.asarray(m, dtype=np.float64) for m in S_by_scene]
    n = len(mats)
    if n == 0:
        return {"null_mean": float("nan"), "null_std": float("nan"),
                "p": float("nan"), "n_perm_used": 0, "exact": False}
    obs = _pair_acc(mats, [0] * n)
    exact = (2 ** n) <= n_perm
    if exact:
        swaps_all = list(itertools.product([0, 1], repeat=n))
        null = np.array([_pair_acc(mats, sw) for sw in swaps_all])
        n_used = len(swaps_all)
    else:
        rng = np.random.default_rng(seed)
        null = np.empty(n_perm)
        for k in range(n_perm):
            null[k] = _pair_acc(mats, rng.integers(0, 2, size=n))
        n_used = n_perm
    return {"null_mean": float(null.mean()), "null_std": float(null.std()),
            "p": float((null >= obs - 1e-12).mean()),
            "n_perm_used": int(n_used), "exact": bool(exact), "obs": float(obs)}


def cluster_ci(values, clusters, n_boot: int, ci: float, seed: int = 0):
    """按基础场景聚类 bootstrap 的均值区间（薄封装，口径统一）。"""
    return cluster_mean_ci(list(values), list(clusters), n_boot=n_boot, ci=ci, seed=seed)


class RidgeProbe:
    """闭式解 ridge 多类读出（E4-B 线性探针）。

    标准化统计（均值/方差）只用训练折；λ 在验证折选，选准确率最高、并列取
    最小 λ（确定性）。输入为冻结特征，不训练骨干（第 5 节）。
    """

    def __init__(self, lambdas):
        self.lambdas = list(lambdas)
        self.mu = self.sd = self.W = self.lam = self.classes = None

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-12] = 1.0
        Z = (X - self.mu) / self.sd
        self.classes = np.array(sorted(set(y)))
        idx = {c: i for i, c in enumerate(self.classes)}
        Y = np.zeros((len(y), len(self.classes)))
        for r, c in enumerate(y):
            Y[r, idx[c]] = 1.0
        D = Z.shape[1]
        if X_val is None or y_val is None:
            self.lam = self.lambdas[0]
        else:
            Zv = (np.asarray(X_val, dtype=np.float64) - self.mu) / self.sd
            best, best_acc = self.lambdas[0], -1.0
            for lam in self.lambdas:
                W = np.linalg.solve(Z.T @ Z + lam * np.eye(D), Z.T @ Y)
                acc = float((self.classes[np.argmax(Zv @ W, axis=1)]
                             == np.asarray(y_val)).mean())
                if acc > best_acc + 1e-12:
                    best, best_acc = lam, acc
            self.lam = best
        self.W = np.linalg.solve(Z.T @ Z + self.lam * np.eye(D), Z.T @ Y)
        return self

    def predict(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        return self.classes[np.argmax(Z @ self.W, axis=1)]


def cosine_rows(A, B) -> np.ndarray:
    """对齐行的 cosine；近零行返回 0（不除以 0）。"""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    num = (A * B).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    out = np.zeros_like(num)
    good = den > 1e-12
    out[good] = num[good] / den[good]
    return out


def cosine_matrix(A, B) -> np.ndarray:
    """A 每行与 B 每行的 cosine 全组合，形状 (len(A), len(B))；近零行记 0。

    行数不同的两组向量（如留出 Δv vs 全部随机方向）不能用 cosine_rows
    逐行配对——那要求行数可广播，行数恰好相等时还会算成逐对错配。
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    Bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return An @ Bn.T
