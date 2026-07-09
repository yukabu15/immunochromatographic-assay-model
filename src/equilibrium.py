import numpy as np
from scipy.optimize import brentq

from .kinetics import rate_constants


# ============================================================
# PME 平衡濃度計算 (PA₂ を含む拡張)
# ============================================================
def compute_equilibrium(A0, P0, with_PA2=True, d_nm=None, alpha=1.0):
    """
    PME 平衡濃度を計算する。膜に入る前の混合チャンバーでの平衡なので、
    リガンドLは関与せず、反応 (1) A+P⇌PA と (1') A+PA⇌PA2 のみを考える。

    PA₂ なし (元モデル):
        二次方程式で解ける: [PA]e = (b - sqrt(b²-4c)) / 2

    PA₂ あり (拡張モデル):
        保存則: [A]₀ = [A]e + [PA]e + 2[PA₂]e
                [P]₀ = [P]e + [PA]e + [PA₂]e
        平衡:   [PA]e  = K₁[A]e[P]e
                [PA₂]e = K₁'[PA]e[A]e = K₁'K₁[A]e²[P]e
        → [A]e について非線形方程式を brentq で解く

    d_nm, alpha: 粒子径依存の速度定数を使う場合に指定 (None なら旧モデル互換)。

    Returns: (Ae, Pe, PAe, PA2e)
    """
    rc = rate_constants(d_nm, alpha, with_PA2)
    K1 = rc.ka1 / rc.kd1

    if A0 <= 0 or P0 <= 0:
        return max(A0, 0.0), max(P0, 0.0), 0.0, 0.0

    if not with_PA2:
        # 元モデル: 二次方程式
        b    = A0 + P0 + 1.0 / K1
        c    = A0 * P0
        disc = max(b**2 - 4*c, 0.0)
        PAe  = (b - np.sqrt(disc)) / 2.0
        Ae   = A0 - PAe
        Pe   = P0 - PAe
        return Ae, Pe, PAe, 0.0

    # 拡張モデル: [A]e を数値的に解く
    K1p = rc.ka1p / rc.kd1p

    def f(Ae):
        denom = 1.0 + K1*Ae + K1p*K1*Ae**2
        Pe    = P0 / denom
        return Ae + K1*Ae*Pe + 2.0*K1p*K1*Ae**2*Pe - A0

    Ae = brentq(f, 0, A0, xtol=1e-18, rtol=1e-14)
    denom = 1.0 + K1*Ae + K1p*K1*Ae**2
    Pe    = P0 / denom
    PAe   = K1 * Ae * Pe
    PA2e  = K1p * PAe * Ae
    return Ae, Pe, PAe, PA2e
