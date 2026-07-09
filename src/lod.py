import numpy as np

from .params import EPSILON_TABLE, MEMBRANE_THICKNESS_CM, ABSORBANCE_THRESHOLD


# ============================================================
# LOD 計算関数
# ============================================================
def calc_absorbance(captured_conc_M, d_nm):
    """
    吸光度 = ε(d) × ℓ × [LPA+LPA₂]
    d_nm: 粒子径 [nm]
    captured_conc_M: [LPA]+[LPA₂] [M]
    """
    epsilon = EPSILON_TABLE[d_nm]
    return epsilon * MEMBRANE_THICKNESS_CM * captured_conc_M


def find_LOD(A0_vals, absorbances, threshold=ABSORBANCE_THRESHOLD):
    """吸光度が閾値を超える最小 [A]₀ を対数空間の線形補間で求める"""
    if absorbances[0] >= threshold:
        return A0_vals[0]  # 最低濃度ですでに閾値を超えている場合

    for i in range(len(absorbances) - 1):
        if absorbances[i] < threshold <= absorbances[i + 1]:
            log_lo = np.log10(A0_vals[i])
            log_hi = np.log10(A0_vals[i + 1])
            frac   = ((threshold - absorbances[i])
                      / (absorbances[i + 1] - absorbances[i]))
            return 10 ** (log_lo + frac * (log_hi - log_lo))

    return float('inf')                 # 閾値到達せず


def find_time_to_LOD(t_vals, absorbances, threshold=ABSORBANCE_THRESHOLD):
    """吸光度が閾値を超える最初の時刻を線形補間で求める。到達しなければ inf。"""
    if absorbances[0] >= threshold:
        return t_vals[0]

    for i in range(len(absorbances) - 1):
        if absorbances[i] < threshold <= absorbances[i + 1]:
            frac = (threshold - absorbances[i]) / (absorbances[i + 1] - absorbances[i])
            return t_vals[i] + frac * (t_vals[i + 1] - t_vals[i])

    return float('inf')                 # 閾値到達せず
