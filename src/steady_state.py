import numpy as np
from scipy.integrate import solve_ivp

from .params import (
    ka1, ka2, ka3, ka4, ka5, ka6,
    kd1, kd2, kd3, kd4, kd5, kd6,
    L, xL1, xL2, U,
)
from .equilibrium import compute_equilibrium


# ============================================================
# 定常状態ソルバー (拡張)
# ============================================================
def _ss_algebraic(A, P, PA, PA2, L0x):
    """
    定常状態での LA, LPA, LPA₂, L_free を代数的に求める。
    全 k 値が等しい (K = ka/kd) 前提。

    導出:
        LA + LPA = K(A+PA)L_free    (F_LA + F_LPA = 0 から)
        LPA₂ = K·PA₂·L_free        (F_LPA₂ = 0 から)
        L_free = L₀ / (1 + K(A+PA+PA₂))
    """
    K = ka1 / kd1
    L_free = L0x / (1.0 + K * (A + PA + PA2) + 1e-300)
    LPA    = K * L_free * (PA + K * (A + PA) * P) / (2.0 + K * P)
    LA     = K * (A + PA) * L_free - LPA
    LPA2   = K * PA2 * L_free
    return LA, LPA, LPA2, L_free


def sweep_point_ss(A0, P0, L0_tot, case='PME', Nb=500, with_PA2=True):
    """
    定常状態を Pe>>1 近似 (移流支配) で IVP として解く。
    拡張: 4 変数 (A, P, PA, PA₂) の ODE を x=0 から積分。

    Returns: (avg_S_nM, avg_captured_M)
    """
    Ae, Pe, PAe, PA2e = compute_equilibrium(A0, P0, with_PA2)

    if case == 'PMU':
        y_in = [A0, P0, 0.0, 0.0]
    else:
        y_in = [Ae, Pe, PAe, PA2e]

    _ka5_ = ka5 if with_PA2 else 0.0
    _kd5_ = kd5 if with_PA2 else 0.0
    _ka6_ = ka6 if with_PA2 else 0.0
    _kd6_ = kd6 if with_PA2 else 0.0

    def ode(x, y):
        A_, P_, PA_, PA2_ = np.maximum(y, 0.0)  # 負値防止
        L0x = L0_tot if xL1 <= x <= xL2 else 0.0
        LA_, LPA_, LPA2_, Lf_ = _ss_algebraic(A_, P_, PA_, PA2_, L0x)

        F_PA    = ka1*A_*P_      - kd1*PA_
        F_PA2   = _ka5_*PA_*A_   - _kd5_*PA2_
        r_A     = ka2*A_*Lf_     - kd2*LA_
        F1_LPA  = ka3*PA_*Lf_    - kd3*LPA_
        F2_LPA  = ka4*LA_*P_     - kd4*LPA_
        F_LPA2  = _ka6_*PA2_*Lf_ - _kd6_*LPA2_

        dA_dx   = -(F_PA + r_A + F_PA2)  / U
        dP_dx   = -(F_PA + F2_LPA)       / U
        dPA_dx  = (F_PA - F1_LPA - F_PA2) / U
        dPA2_dx = (F_PA2 - F_LPA2)       / U
        return [dA_dx, dP_dx, dPA_dx, dPA2_dx]

    x_eval = np.linspace(0, L, Nb)
    sol = solve_ivp(ode, [0, L], y_in,
                    method='RK45', t_eval=x_eval, rtol=1e-6, atol=1e-15)
    if not sol.success:
        return P0 * 1e9, 0.0

    xs   = sol.t
    A_s, P_s, PA_s, PA2_s = [np.maximum(sol.y[i], 0.0) for i in range(4)]
    L0xs = np.where((xs >= xL1) & (xs <= xL2), L0_tot, 0.0)
    _, LPA_s, LPA2_s, _ = _ss_algebraic(A_s, P_s, PA_s, PA2_s, L0xs)

    # S = P + PA + PA₂ + LPA + LPA₂
    S       = (P_s + PA_s + PA2_s + LPA_s + LPA2_s) * 1e9
    captured = LPA_s + LPA2_s  # [M]

    mask = (xs >= xL1) & (xs <= xL2)
    if mask.any():
        return float(np.mean(S[mask])), float(np.mean(captured[mask]))
    return P0 * 1e9, 0.0
