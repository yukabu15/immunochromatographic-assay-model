import numpy as np
from scipy.integrate import solve_ivp

from .params import (
    ka1, ka2, ka3, ka4, ka5, ka6,
    kd1, kd2, kd3, kd4, kd5, kd6,
    L, xL1, xL2, U, DA, DP,
)
from .equilibrium import compute_equilibrium


# ============================================================
# コアシミュレーション関数 (7 変数 PDE)
# ============================================================
def simulate(A0, P0, L0_tot, case='PMU', N=200, T=600.0,
             t_eval=None, with_PA2=True):
    """
    拡張 PDE を BDF 法で数値積分。

    状態変数 (各 N 格子点):
        y[0:N]      = [A]       アナライト
        y[N:2N]     = [P]       自由粒子
        y[2N:3N]    = [PA]      粒子-アナライト複合体
        y[3N:4N]    = [LA]      リガンド-アナライト複合体
        y[4N:5N]    = [LPA]     リガンド-粒子-アナライト複合体
        y[5N:6N]    = [PA₂]     粒子-二重アナライト複合体  ★新規
        y[6N:7N]    = [LPA₂]    リガンド-粒子-二重アナライト ★新規

    Returns: (sol, x, in_cap, Ae, Pe, PAe, PA2e)
    """
    dx = L / N
    x  = np.linspace(dx, L, N)
    in_cap = (x >= xL1) & (x <= xL2)
    L0_arr = np.zeros(N)
    L0_arr[in_cap] = L0_tot

    # PME 平衡濃度
    Ae, Pe, PAe, PA2e = compute_equilibrium(A0, P0, with_PA2)

    if case == 'PMU':
        A_in, P_in, PA_in, PA2_in = A0, P0, 0.0, 0.0
    else:  # PME
        A_in, P_in, PA_in, PA2_in = Ae, Pe, PAe, PA2e

    # PA₂ 反応を無効化するためのフラグ
    _ka5 = ka5 if with_PA2 else 0.0
    _kd5 = kd5 if with_PA2 else 0.0
    _ka6 = ka6 if with_PA2 else 0.0
    _kd6 = kd6 if with_PA2 else 0.0

    def derivs(t, y):
        A    = y[0:N]
        P    = y[N:2*N]
        PA   = y[2*N:3*N]
        LA   = y[3*N:4*N]
        LPA  = y[4*N:5*N]
        PA2  = y[5*N:6*N]
        LPA2 = y[6*N:7*N]

        # --- 移流項 (風上差分) ---
        def adv(f, f0):
            return np.concatenate(([f[0] - f0], f[1:] - f[:-1])) / dx
        dA_adv   = adv(A,   A_in)
        dP_adv   = adv(P,   P_in)
        dPA_adv  = adv(PA,  PA_in)
        dPA2_adv = adv(PA2, PA2_in)

        # --- 拡散項 (中心差分, Neumann 出口) ---
        def diff2(f, f0):
            d = np.empty(N)
            d[0]    = (f[1]  - 2*f[0]  + f0)       / dx**2
            d[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
            d[-1]   = (f[-2] - f[-1])               / dx**2
            return d
        d2A   = diff2(A,   A_in)
        d2P   = diff2(P,   P_in)
        d2PA  = diff2(PA,  PA_in)
        d2PA2 = diff2(PA2, PA2_in)

        # --- 自由リガンド濃度 (LPA₂ を含む) ---
        L_free = L0_arr - LA - LPA - LPA2        # ★ LPA₂ 追加

        # --- 反応速度 ---
        F_PA    = ka1*A*P       - kd1*PA          # ① A+P → PA
        F_LA    = (ka2*A*L_free - kd2*LA           # ② A+L → LA (net)
                   - ka4*LA*P   + kd4*LPA)
        F1_LPA  = ka3*PA*L_free - kd3*LPA          # ③ PA+L → LPA
        F2_LPA  = ka4*LA*P      - kd4*LPA          # ④ P+LA → LPA
        F_LPA   = F1_LPA + F2_LPA
        F_PA2   = _ka5*PA*A     - _kd5*PA2         # ⑤ PA+A → PA₂ ★
        F_LPA2  = _ka6*PA2*L_free - _kd6*LPA2      # ⑥ PA₂+L → LPA₂ ★

        # A の L 直接消費 (dA/dt に使用)
        r_A = ka2*A*L_free - kd2*LA

        # --- 微分方程式 ---
        dA_dt    = DA*d2A   - U*dA_adv   - F_PA - r_A - F_PA2    # ★ F_PA2 追加
        dP_dt    = DP*d2P   - U*dP_adv   - F_PA - F2_LPA
        dPA_dt   = DP*d2PA  - U*dPA_adv  + F_PA - F1_LPA - F_PA2  # ★ F_PA2 追加
        dLA_dt   = F_LA
        dLPA_dt  = F_LPA
        dPA2_dt  = DP*d2PA2 - U*dPA2_adv + F_PA2 - F_LPA2         # ★ 新規
        dLPA2_dt = F_LPA2                                          # ★ 新規

        return np.concatenate((dA_dt, dP_dt, dPA_dt, dLA_dt, dLPA_dt,
                               dPA2_dt, dLPA2_dt))

    y0 = np.zeros(7 * N)
    if t_eval is None:
        t_eval = np.linspace(0, T, 121)

    sol = solve_ivp(derivs, [0, T], y0, method='BDF',
                    t_eval=t_eval, rtol=1e-4, atol=1e-12)
    return sol, x, in_cap, Ae, Pe, PAe, PA2e
