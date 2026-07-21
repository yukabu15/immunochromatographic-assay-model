import numpy as np
from scipy.integrate import solve_ivp

from .params import L, xL1, xL2, U, DA
from .equilibrium import compute_equilibrium
from .kinetics import rate_constants, steric_factor, diffusion_coefficient


# ============================================================
# コアシミュレーション関数 (7 変数 PDE)
# ============================================================
def simulate(A0, P0, L0_tot, case='PMU', N=200, T=600.0,
             t_eval=None, with_PA2=True, d_nm=None, alpha=1.0, with_steric=True):
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

    d_nm, alpha: 粒子径依存キネティクスを使う場合に指定する。
        d_nm=None (デフォルト) なら旧モデル(サイズ非依存の固定速度定数,
        g=1, DP=params.DP)と完全に一致する。
    with_steric: False にすると d_nm 依存の速度定数・拡散係数は維持したまま
        立体障害係数だけ g=1 に固定する (g(r,S)の効果切り分け用)。

    Returns: (sol, x, in_cap, Ae, Pe, PAe, PA2e)
    """
    dx = L / N
    x  = np.linspace(dx, L, N)
    in_cap = (x >= xL1) & (x <= xL2)
    L0_arr = np.zeros(N)
    L0_arr[in_cap] = L0_tot

    # 粒子径依存の速度定数・立体障害係数・拡散係数
    rc     = rate_constants(d_nm, alpha, with_PA2)
    g      = steric_factor(d_nm, L0_tot, with_steric=with_steric)
    DP_eff = diffusion_coefficient(d_nm)

    # PME 平衡濃度 (流入条件と内部PDEのキネティクスを一致させる)
    Ae, Pe, PAe, PA2e = compute_equilibrium(A0, P0, with_PA2, d_nm=d_nm, alpha=alpha)

    if case == 'PMU':
        A_in, P_in, PA_in, PA2_in = A0, P0, 0.0, 0.0
    else:  # PME
        A_in, P_in, PA_in, PA2_in = Ae, Pe, PAe, PA2e

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

        # --- 自由リガンド濃度 (立体障害係数 g: d_nm=None なら g=1 で旧式と一致) ---
        Lf = L0_arr - LA - g*LPA - g*LPA2
        if d_nm is not None:
            Lf = np.maximum(Lf, 0.0)   # 大粒子径で g が大きい場合の数値的安全策

        # --- 反応速度 (8反応) ---
        F1  = rc.ka1 *A*P    - rc.kd1 *PA          # (1)  A+P   → PA
        F1p = rc.ka1p*PA*A   - rc.kd1p*PA2         # (1') A+PA  → PA2
        F2  = rc.ka2 *A*Lf   - rc.kd2 *LA          # (2)  A+L   → LA
        F3  = rc.ka3 *PA*Lf  - rc.kd3 *LPA         # (3)  PA+L  → LPA
        F3p = rc.ka3p*PA2*Lf - rc.kd3p*LPA2        # (3') PA2+L → LPA2
        F4  = rc.ka4 *LA*P   - rc.kd4 *LPA         # (4)  P+LA  → LPA
        F4p = rc.ka4p*LA*PA  - rc.kd4p*LPA2        # (4') PA+LA → LPA2
        F5  = rc.ka5 *A*LPA  - rc.kd5 *LPA2        # (5)  A+LPA → LPA2

        # --- 微分方程式 ---
        dA_dt    = DA*d2A     - U*dA_adv   - (F1 + F1p + F2 + F5)
        dP_dt    = DP_eff*d2P - U*dP_adv   - (F1 + F4)
        dPA_dt   = DP_eff*d2PA  - U*dPA_adv  + F1 - F1p - F3 - F4p
        dPA2_dt  = DP_eff*d2PA2 - U*dPA2_adv + F1p - F3p
        dLA_dt   = F2 - F4 - F4p
        dLPA_dt  = F3 + F4 - F5
        dLPA2_dt = F3p + F4p + F5

        return np.concatenate((dA_dt, dP_dt, dPA_dt, dLA_dt, dLPA_dt,
                               dPA2_dt, dLPA2_dt))

    y0 = np.zeros(7 * N)
    if t_eval is None:
        t_eval = np.linspace(0, T, 121)

    sol = solve_ivp(derivs, [0, T], y0, method='BDF',
                    t_eval=t_eval, rtol=1e-4, atol=1e-12)
    return sol, x, in_cap, Ae, Pe, PAe, PA2e
