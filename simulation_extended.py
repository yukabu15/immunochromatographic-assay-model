import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# パラメータ (Qian & Bau 2003 ベース + 拡張)
# ============================================================
# 反応速度定数 (全て同一値)
#  ①  A + P   → PA      (ka1, kd1)
#  ②  A + L   → LA      (ka2, kd2)   ※L_free を使用
#  ③  PA + L  → LPA     (ka3, kd3)
#  ④  P + LA  → LPA     (ka4, kd4)
#  ⑤  PA + A  → PA₂     (ka5, kd5)   ★新規
#  ⑥  PA₂ + L → LPA₂    (ka6, kd6)   ★新規
ka1 = ka2 = ka3 = ka4 = ka5 = ka6 = 1e6   # 1/(M·s)
kd1 = kd2 = kd3 = kd4 = kd5 = kd6 = 1e-3  # 1/s

# デフォルト濃度
A0_def = P0_def = L0_def = 10e-9  # 10 nM

# メンブレン・流れパラメータ
L    = 0.041       # メンブレン長さ [m]
xL1  = L / 2       # キャプチャーゾーン開始
xL2  = 3 * L / 4   # キャプチャーゾーン終了
U    = 2e-4        # 流速 [m/s]
DA   = 1e-10       # アナライト拡散係数 [m²/s]
DP   = 1e-12       # 粒子拡散係数 [m²/s]  (PA₂ も同値)
T_sim = 600.0      # シミュレーション時間 [s]

# 粒子径・LOD パラメータ
PARTICLE_DIAMETERS_NM = [5, 10, 40, 100]  # 粒子径 [nm]
EPSILON_TABLE = {     # モル吸光係数 ε [M⁻¹cm⁻¹]  (粒子径 nm → ε)
    5:   1.10e7,
    10:  1.01e8,
    40:  8.42e9,
    100: 1.57e11,
}
MEMBRANE_THICKNESS_CM = 0.013   # 光路長 = メンブレン厚さ 130 μm [cm]
ABSORBANCE_THRESHOLD  = 0.05    # LOD 判定閾値 (吸光度)


# ============================================================
# PME 平衡濃度計算 (PA₂ を含む拡張)
# ============================================================
def compute_equilibrium(A0, P0, with_PA2=True):
    """
    PME 平衡濃度を計算する。

    PA₂ なし (元モデル):
        二次方程式で解ける: [PA]e = (b - sqrt(b²-4c)) / 2

    PA₂ あり (拡張モデル):
        保存則: [A]₀ = [A]e + [PA]e + 2[PA₂]e
                [P]₀ = [P]e + [PA]e + [PA₂]e
        平衡:   [PA]e  = K₁[A]e[P]e
                [PA₂]e = K₅[PA]e[A]e = K₅K₁[A]e²[P]e
        → [A]e について非線形方程式を brentq で解く

    Returns: (Ae, Pe, PAe, PA2e)
    """
    K1 = ka1 / kd1
    K5 = ka5 / kd5

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
    def f(Ae):
        denom = 1.0 + K1*Ae + K5*K1*Ae**2
        Pe    = P0 / denom
        return Ae + K1*Ae*Pe + 2.0*K5*K1*Ae**2*Pe - A0

    Ae = brentq(f, 0, A0, xtol=1e-18, rtol=1e-14)
    denom = 1.0 + K1*Ae + K5*K1*Ae**2
    Pe    = P0 / denom
    PAe   = K1 * Ae * Pe
    PA2e  = K5 * PAe * Ae
    return Ae, Pe, PAe, PA2e


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


# ============================================================
# シグナル抽出ヘルパー
# ============================================================
def signal_from_sol(sol, N, t_min):
    """特定時刻 (分) での S(x) [nM] を取得。S = P + PA + PA₂ + LPA + LPA₂"""
    idx   = np.argmin(np.abs(sol.t - t_min * 60))
    P_    = sol.y[N:2*N,    idx]
    PA_   = sol.y[2*N:3*N,  idx]
    LPA_  = sol.y[4*N:5*N,  idx]
    PA2_  = sol.y[5*N:6*N,  idx]
    LPA2_ = sol.y[6*N:7*N,  idx]
    return (P_ + PA_ + PA2_ + LPA_ + LPA2_) * 1e9


def avg_S(sol, N, in_cap):
    """キャプチャーゾーン平均 S̄(t) [nM] の時系列"""
    P_    = sol.y[N:2*N,    :]
    PA_   = sol.y[2*N:3*N,  :]
    LPA_  = sol.y[4*N:5*N,  :]
    PA2_  = sol.y[5*N:6*N,  :]
    LPA2_ = sol.y[6*N:7*N,  :]
    return np.mean((P_ + PA_ + PA2_ + LPA_ + LPA2_)[in_cap, :], axis=0) * 1e9


def avg_captured(sol, N, in_cap):
    """キャプチャーゾーン平均 [LPA]+[LPA₂] [M]  (最終時刻)"""
    LPA_  = sol.y[4*N:5*N, -1]
    LPA2_ = sol.y[6*N:7*N, -1]
    return np.mean((LPA_ + LPA2_)[in_cap])


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


# ============================================================
# プロット用ヘルパー
# ============================================================
def plot_regions(ax, x_cm, y, **kw):
    """キャプチャーゾーン境界で線分を分割してプロット"""
    xL1_cm = xL1 * 100;  xL2_cm = xL2 * 100
    i1 = int(np.searchsorted(x_cm, xL1_cm, side='left'))
    i2 = int(np.searchsorted(x_cm, xL2_cm, side='right'))
    y_plot = y.astype(float).copy()
    if 0 < i1 < len(y_plot):      y_plot[i1]   = np.nan
    if 0 < i2 - 1 < len(y_plot):  y_plot[i2-1] = np.nan
    label = kw.pop('label', None)
    ax.plot(x_cm, y_plot, label=label, **kw)


def add_zone_lines(ax):
    """キャプチャーゾーン境界を破線で表示"""
    for xv in [xL1 * 100, xL2 * 100]:
        ax.axvline(xv, color='k', ls=':', lw=0.8)


# ============================================================
# メイン: シミュレーション実行 & プロット
# ============================================================
if __name__ == '__main__':
    Nbase = 200

    # ==== 1. 基本シミュレーション (デフォルト条件) ====
    print("=" * 60)
    print("拡張モデル: PA₂ / LPA₂ 含む 7 変数 PDE")
    print("=" * 60)

    print("\n[1] 基本シミュレーション (PME のみ)...")
    t_base = np.linspace(0, T_sim, 121)

    sol_pme, x, in_cap, Ae, Pe, PAe, PA2e = simulate(
        A0_def, P0_def, L0_def, 'PME', Nbase, t_eval=t_base)

    print(f"  PME 平衡濃度:")
    print(f"    [A]e  = {Ae*1e9:.3f} nM")
    print(f"    [P]e  = {Pe*1e9:.3f} nM")
    print(f"    [PA]e = {PAe*1e9:.3f} nM")
    print(f"    [PA₂]e = {PA2e*1e9:.3f} nM")
    print(f"    保存確認: [A]e + [PA]e + 2[PA₂]e = "
          f"{(Ae + PAe + 2*PA2e)*1e9:.3f} nM  (= [A]₀ = {A0_def*1e9:.1f} nM)")
    print(f"    保存確認: [P]e + [PA]e + [PA₂]e = "
          f"{(Pe + PAe + PA2e)*1e9:.3f} nM  (= [P]₀ = {P0_def*1e9:.1f} nM)")

    x_cm  = x * 100
    grays = ['0.72', '0.57', '0.43', '0.29', '0.14', '0.0']

    # ---- Fig 1: S vs x — PME ----
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    for t_min, lbl, gray in zip([4,5,6,7,8,9],
                                 ['(a)','(b)','(c)','(d)','(e)','(f)'], grays):
        plot_regions(ax1, x_cm, signal_from_sol(sol_pme, Nbase, t_min),
                     color=gray, lw=1.4, label=f't={t_min} min {lbl}')
    add_zone_lines(ax1)
    ax1.set(xlabel='x (cm)', ylabel='S (nM)', xlim=(0, 4.1), ylim=(0, 18),
            title='Signal S vs x  [PME, Extended Model]')
    ax1.legend(fontsize=8); fig1.tight_layout()
    fig1.savefig('fig1_pme.png', dpi=200)
    print("  → fig1_pme.png 保存")

    # ---- Fig 2: S̄ vs time ----
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(sol_pme.t, avg_S(sol_pme, Nbase, in_cap), 'k-',  label='PME')
    ax2.set(xlabel='Time (s)', ylabel=r'$\bar{S}$ (nM)',
            xlim=(0, 600), ylim=(-5, 20),
            title=r'Average $\bar{S}$ in capture zone [Extended Model]')
    ax2.set_xticks([0, 200, 400, 600])
    ax2.legend(); fig2.tight_layout()
    fig2.savefig('fig2_sbar.png', dpi=200)
    print("  → fig2_sbar.png 保存")

    # ==== 2. A₀ スイープ (定常状態ソルバー) ====
    print("\n[2] A₀ スイープ (定常状態, PME)...")
    A0_vals = np.logspace(-11, -5, 30)   # 0.01 nM ~ 10 μM
    A0_nM   = A0_vals * 1e9
    S0_val  = P0_def * 1e9               # ベースライン = [P]₀

    # --- PA₂ あり (拡張モデル) ---
    S_ext  = np.empty(len(A0_vals))
    cap_ext = np.empty(len(A0_vals))
    for i, A0 in enumerate(A0_vals):
        S_ext[i], cap_ext[i] = sweep_point_ss(
            A0, P0_def, L0_def, 'PME', with_PA2=True)
        if (i + 1) % 10 == 0:
            print(f"    PA₂あり: {i+1}/{len(A0_vals)} 完了")

    # --- PA₂ なし (元モデル相当) ---
    S_orig  = np.empty(len(A0_vals))
    cap_orig = np.empty(len(A0_vals))
    for i, A0 in enumerate(A0_vals):
        S_orig[i], cap_orig[i] = sweep_point_ss(
            A0, P0_def, L0_def, 'PME', with_PA2=False)
        if (i + 1) % 10 == 0:
            print(f"    PA₂なし: {i+1}/{len(A0_vals)} 完了")

    DS_ext  = S_ext  - S0_val   # ΔS [nM]
    DS_orig = S_orig - S0_val

    # ---- Fig 3: ΔS vs [A]₀ — PA₂ あり/なし比較 ----
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.loglog(A0_nM, np.maximum(DS_ext,  1e-6), 'b-',  lw=1.5,
               label='With PA₂ (Extended)')
    ax3.loglog(A0_nM, np.maximum(DS_orig, 1e-6), 'r--', lw=1.5,
               label='Without PA₂ (Original)')
    ax3.set(xlabel='$[A]_0$ (nM)',
            ylabel=r'$\Delta S = \bar{S} - S_0$ (nM)',
            title=r'Signal amplitude $\Delta S$ vs $[A]_0$: Effect of PA₂')
    ax3.legend(); ax3.grid(True, alpha=0.3); fig3.tight_layout()
    fig3.savefig('fig3_PA2_comparison.png', dpi=200)
    print("  → fig3_PA2_comparison.png 保存")

    # ---- Fig 4: ΔS̃ vs [A]₀ — PA₂ あり/なし比較 (コントラスト指標) ----
    DSt_ext  = DS_ext  / S0_val
    DSt_orig = DS_orig / S0_val

    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.loglog(A0_nM, np.maximum(np.abs(DSt_ext),  1e-8), 'b-',  lw=1.5,
                label='With PA₂ (Extended)')
    ax4.loglog(A0_nM, np.maximum(np.abs(DSt_orig), 1e-8), 'r--', lw=1.5,
                label='Without PA₂ (Original)')
    ax4.set(xlabel='$[A]_0$ (nM)',
             ylabel=r'$\widetilde{\Delta S}$',
             title=r'Contrast index $\widetilde{\Delta S}$ vs $[A]_0$: Effect of PA₂')
    ax4.legend(); ax4.grid(True, alpha=0.3); fig4.tight_layout()
    fig4.savefig('fig4_contrast.png', dpi=200)
    print("  → fig4_contrast.png 保存")

    # ==== 3. LOD 解析 (粒子径ごと) ====
    print("\n[3] LOD 解析 (粒子径: {} nm)...".format(
        ', '.join(str(d) for d in PARTICLE_DIAMETERS_NM)))

    colors_ps = {5: '#1f77b4', 10: '#ff7f0e', 40: '#2ca02c', 100: '#d62728'}
    LODs = {}

    # ---- Fig 5: Absorbance vs [A]₀ for each particle size ----
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    for d_nm in PARTICLE_DIAMETERS_NM:
        abs_vals = np.array([calc_absorbance(c, d_nm) for c in cap_ext])
        ax5.loglog(A0_nM, np.maximum(abs_vals, 1e-12), '-',
                   color=colors_ps[d_nm], lw=1.5,
                   label=f'd = {d_nm} nm  (ε = {EPSILON_TABLE[d_nm]:.2e})')
        lod = find_LOD(A0_vals, abs_vals)
        LODs[d_nm] = lod

    ax5.axhline(ABSORBANCE_THRESHOLD, color='k', ls='--', lw=1,
                label=f'Threshold = {ABSORBANCE_THRESHOLD}')
    ax5.set(xlabel='$[A]_0$ (nM)', ylabel='Absorbance',
            title='Absorbance vs $[A]_0$ for different particle sizes')
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3); fig5.tight_layout()
    fig5.savefig('fig5_absorbance.png', dpi=200)
    print("  → fig5_absorbance.png 保存")

    # ---- Fig 6: LOD vs particle size ----
    valid_d    = [d for d in PARTICLE_DIAMETERS_NM if np.isfinite(LODs[d])]
    valid_lods = [LODs[d] * 1e9 for d in valid_d]   # nM

    fig6, ax6 = plt.subplots(figsize=(6, 4))
    if valid_d:
        ax6.semilogy(valid_d, valid_lods, 'ko-', ms=8, lw=1.5)
        for d, l in zip(valid_d, valid_lods):
            ax6.annotate(f'{l:.2f} nM', (d, l),
                         textcoords='offset points', xytext=(10, 5), fontsize=9)
    # 検出不可の粒子径を赤で表示
    invalid_d = [d for d in PARTICLE_DIAMETERS_NM if not np.isfinite(LODs[d])]
    for d in invalid_d:
        ax6.axvline(d, color='r', ls=':', alpha=0.5)
        ax6.annotate(f'd={d} nm\n(Not Detected)', (d, ax6.get_ylim()[1]*0.5),
                     fontsize=8, color='r', ha='center')

    ax6.set(xlabel='Particle diameter (nm)', ylabel='LOD (nM)',
            title='LOD vs Particle Size  (Extended Model, PME)')
    ax6.grid(True, alpha=0.3); fig6.tight_layout()
    fig6.savefig('fig6_LOD.png', dpi=200)
    print("  → fig6_LOD.png 保存")

    # ==== 4. 結果サマリー ====
    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)

    print("\n--- PME 平衡濃度 (拡張モデル) ---")
    print(f"  [A]e   = {Ae*1e9:.3f} nM")
    print(f"  [P]e   = {Pe*1e9:.3f} nM")
    print(f"  [PA]e  = {PAe*1e9:.3f} nM")
    print(f"  [PA₂]e = {PA2e*1e9:.3f} nM")

    print("\n--- PA₂ 効果の比較 ([A]₀=10 nM, 定常状態) ---")
    idx10 = np.argmin(np.abs(A0_vals - 10e-9))
    print(f"  ΔS (PA₂あり)  = {DS_ext[idx10]:.4f} nM")
    print(f"  ΔS (PA₂なし)  = {DS_orig[idx10]:.4f} nM")
    ratio = DS_ext[idx10] / DS_orig[idx10] if DS_orig[idx10] != 0 else float('inf')
    print(f"  シグナル増加率 = {ratio:.2f} 倍")

    print("\n--- LOD (粒子径別) ---")
    print(f"  {'粒子径 (nm)':>12}  {'ε (M⁻¹cm⁻¹)':>14}  {'LOD (nM)':>10}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*10}")
    for d_nm in PARTICLE_DIAMETERS_NM:
        eps_str = f"{EPSILON_TABLE[d_nm]:.2e}"
        if np.isfinite(LODs[d_nm]):
            lod_str = f"{LODs[d_nm]*1e9:.3f}"
        else:
            lod_str = "N/A"
        print(f"  {d_nm:>12}  {eps_str:>14}  {lod_str:>10}")

    print(f"\n  判定条件: 吸光度 ≥ {ABSORBANCE_THRESHOLD}")
    print(f"  光路長:   {MEMBRANE_THICKNESS_CM*10:.0f} mm "
          f"({MEMBRANE_THICKNESS_CM*1e4:.0f} μm)")

    print("\n全グラフ出力完了 (fig1_pme.png ~ fig6_LOD.png)")
