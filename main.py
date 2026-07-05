import os

import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

IMG_DIR = 'img'

from src.params import (
    A0_def, P0_def, L0_def, T_sim,
    PARTICLE_DIAMETERS_NM, EPSILON_TABLE, ABSORBANCE_THRESHOLD,
    MEMBRANE_THICKNESS_CM,
)
from src.simulation import simulate
from src.signals import signal_from_sol, avg_S
from src.lod import calc_absorbance, find_LOD
from src.steady_state import sweep_point_ss
from src.plotting import plot_regions, add_zone_lines


# ============================================================
# メイン: シミュレーション実行 & プロット
# ============================================================
if __name__ == '__main__':
    os.makedirs(IMG_DIR, exist_ok=True)
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
    fig1.savefig(os.path.join(IMG_DIR, 'fig1_pme.png'), dpi=200)
    print("  → img/fig1_pme.png 保存")

    # ---- Fig 2: S̄ vs time ----
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(sol_pme.t, avg_S(sol_pme, Nbase, in_cap), 'k-',  label='PME')
    ax2.set(xlabel='Time (s)', ylabel=r'$\bar{S}$ (nM)',
            xlim=(0, 600), ylim=(-5, 20),
            title=r'Average $\bar{S}$ in capture zone [Extended Model]')
    ax2.set_xticks([0, 200, 400, 600])
    ax2.legend(); fig2.tight_layout()
    fig2.savefig(os.path.join(IMG_DIR, 'fig2_sbar.png'), dpi=200)
    print("  → img/fig2_sbar.png 保存")

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
    fig3.savefig(os.path.join(IMG_DIR, 'fig3_PA2_comparison.png'), dpi=200)
    print("  → img/fig3_PA2_comparison.png 保存")

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
    fig4.savefig(os.path.join(IMG_DIR, 'fig4_contrast.png'), dpi=200)
    print("  → img/fig4_contrast.png 保存")

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
    fig5.savefig(os.path.join(IMG_DIR, 'fig5_absorbance.png'), dpi=200)
    print("  → img/fig5_absorbance.png 保存")

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
    fig6.savefig(os.path.join(IMG_DIR, 'fig6_LOD.png'), dpi=200)
    print("  → img/fig6_LOD.png 保存")

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

    print("\n全グラフ出力完了 (img/fig1_pme.png ~ img/fig6_LOD.png)")
