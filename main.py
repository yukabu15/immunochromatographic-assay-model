import os

import numpy as np
import matplotlib.pyplot as plt

IMG_DIR = 'img'

from src.params import (
    A0_def, P0_def, L0_def, T_sim,
    PARTICLE_DIAMETERS_NM, ABSORBANCE_THRESHOLD, A0_VALUES,
)
from src.simulation import simulate
from src.signals import LPA_LPA2_series
from src.lod import calc_absorbance, find_time_to_LOD
from src.kinetics import antibodies_per_particle, steric_factor, diffusion_coefficient


# ============================================================
# メイン: 粒子径依存キネティクス (n, α, 立体障害, 拡散係数)
# ============================================================
if __name__ == '__main__':
    os.makedirs(IMG_DIR, exist_ok=True)

    print("=" * 60)
    print("粒子径依存キネティクス: 吸光度の時間発展とLOD")
    print("=" * 60)

    ALPHAS = [0.0, 0.5, 1.0]
    PA2_MODES = [True, False]
    Nbase_sweep = 100
    t_sweep = np.linspace(0, T_sim, 121)
    alpha_styles = {0.0: (':', 'C0'), 0.5: ('--', 'C1'), 1.0: ('-', 'C2')}

    cmap = plt.cm.viridis
    colors_ps = {d: cmap(i / (len(PARTICLE_DIAMETERS_NM) - 1))
                 for i, d in enumerate(PARTICLE_DIAMETERS_NM)}

    print("\n  粒子径ごとの n(d), g(d,L0), DP(d):")
    print(f"  {'d(nm)':>6}  {'n':>8}  {'g(r,S)':>10}  {'DP(m2/s)':>10}")
    for d_nm in PARTICLE_DIAMETERS_NM:
        n_d  = antibodies_per_particle(d_nm)
        g_d  = steric_factor(d_nm, L0_def)
        dp_d = diffusion_coefficient(d_nm)
        print(f"  {d_nm:>6}  {n_d:>8.3f}  {g_d:>10.3g}  {dp_d:>10.3g}")

    results = {}
    print("\n  スイープ実行 (PA2 2通り × α 3通り × 粒子径 10通り = 60回)...")
    for with_PA2 in PA2_MODES:
        for alpha in ALPHAS:
            for d_nm in PARTICLE_DIAMETERS_NM:
                sol5, x5, in_cap5, *_ = simulate(
                    A0_def, P0_def, L0_def, 'PME', Nbase_sweep,
                    t_eval=t_sweep, with_PA2=with_PA2, d_nm=d_nm, alpha=alpha)

                lpa_series, lpa2_series = LPA_LPA2_series(sol5, Nbase_sweep, in_cap5)
                cap_series = lpa_series + lpa2_series
                abs_series = calc_absorbance(cap_series, d_nm)
                abs_final  = abs_series[-1]

                rel_change = abs(abs_series[-1] - abs_series[-2]) / max(abs_series[-1], 1e-30)
                if rel_change > 0.01:
                    print(f"    警告: PA2={with_PA2}, α={alpha}, d={d_nm}nm は "
                          f"T_sim={T_sim:.0f}sで未収束の可能性 "
                          f"(最終ステップ相対変化={rel_change:.1%})")

                t_lod = find_time_to_LOD(t_sweep, abs_series, ABSORBANCE_THRESHOLD)
                results[(with_PA2, alpha, d_nm)] = dict(
                    t=sol5.t, abs_series=abs_series, abs_final=abs_final, t_lod=t_lod,
                    lpa_nM=lpa_series * 1e9, lpa2_nM=lpa2_series * 1e9)

                t_lod_str = f"{t_lod:.1f}s" if np.isfinite(t_lod) else "N/A"
                print(f"    [PA2={str(with_PA2):5}, α={alpha:.1f}, d={d_nm:>3}nm] "
                      f"A_final={abs_final:.4g}, t_LOD={t_lod_str}")

    # ---- Fig 7: 粒子径 vs 吸光度 (T_sim時点, 定常の近似) ----
    fig7, axes7 = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, with_PA2 in zip(axes7, PA2_MODES):
        for alpha in ALPHAS:
            vals = [results[(with_PA2, alpha, d)]['abs_final'] for d in PARTICLE_DIAMETERS_NM]
            ls, c = alpha_styles[alpha]
            ax.plot(PARTICLE_DIAMETERS_NM, np.maximum(vals, 1e-12), ls, color=c,
                    marker='o', ms=4, label=f'α={alpha}')
        ax.axhline(ABSORBANCE_THRESHOLD, color='k', ls='--', lw=1, alpha=0.6)
        ax.set_yscale('log')
        ax.set(xlabel='Particle diameter (nm)',
               title=("With PA2" if with_PA2 else "Without PA2"))
        ax.grid(True, alpha=0.3)
    axes7[0].set_ylabel(f'Absorbance (t={T_sim:.0f}s)')
    axes7[0].legend(fontsize=8)
    fig7.suptitle('Absorbance (steady) vs particle diameter')
    fig7.tight_layout()
    fig7.savefig(os.path.join(IMG_DIR, 'fig7_absorbance_vs_diameter.png'), dpi=200)
    print("  → img/fig7_absorbance_vs_diameter.png 保存")

    # ---- Fig 8: 時間 vs 吸光度 (粒径ごと, α ごとに1枚) ----
    for alpha in ALPHAS:
        fig8, axes8 = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, with_PA2 in zip(axes8, PA2_MODES):
            for d_nm in PARTICLE_DIAMETERS_NM:
                r = results[(with_PA2, alpha, d_nm)]
                ax.plot(r['t'], np.maximum(r['abs_series'], 1e-12), '-',
                        color=colors_ps[d_nm], lw=1.3, label=f'{d_nm}nm')
            ax.axhline(ABSORBANCE_THRESHOLD, color='k', ls='--', lw=1, alpha=0.6)
            ax.set_yscale('log')
            ax.set(xlabel='Time (s)', title=("With PA2" if with_PA2 else "Without PA2"))
            ax.grid(True, alpha=0.3)
        axes8[0].set_ylabel('Absorbance')
        axes8[0].legend(fontsize=7, ncol=2, title='diameter')
        alpha_tag = str(alpha).replace('.', 'p')
        fig8.suptitle(f'Absorbance vs time (α={alpha})')
        fig8.tight_layout()
        fig8.savefig(os.path.join(IMG_DIR, f'fig8_timeseries_alpha{alpha_tag}.png'), dpi=200)
        print(f"  → img/fig8_timeseries_alpha{alpha_tag}.png 保存")

    # ---- Fig 9: 粒子径 vs LOD到達時間 ----
    fig9, axes9 = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, with_PA2 in zip(axes9, PA2_MODES):
        for alpha in ALPHAS:
            ls, c = alpha_styles[alpha]
            finite_d, finite_t, inf_d = [], [], []
            for d_nm in PARTICLE_DIAMETERS_NM:
                t_lod = results[(with_PA2, alpha, d_nm)]['t_lod']
                if np.isfinite(t_lod):
                    finite_d.append(d_nm); finite_t.append(t_lod)
                else:
                    inf_d.append(d_nm)
            ax.plot(finite_d, finite_t, ls, color=c, marker='o', ms=4, label=f'α={alpha}')
            for d_nm in inf_d:
                ax.plot(d_nm, T_sim, 'x', color=c, ms=8, mew=2)
        ax.set(xlabel='Particle diameter (nm)',
               title=("With PA2" if with_PA2 else "Without PA2"))
        ax.grid(True, alpha=0.3)
    axes9[0].set_ylabel('Time to reach LOD (s)')
    axes9[0].legend(fontsize=8)
    fig9.suptitle('Time to LOD vs particle diameter  (x = not detected within T_sim)')
    fig9.tight_layout()
    fig9.savefig(os.path.join(IMG_DIR, 'fig9_time_to_LOD_vs_diameter.png'), dpi=200)
    print("  → img/fig9_time_to_LOD_vs_diameter.png 保存")

    print("\n--- 粒子径依存モデル サマリー ---")
    for with_PA2 in PA2_MODES:
        for alpha in ALPHAS:
            finite = [(d, results[(with_PA2, alpha, d)]['t_lod'])
                      for d in PARTICLE_DIAMETERS_NM
                      if np.isfinite(results[(with_PA2, alpha, d)]['t_lod'])]
            if finite:
                d_min, t_min = min(finite, key=lambda x: x[1])
                d_max, t_max = max(finite, key=lambda x: x[1])
                print(f"  PA2={str(with_PA2):5}, α={alpha:.1f}: "
                      f"最短LOD到達 d={d_min}nm ({t_min:.1f}s), "
                      f"最長LOD到達 d={d_max}nm ({t_max:.1f}s)")
            else:
                print(f"  PA2={str(with_PA2):5}, α={alpha:.1f}: T_sim内でLOD到達なし")

    # ==== 6. α=0.5, PA2ありに固定した追加解析 ====
    print("\n" + "=" * 60)
    print("[6] α=0.5, PA2あり固定の追加解析")
    print("=" * 60)

    ALPHA_FIXED = 0.5

    # ---- Fig 11: 粒径 vs LOD到達時間 (抗原量=A0_def, α=0.5, PA2あり) ----
    # 既存のスイープ結果 (results) から α=0.5, PA2=True のスライスを再利用する
    fig11, ax11 = plt.subplots(figsize=(6, 4.5))
    finite_d, finite_t, inf_d = [], [], []
    for d_nm in PARTICLE_DIAMETERS_NM:
        t_lod = results[(True, ALPHA_FIXED, d_nm)]['t_lod']
        if np.isfinite(t_lod):
            finite_d.append(d_nm); finite_t.append(t_lod)
        else:
            inf_d.append(d_nm)
    ax11.plot(finite_d, finite_t, '-o', color='C1', ms=5)
    for d_nm in inf_d:
        ax11.plot(d_nm, T_sim, 'x', color='C1', ms=8, mew=2)
    ax11.set(xlabel='Particle diameter (nm)', ylabel='Time to reach LOD (s)',
             title=f'Time to LOD vs diameter  (α={ALPHA_FIXED}, With PA2, [A]₀={A0_def*1e9:.0f}nM)')
    ax11.grid(True, alpha=0.3)
    fig11.tight_layout()
    fig11.savefig(os.path.join(IMG_DIR, 'fig11_time_to_LOD_vs_diameter_alpha0p5.png'), dpi=200)
    print("  → img/fig11_time_to_LOD_vs_diameter_alpha0p5.png 保存")

    # ---- 抗原量 × 粒径 のスイープ (α=0.5, PA2あり) ----
    results_A0 = {}
    n_calls = len(PARTICLE_DIAMETERS_NM) * len(A0_VALUES)
    print(f"\n  [A]₀ × 粒径 スイープ実行 (粒径{len(PARTICLE_DIAMETERS_NM)}通り × "
          f"[A]₀ {len(A0_VALUES)}通り = {n_calls}回)...")
    for d_nm in PARTICLE_DIAMETERS_NM:
        for A0 in A0_VALUES:
            sol6, x6, in_cap6, *_ = simulate(
                A0, P0_def, L0_def, 'PME', Nbase_sweep,
                t_eval=t_sweep, with_PA2=True, d_nm=d_nm, alpha=ALPHA_FIXED)
            lpa6, lpa26 = LPA_LPA2_series(sol6, Nbase_sweep, in_cap6)
            abs_series6 = calc_absorbance(lpa6 + lpa26, d_nm)
            t_lod6 = find_time_to_LOD(t_sweep, abs_series6, ABSORBANCE_THRESHOLD)
            results_A0[(d_nm, A0)] = dict(abs_final=abs_series6[-1], t_lod=t_lod6)
        print(f"    d={d_nm}nm 完了")

    # ---- Fig 12: 抗原量 vs LOD到達時間 (粒径ごと) ----
    # T_sim内で一度もLODに到達しない粒径は、点(x)を大量に並べると見づらいため、
    # 折れ線を描かず脚注にまとめて明記する
    fig12, ax12 = plt.subplots(figsize=(7, 5))
    never_detected = []
    for d_nm in PARTICLE_DIAMETERS_NM:
        finite_A0, finite_t = [], []
        for A0 in A0_VALUES:
            t_lod = results_A0[(d_nm, A0)]['t_lod']
            if np.isfinite(t_lod):
                finite_A0.append(A0 * 1e9); finite_t.append(t_lod)
        if finite_A0:
            ax12.plot(finite_A0, finite_t, '-o', color=colors_ps[d_nm], ms=4, lw=1.3,
                      label=f'{d_nm}nm')
        else:
            never_detected.append(d_nm)
    ax12.set_xscale('log')
    ax12.set(xlabel='$[A]_0$ (nM)', ylabel='Time to reach LOD (s)',
              title=f'Time to LOD vs $[A]_0$ by particle diameter  (α={ALPHA_FIXED}, With PA2)')
    ax12.legend(fontsize=7, ncol=2, title='diameter')
    ax12.grid(True, alpha=0.3)
    if never_detected:
        note = ('Not detected within T_sim for any tested [A]0: '
                 + ', '.join(f'{d}nm' for d in never_detected))
        ax12.text(0.02, 0.02, note, transform=ax12.transAxes, fontsize=7.5,
                  color='0.4', va='bottom', ha='left', wrap=True)
    fig12.tight_layout()
    fig12.savefig(os.path.join(IMG_DIR, 'fig12_time_to_LOD_vs_A0.png'), dpi=200)
    print("  → img/fig12_time_to_LOD_vs_A0.png 保存")

    # ---- Fig 13: 抗原量 vs 吸光度 (粒径ごと) ----
    fig13, ax13 = plt.subplots(figsize=(7, 5))
    for d_nm in PARTICLE_DIAMETERS_NM:
        vals = [results_A0[(d_nm, A0)]['abs_final'] for A0 in A0_VALUES]
        ax13.plot(np.array(A0_VALUES) * 1e9, np.maximum(vals, 1e-12), '-o',
                  color=colors_ps[d_nm], ms=4, lw=1.3, label=f'{d_nm}nm')
    ax13.axhline(ABSORBANCE_THRESHOLD, color='k', ls='--', lw=1, alpha=0.6)
    ax13.set_xscale('log'); ax13.set_yscale('log')
    ax13.set(xlabel='$[A]_0$ (nM)', ylabel=f'Absorbance (t={T_sim:.0f}s)',
              title=f'Absorbance vs $[A]_0$ by particle diameter  (α={ALPHA_FIXED}, With PA2)')
    ax13.legend(fontsize=7, ncol=2, title='diameter')
    ax13.grid(True, alpha=0.3)
    fig13.tight_layout()
    fig13.savefig(os.path.join(IMG_DIR, 'fig13_absorbance_vs_A0.png'), dpi=200)
    print("  → img/fig13_absorbance_vs_A0.png 保存")

    # ---- Fig 14: g(r,S) 有無での LOD到達時間の違い (d=100nm, [A]0=A0_def) ----
    D_STERIC_TEST = 100
    print(f"\n  g(r,S) 有無比較 (d={D_STERIC_TEST}nm, α={ALPHA_FIXED}, "
          f"[A]₀={A0_def*1e9:.0f}nM, PA2あり)...")
    steric_runs = {}
    for with_steric in [True, False]:
        sol14, x14, in_cap14, *_ = simulate(
            A0_def, P0_def, L0_def, 'PME', Nbase_sweep, t_eval=t_sweep,
            with_PA2=True, d_nm=D_STERIC_TEST, alpha=ALPHA_FIXED, with_steric=with_steric)
        lpa14, lpa214 = LPA_LPA2_series(sol14, Nbase_sweep, in_cap14)
        abs_series14 = calc_absorbance(lpa14 + lpa214, D_STERIC_TEST)
        t_lod14 = find_time_to_LOD(t_sweep, abs_series14, ABSORBANCE_THRESHOLD)
        steric_runs[with_steric] = dict(t=sol14.t, abs_series=abs_series14, t_lod=t_lod14)
        t_lod14_str = f"{t_lod14:.1f}s" if np.isfinite(t_lod14) else "N/A"
        print(f"    g(r,S){'あり' if with_steric else 'なし'}: t_LOD={t_lod14_str}")

    fig14, ax14 = plt.subplots(figsize=(6.5, 4.5))
    ax14.plot(steric_runs[True]['t'], np.maximum(steric_runs[True]['abs_series'], 1e-12),
              '-', color='C3', lw=1.5, label='With g(r,S)')
    ax14.plot(steric_runs[False]['t'], np.maximum(steric_runs[False]['abs_series'], 1e-12),
              '--', color='C0', lw=1.5, label='Without g(r,S)')
    ax14.axhline(ABSORBANCE_THRESHOLD, color='k', ls=':', lw=1, alpha=0.6, label='LOD threshold')
    ax14.set_yscale('log')
    ax14.set(xlabel='Time (s)', ylabel='Absorbance',
              title=f'Effect of g(r,S) on time to LOD  (d={D_STERIC_TEST}nm, α={ALPHA_FIXED}, '
                    f'[A]₀={A0_def*1e9:.0f}nM)')
    ax14.legend(fontsize=8)
    ax14.grid(True, alpha=0.3)
    fig14.tight_layout()
    fig14.savefig(os.path.join(IMG_DIR, 'fig14_steric_effect_on_LOD.png'), dpi=200)
    print("  → img/fig14_steric_effect_on_LOD.png 保存")

    print("\n全グラフ出力完了 (img/fig7_absorbance_vs_diameter.png ~ img/fig14_steric_effect_on_LOD.png)")
