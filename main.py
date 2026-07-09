import os

import numpy as np
import matplotlib.pyplot as plt

IMG_DIR = 'img'

from src.params import (
    A0_def, P0_def, L0_def, T_sim,
    PARTICLE_DIAMETERS_NM, ABSORBANCE_THRESHOLD,
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

    print("\n全グラフ出力完了 (img/fig7_absorbance_vs_diameter.png ~ img/fig9_time_to_LOD_vs_diameter.png)")
