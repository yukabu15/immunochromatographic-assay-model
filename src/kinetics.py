import numpy as np
from dataclasses import dataclass

from .params import (
    KA0, KD0, DP, MEMBRANE_THICKNESS_CM,
    BOLTZMANN_J_PER_K, TEMPERATURE_K, WATER_VISCOSITY_PAS,
)

AVOGADRO = 6.02214076e23  # [1/mol]

# 抗体数 n(d) の基準点: d=5nm の粒子に抗体0.5個、以降は表面積(∝d^2)に比例
N_REF_ANTIBODIES = 0.5
D_REF_NM = 5.0


@dataclass
class RateConstants:
    """粒子径依存の反応網 (1)(1')(2)(3)(3')(4)(4')(5) の速度定数一式。

        (1)  A + P   ⇌ PA     ka1 , kd1
        (1') A + PA  ⇌ PA2    ka1p, kd1p
        (2)  A + L   ⇌ LA     ka2 , kd2
        (3)  PA + L  ⇌ LPA    ka3 , kd3
        (3') PA2+ L  ⇌ LPA2   ka3p, kd3p
        (4)  P + LA  ⇌ LPA    ka4 , kd4
        (4') PA+ LA  ⇌ LPA2   ka4p, kd4p
        (5)  A + LPA ⇌ LPA2   ka5 , kd5
    """
    ka1: float; kd1: float
    ka1p: float; kd1p: float
    ka2: float; kd2: float
    ka3: float; kd3: float
    ka3p: float; kd3p: float
    ka4: float; kd4: float
    ka4p: float; kd4p: float
    ka5: float; kd5: float
    n: float  # 粒子1個あたりの抗体数 (d_nm=None の旧モデルでは None)


def antibodies_per_particle(d_nm):
    """粒子表面の抗体数 n(d)。表面積(∝d^2)比例、d=5nmでn=0.5個と仮定。"""
    return N_REF_ANTIBODIES * (d_nm / D_REF_NM) ** 2


def steric_factor(d_nm, L0_tot):
    """
    リガンド占有の立体障害係数 g(r,S) = pi*r^2 / S。

    S は「リガンド1分子が専有する面積」。膜厚 b の中に濃度 [L0] で
    リガンドが分布していると考えると、面密度は [L0]*b*N_A [個/m^2] となり、
    S はその逆数 S = 1/([L0]*b*N_A) [m^2]。
    (元スライドの S=[L0]/(b*N_A) は次元・符号が破綻するため逆数を採用)

    d_nm=None の場合は旧モデル互換として g=1.0 を返す
    (L_free = L0 - LA - LPA - LPA2 と一致させるため)。
    """
    if d_nm is None:
        return 1.0

    r_m = (d_nm * 1e-9) / 2.0
    b_m = MEMBRANE_THICKNESS_CM / 100.0
    L0_mol_per_m3 = L0_tot * 1000.0
    S = 1.0 / (L0_mol_per_m3 * b_m * AVOGADRO)
    return np.pi * r_m ** 2 / S


def diffusion_coefficient(d_nm):
    """
    粒子関連種 (P, PA, PA2) の拡散係数。ストークス・アインシュタイン式
    D = kB*T / (6*pi*eta*r) で粒子径に依存させる。

    d_nm=None の場合は旧モデル互換として params.DP(=1e-12 m^2/s) を返す。
    """
    if d_nm is None:
        return DP

    r_m = (d_nm * 1e-9) / 2.0
    return BOLTZMANN_J_PER_K * TEMPERATURE_K / (6.0 * np.pi * WATER_VISCOSITY_PAS * r_m)


def rate_constants(d_nm, alpha=1.0, with_PA2=True):
    """
    粒子径 d_nm・律速指数 alpha に応じた速度定数一式を返す。

    d_nm=None の場合は旧モデル(サイズ非依存)の値をそのまま再現する
    (kd1p は 2*KD0 ではなく KD0 のまま — この "2倍" 補正は新モデル専用)。
    """
    if d_nm is None:
        ka1p, kd1p = (KA0, KD0) if with_PA2 else (0.0, 0.0)
        ka3p, kd3p = (KA0, KD0) if with_PA2 else (0.0, 0.0)
        return RateConstants(
            ka1=KA0, kd1=KD0,
            ka1p=ka1p, kd1p=kd1p,
            ka2=KA0, kd2=KD0,
            ka3=KA0, kd3=KD0,
            ka3p=ka3p, kd3p=kd3p,
            ka4=KA0, kd4=KD0,
            ka4p=0.0, kd4p=0.0,
            ka5=0.0, kd5=0.0,
            n=None,
        )

    n = antibodies_per_particle(d_nm)
    n_alpha = n ** alpha
    occ = max(n - 1.0, 0.0) / n if n > 0 else 0.0  # (n-1)/n をクランプ

    ka1 = n_alpha * KA0
    ka2 = KA0 / np.sqrt(2.0)
    ka3 = (n_alpha / np.sqrt(2.0)) * KA0
    ka4 = (n_alpha / np.sqrt(2.0)) * KA0
    kd1 = kd2 = kd3 = kd4 = KD0

    if with_PA2:
        ka1p, kd1p = occ * n_alpha * KA0, 2.0 * KD0
        ka3p, kd3p = (occ * n_alpha / np.sqrt(2.0)) * KA0, KD0
        ka4p, kd4p = (occ * n_alpha / np.sqrt(2.0)) * KA0, KD0
        ka5,  kd5  = occ * n_alpha * KA0, KD0
    else:
        ka1p = kd1p = ka3p = kd3p = ka4p = kd4p = ka5 = kd5 = 0.0

    return RateConstants(
        ka1=ka1, kd1=kd1,
        ka1p=ka1p, kd1p=kd1p,
        ka2=ka2, kd2=kd2,
        ka3=ka3, kd3=kd3,
        ka3p=ka3p, kd3p=kd3p,
        ka4=ka4, kd4=kd4,
        ka4p=ka4p, kd4p=kd4p,
        ka5=ka5, kd5=kd5,
        n=n,
    )
