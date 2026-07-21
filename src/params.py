# ============================================================
# パラメータ (Qian & Bau 2003 ベース + 拡張)
# ============================================================
import numpy as np
# 反応速度定数 (全て同一値)
#  ①  A + P   → PA      (ka1, kd1)
#  ②  A + L   → LA      (ka2, kd2)   ※L_free を使用
#  ③  PA + L  → LPA     (ka3, kd3)
#  ④  P + LA  → LPA     (ka4, kd4)
#  ⑤  PA + A  → PA₂     (ka5, kd5)   ★新規
#  ⑥  PA₂ + L → LPA₂    (ka6, kd6)   ★新規
KA0 = 1e6   # 1/(M·s)  基準の会合速度定数 (粒子径依存モデルの ka,0)
KD0 = 1e-3  # 1/s      基準の解離速度定数 (粒子径依存モデルの kd,0)
ka1 = ka2 = ka3 = ka4 = ka5 = ka6 = KA0
kd1 = kd2 = kd3 = kd4 = kd5 = kd6 = KD0

# デフォルト濃度
A0_def = P0_def = L0_def = 10e-9  # 10 nM

# 抗原量 [A]0 を振る場合のスイープ候補値 [M] (0.01 nM ~ 10 uM, 対数間隔10点)
A0_VALUES = np.logspace(-11, -5, 10)

# メンブレン・流れパラメータ
L    = 0.041       # メンブレン長さ [m]
xL1  = L / 2       # キャプチャーゾーン開始
xL2  = 3 * L / 4   # キャプチャーゾーン終了
U    = 2e-4        # 流速 [m/s]
DA   = 1e-10       # アナライト拡散係数 [m²/s]
DP   = 1e-12       # 粒子拡散係数 [m²/s]  (PA₂ も同値)
T_sim = 600.0      # シミュレーション時間 [s]

# 粒子径・LOD パラメータ
PARTICLE_DIAMETERS_NM = [3, 10, 15, 20, 30, 40, 50, 60, 80, 100]  # 粒子径 [nm]
MEMBRANE_THICKNESS_CM = 0.013   # 光路長 = メンブレン厚さ 130 μm [cm]
ABSORBANCE_THRESHOLD  = 0.05    # LOD 判定閾値 (吸光度)

# 粒子径依存キネティクス用の物理定数 (ストークス・アインシュタイン式 D=kB*T/(6*pi*eta*r) 用)
BOLTZMANN_J_PER_K   = 1.380649e-23  # ボルツマン定数 [J/K]
TEMPERATURE_K       = 298.0         # 温度 25℃ [K]
WATER_VISCOSITY_PAS = 1.0e-3        # 水の粘度 [Pa·s]
