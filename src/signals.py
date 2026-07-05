import numpy as np


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
