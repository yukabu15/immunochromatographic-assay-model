import numpy as np

from .params import xL1, xL2


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
