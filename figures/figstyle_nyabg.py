"""Shared style for the NYABG revision figures (Microbiome resubmission).
Import: sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *
"""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from PIL import Image
# Entity colours (thread across every figure). Okabe-Ito based, CVD-safe; same hue families as the submitted figures.
COL = {"burial": "#A6611A", "control": "#009E73", "emp": "#0072B2", "hmp": "#D55E00", "unknown": "#BDBDBD"}
LABEL = {"burial": "Burial", "control": "Control", "emp": "EMP", "hmp": "HMP", "unknown": "Unattributed"}
# Proximity levels within burial (dark to light brown), control keeps its own colour
PROX = {"Cranium": "#7F4A10", "Other Body": "#B5772F", "Soil Matrix": "#DDB57F", "Control": COL["control"]}
PROX_LABEL = {"Cranium": "Cranium", "Other Body": "Other body region", "Soil Matrix": "Soil matrix", "Control": "Control"}
# HMP body sites within the vermillion family; EMP habitat categories within the blue family
HMP_SITE = {"hmp_oral": "#8C2D04", "hmp_nasal": "#D55E00", "hmp_skin": "#F59E5B", "hmp_stool": "#FDD0A2", "hmp_vaginal": "#E7A07A"}
EMP_TYPE = {"emp_soil": "#08306B", "emp_animal": "#2171B5", "emp_other": "#6BAED6", "emp_sediment": "#9ECAE1", "emp_water": "#4292C6", "emp_plant": "#C6DBEF"}
ALARM = "#CC0000"   # reserved: contamination / blank marks only
FIG_W_DOUBLE_MM, FIG_W_SINGLE_MM = 170, 85
def mm(x): return x / 25.4
def setup():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150, "axes.titleweight": "regular",
        "axes.titlelocation": "left"})
def letter(ax, s, dx=-0.12, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
def check_overlaps(fig):
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    bad = [(a.get_text()[:30], b.get_text()[:30]) for i, (a, ba) in enumerate(texts) for b, bb in texts[i+1:] if ba.overlaps(bb)]
    fb = fig.bbox; out = [t.get_text()[:30] for t, bt in texts if bt.x0 < fb.x0-1 or bt.y0 < fb.y0-1 or bt.x1 > fb.x1+1 or bt.y1 > fb.y1+1]
    return bad, out
def save(fig, stem, outdir="figures/out"):
    """Save PNG (300 dpi), TIFF (300 dpi, LZW, RGB) and PDF; return overlap report."""
    fig.canvas.draw(); bad, out = check_overlaps(fig)
    png = f"{outdir}/{stem}.png"; fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(f"{outdir}/{stem}.pdf", bbox_inches="tight")
    Image.open(png).convert("RGB").save(f"{outdir}/{stem}.tiff", compression="tiff_lzw", dpi=(300, 300))
    return {"png": png, "overlaps": bad, "outside": out}
