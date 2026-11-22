"""Print the rows of Tabs. 9 and 10 from results/log_*.log of _run_bccb.py."""
import math
import re
from pathlib import Path

OUT = Path(__file__).parent / "results"
N = {"imagenette": 307200, "div2k": 3145728}
LAPLEX = {"imagenette": ("ImageNette-320", "LAPLEX ($I=1$)", 2.23e6, 198549),
          "div2k": ("DIV2K-1024", "LAPLEX (width $1000$)", 13.6e6, 1266679)}


def read(tag):
    f = OUT / f"log_{tag}.log"
    if not f.exists():
        return None
    text = f.read_text()
    m = re.search(r"DONE \S+: test_LL=(-?[\d.]+).*params=(\d+)", text)
    if m:
        return float(m[1]), int(m[2])
    m = re.search(r"diag floor: test_LL=(-?[\d.]+)", text)
    return (float(m[1]), N[tag.split("_")[0]]) if m else None


def row(first, name, res, n):
    ll, params = res
    tex_ll = f"${ll:+,.0f}$".replace(",", "{,}")
    return (f"{first} & {name} & ${params / 1e6:.3g}$M & {tex_ll} & "
            f"${-ll / (n * math.log(2)):+.3f}$\\\\")


for table, order in (("9", "permuted"), ("10", "raster")):
    print(f"% Tab. {table}, {order} coordinates")
    for ds, K in (("imagenette", 5), ("div2k", 3)):
        first, name, params, ll = LAPLEX[ds]
        print(row(first, name, (ll, params), N[ds]))
        for label, tag in ((f"Circulant mixture ($K={K}$)", f"{ds}_mix{K}_{order}"),
                           ("Circulant ($K=1$)", f"{ds}_full_{order}"),
                           ("Diagonal", f"{ds}_diag_raster")):
            res = read(tag)
            print(row("", label, res, N[ds]) if res else f"% {tag}: no result")
        print("\\midrule" if ds == "imagenette" else "")
