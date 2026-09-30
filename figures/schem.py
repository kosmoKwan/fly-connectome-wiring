"""Small vector-drawing helpers for the schematic panels (Fig. 1 and Fig. 4a). Pure matplotlib, no data."""
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Polygon, PathPatch
from matplotlib.path import Path
import common as C


def node(ax, x, y, r=0.9, fc="white", ec=C.INK, lw=0.6, z=5, **kw):
    c = Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z, **kw)
    ax.add_patch(c)
    return c


def box(ax, x, y, w, h, fc="white", ec=C.INK, lw=0.6, r=1.2, z=2, **kw):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z, **kw)
    ax.add_patch(b)
    return b


def arrow(ax, p0, p1, color=C.INK, lw=0.8, head=4.5, style="-|>", rad=0.0, z=3, shrinkA=0, shrinkB=0, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=f"{style},head_length={head * 0.12},head_width={head * 0.07}" if style != "-" else "-",
                        connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, zorder=z, shrinkA=shrinkA, shrinkB=shrinkB,
                        linestyle=ls, mutation_scale=10)
    ax.add_patch(a)
    return a


def inhib(ax, p0, p1, color=C.INK, lw=0.8, bar=0.9, z=3, rad=0.0):
    """Line ending in a flat bar (inhibitory synapse convention)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax.add_patch(FancyArrowPatch(tuple(p0), tuple(p1), arrowstyle="-", connectionstyle=f"arc3,rad={rad}", color=color, lw=lw,
                                 zorder=z, shrinkA=0, shrinkB=0))
    d = p1 - p0; d = d / (np.linalg.norm(d) + 1e-12); n = np.array([-d[1], d[0]])
    a, b = p1 + n * bar, p1 - n * bar
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw * 1.3, solid_capstyle="butt", zorder=z)


def curve(ax, pts, color=C.INK, lw=0.8, z=3, ls="-"):
    """Smooth quadratic-Bezier chain through control points (for leg outline etc.)."""
    verts = [pts[0]]; codes = [Path.MOVETO]
    for i in range(1, len(pts) - 1, 2):
        verts += [pts[i], pts[i + 1]]; codes += [Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(verts, codes), fc="none", ec=color, lw=lw, zorder=z, ls=ls))


def capsule(ax, p0, p1, width, fc, ec=C.INK, lw=0.6, z=2):
    """Rounded segment (leg segment) from p0 to p1 with given width."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; L = np.linalg.norm(d); ang = np.degrees(np.arctan2(d[1], d[0]))
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.transforms import Affine2D
    b = FancyBboxPatch((0, -width / 2), L, width, boxstyle=f"round,pad=0,rounding_size={width / 2}", fc=fc, ec=ec, lw=lw, zorder=z)
    b.set_transform(Affine2D().rotate_deg(ang).translate(*p0) + ax.transData)
    ax.add_patch(b)
    return b
