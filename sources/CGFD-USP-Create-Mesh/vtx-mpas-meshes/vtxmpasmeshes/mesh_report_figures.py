"""
Schematic drawings of what the mesh checks test (used in the per-mesh report).

Drawn with matplotlib only; no mesh file needed.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
from scipy.spatial import Voronoi

NAVY, GREEN, ORANGE, GREY, LIGHT = '#14213D', '#1B6F57', '#C8601A', '#8A94A6', '#E8EDEE'
plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans'})


def hex_points(n=4, jitter=0.0, seed=1):
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            pts.append((x, y))
    pts = np.array(pts) + rng.normal(0, jitter, (len(pts), 2))
    return pts


def cell_polygon(vor, k):
    reg = vor.regions[vor.point_region[k]]
    return vor.vertices[reg] if -1 not in reg and reg else None


def order_ccw(poly, c):
    ang = np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0])
    return poly[np.argsort(ang)]


def circumcentre(p, q, r):
    d = 2 * (p[0] * (q[1] - r[1]) + q[0] * (r[1] - p[1]) + r[0] * (p[1] - q[1]))
    ux = ((p @ p) * (q[1] - r[1]) + (q @ q) * (r[1] - p[1]) + (r @ r) * (p[1] - q[1])) / d
    uy = ((p @ p) * (r[0] - q[0]) + (q @ q) * (p[0] - r[0]) + (r @ r) * (q[0] - p[0])) / d
    return np.array([ux, uy])


def fig_connectivity(path, dpi=150):
    pts = hex_points(3, jitter=0.07, seed=4)
    vor = Voronoi(pts)
    centre = int(np.argmin(np.hypot(*pts.T)))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))

    ax = axs[0]
    ax.set_title('Connectivity: one cell, its edges, vertices and neighbours', fontsize=12)
    poly_c = order_ccw(cell_polygon(vor, centre), pts[centre])
    nb = [k for k in range(len(pts)) if k != centre and cell_polygon(vor, k) is not None
          and np.linalg.norm(pts[k] - pts[centre]) < 1.5]
    nb = sorted(nb, key=lambda k: np.arctan2(*(pts[k] - pts[centre])[::-1]))
    for k in nb:
        ax.add_patch(Polygon(order_ccw(cell_polygon(vor, k), pts[k]), fc=LIGHT, ec=GREY, lw=1))
    ax.add_patch(Polygon(poly_c, fc='#CFE6DD', ec=NAVY, lw=2))
    ax.plot(*pts[centre], 'o', color=NAVY)
    for j, k in enumerate(nb):
        ax.plot(*pts[k], 'o', color=GREY, ms=4)
        ax.text(*(pts[k] + [0, 0.12]), f'cellsOnCell[{j + 1}]', ha='center', fontsize=8, color=GREY)
    nv = len(poly_c)
    for j in range(nv):
        a, b = poly_c[j], poly_c[(j + 1) % nv]
        mid = (a + b) / 2
        d = mid - pts[centre]
        ax.text(*(pts[centre] + 0.62 * d), f'e{j + 1}', ha='center', va='center', fontsize=9, color=GREEN, weight='bold')
        ax.plot(*a, 's', color=ORANGE, ms=5)
        ax.text(*(pts[centre] + 1.18 * (a - pts[centre])), f'v{j + 1}', ha='center', va='center', fontsize=9, color=ORANGE)
    ax.text(0.02, 0.02, 'Shaded: cell i. edgesOnCell e1…e6, verticesOnCell v1…v6 and cellsOnCell\n'
            'are listed counter-clockwise; neighbour j lies across edge j.\n'
            'Each vertex is shared by exactly 3 cells (vertexDegree = 3).',
            transform=ax.transAxes, fontsize=9.5, va='bottom')
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.85, 1.75); ax.set_aspect('equal'); ax.axis('off')

    ax = axs[1]
    ax.set_title('Voronoi property at one edge', fontsize=12)
    A, B = np.array([-1.0, 0.0]), np.array([1.0, 0.1])
    C1, C2 = np.array([0.15, 1.75]), np.array([-0.1, -1.7])




    V1, V2 = circumcentre(A, B, C1), circumcentre(A, B, C2)
    ax.add_patch(Circle(V1, np.linalg.norm(V1 - A), fill=False, ls=':', ec=ORANGE, lw=1.2))
    for P in (C1, C2):
        ax.plot([A[0], P[0], B[0]], [A[1], P[1], B[1]], ls='--', color=GREY, lw=0.8)
    ax.plot([A[0], B[0]], [A[1], B[1]], color=NAVY, lw=2)
    ax.plot([V1[0], V2[0]], [V1[1], V2[1]], color=GREEN, lw=3)
    for P, lab in ((A, 'cell 1'), (B, 'cell 2'), (C1, 'cell 3'), (C2, 'cell 4')):
        ax.plot(*P, 'o', color=NAVY)
        ax.text(P[0], P[1] + 0.15, lab, ha='center', fontsize=10, color=NAVY)
    ax.plot(*V1, 's', color=ORANGE); ax.plot(*V2, 's', color=ORANGE)
    ax.text(V1[0] - 0.12, V1[1] + 0.12, 'vertex = circumcentre\n(equally far from cells 1, 2, 3)', fontsize=9, color=ORANGE, ha='right')
    X = (A + B) / 2 + ((V1 - (A + B) / 2) @ ((B - A) / np.linalg.norm(B - A))) * 0
    t = (B - A) / np.linalg.norm(B - A); n = np.array([-t[1], t[0]])
    cross = V1 + ((A - V1) @ n) / (((V2 - V1) / np.linalg.norm(V2 - V1)) @ n) * (V2 - V1) / np.linalg.norm(V2 - V1)
    s = 0.13
    ax.add_patch(Polygon([cross, cross + s * t, cross + s * t + s * n * np.sign(((V1 - cross) @ n) or 1), cross + s * n * np.sign(((V1 - cross) @ n) or 1)],
                         fill=False, ec='k', lw=1))
    ax.text(A[0] - 0.05, A[1] - 0.35, 'dcEdge\n(cell-to-cell distance)', fontsize=9.5, color=NAVY, va='top')
    ax.text(V2[0] + 0.12, V2[1] + 0.25, 'dvEdge\n(Voronoi edge)', fontsize=9.5, color=GREEN)
    ax.add_patch(FancyArrowPatch(cross, cross + 0.7 * t, arrowstyle='-|>', mutation_scale=16, color=ORANGE, lw=2))
    ax.text(*(cross + [0.08, 0.42]), 'normal velocity u →', fontsize=9.5, color=ORANGE)
    ax.text(0.02, 0.02, 'Checked: the edge is perpendicular to the cell-to-cell line (right angle),\n'
            'each vertex is equidistant from its 3 cells, dcEdge and dvEdge are\n'
            'the true arc lengths, and dvEdge/dcEdge ≥ 0.12.', transform=ax.transAxes, fontsize=9.5, va='bottom')
    ax.set_xlim(-1.9, 2.1); ax.set_ylim(-2.6, 2.6); ax.set_aspect('equal'); ax.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)


def fig_areas(path, dpi=150):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
    P = np.array([[-1.0, -0.6], [1.1, -0.5], [0.05, 1.2]])
    cc = circumcentre(*P)
    cols = ['#BFE0D3', '#F3D2BC', '#C9D6E8']
    ax = axs[0]
    ax.set_title('One triangle (dual cell) = 3 kites', fontsize=12)
    for k in range(3):
        p, q, r = P[k], P[(k + 1) % 3], P[(k + 2) % 3]
        kite = [p, (p + q) / 2, cc, (p + r) / 2]
        ax.add_patch(Polygon(kite, fc=cols[k], ec='white', lw=2))
        ax.text(*np.mean(kite, axis=0), f'kite {k + 1}\n(part of cell {k + 1})', ha='center', va='center', fontsize=9)
    ax.add_patch(Polygon(P, fill=False, ec=NAVY, lw=2))
    ax.plot(*P.T, 'o', color=NAVY); ax.plot(*cc, 's', color=ORANGE)
    ax.text(cc[0] + 0.07, cc[1] + 0.08, 'vertex', color=ORANGE, fontsize=9)
    ax.text(0.02, 0.02, 'kite 1 + kite 2 + kite 3 = areaTriangle', transform=ax.transAxes, fontsize=10.5)
    ax.set_xlim(-1.3, 1.4); ax.set_ylim(-1.0, 1.5); ax.set_aspect('equal'); ax.axis('off')

    ax = axs[1]
    ax.set_title('One cell = the kites of its vertices', fontsize=12)
    pts = hex_points(2, jitter=0.0)
    vor = Voronoi(pts)
    c0 = int(np.argmin(np.hypot(*pts.T)))
    poly = order_ccw(cell_polygon(vor, c0), pts[c0])
    g = pts[c0]
    for j in range(len(poly)):
        v = poly[j]; vn = poly[(j + 1) % len(poly)]; vp = poly[j - 1]
        kite = [g, (v + vp) / 2, v, (v + vn) / 2]
        ax.add_patch(Polygon(kite, fc=['#BFE0D3', '#D6EADF'][j % 2], ec='white', lw=1.5))
    ax.add_patch(Polygon(poly, fill=False, ec=NAVY, lw=2))
    ax.plot(*g, 'o', color=NAVY); ax.plot(*poly.T, 's', color=ORANGE, ms=5)
    ax.text(0.02, 0.02, 'sum of the cell\'s kites = areaCell\nglobal mesh: all cells (and all triangles) cover 4πR²',
            transform=ax.transAxes, fontsize=10.5)
    ax.set_xlim(-0.95, 0.95); ax.set_ylim(-0.95, 0.85); ax.set_aspect('equal'); ax.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)


def fig_quality(path, dpi=150):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.0))
    ax = axs[0]
    fig.text(0.17, 0.93, 'dvEdge / dcEdge ≥ 0.12', ha='center', fontsize=12)
    for x0, dv, ok in ((0.0, 0.55, True), (2.2, 0.05, False)):
        A, B = np.array([x0 - 0.6, 0]), np.array([x0 + 0.6, 0])
        ax.plot([A[0], B[0]], [0, 0], color=NAVY, lw=1.5, ls='--')
        ax.plot([x0, x0], [-dv / 2, dv / 2], color=GREEN if ok else ORANGE, lw=4)
        ax.plot(*A, 'o', color=NAVY); ax.plot(*B, 'o', color=NAVY)
        ax.text(x0, -0.75, f'dv/dc = {dv / 1.2:.2f}\n' + ('fine' if ok else 'near-degenerate edge: error'),
                ha='center', fontsize=10, color=GREEN if ok else ORANGE)
    ax.set_xlim(-1, 3.2); ax.set_ylim(-1.1, 0.8); ax.set_aspect('equal'); ax.axis('off')

    ax = axs[1]
    fig.text(0.5, 0.93, 'Generator close to the centroid (≤ 0.15·√area)', ha='center', fontsize=12)
    for x0, shift, ok in ((0.0, 0.0, True), (2.3, 0.33, False)):
        ang = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
        poly = np.c_[x0 + 0.9 * np.cos(ang), 0.9 * np.sin(ang)]
        ax.add_patch(Polygon(poly, fc=LIGHT, ec=NAVY, lw=1.5))
        cen = poly.mean(axis=0); gen = cen + [shift, 0.12 * (shift > 0)]
        ax.plot(*cen, 'x', color=NAVY, ms=10, mew=2)
        ax.plot(*gen, 'o', color=GREEN if ok else ORANGE, ms=8)
        ax.text(x0, -1.35, 'generator on centroid' if ok else 'generator off centroid:\nwarning if > 0.15·√area',
                ha='center', fontsize=10, color=GREEN if ok else ORANGE)
    ax.text(0.5, 1.02, '× centroid   ● generator', transform=ax.transAxes, ha='center', fontsize=10)
    ax.set_xlim(-1.1, 3.4); ax.set_ylim(-1.75, 1.1); ax.set_aspect('equal'); ax.axis('off')

    ax = axs[2]
    fig.text(0.83, 0.93, 'Smooth grading (neighbour size ratio ≤ 1.5)', ha='center', fontsize=12)
    for y0, sizes, ok in ((0.9, [0.30, 0.34, 0.39, 0.45, 0.52, 0.60], True), (-0.6, [0.30, 0.30, 0.30, 0.60, 0.60, 0.60], False)):
        x = 0.0
        for s in sizes:
            ang = np.linspace(0, 2 * np.pi, 7)[:-1]
            ax.add_patch(Polygon(np.c_[x + s / 2 + s / 2 * np.cos(ang), y0 + s / 2 * np.sin(ang)], fc=LIGHT, ec=NAVY, lw=1.2))
            x += s
        r = max(b / a for a, b in zip(sizes, sizes[1:]))
        ax.text(x / 2, y0 - 0.55, f'max ratio {r:.2f}: ' + ('smooth' if ok else 'abrupt jump, warning'),
                ha='center', fontsize=10, color=GREEN if ok else ORANGE)
    ax.set_xlim(-0.2, 2.9); ax.set_ylim(-1.4, 1.4); ax.set_aspect('equal'); ax.axis('off')
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(path, dpi=dpi); plt.close(fig)
