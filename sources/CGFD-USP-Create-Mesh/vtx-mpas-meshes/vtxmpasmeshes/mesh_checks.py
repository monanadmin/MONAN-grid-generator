"""
Checks that an MPAS mesh satisfies what MPAS and this generator assume.

Three groups:
  1. MPAS Voronoi (SCVT) structure: geometry on the sphere, connectivity,
     Voronoi property, areas, dvEdge/dcEdge, near-centroidal cells, smooth
     grading.
  2. Limited-area structure (regional meshes): the 7 boundary layers, graph
     file, partition file.
  3. Generator assumptions (need the parameters p from mesh_config.resolve):
     cell size follows the requested map, boundary layers sit in the constant
     ring resolution, the cut region is fully kept.

Each check returns a CheckResult. Severity 'error' means the mesh must not be
used; 'warning' means it deviates from what is expected and should be looked
at. Thresholds marked "empirical" were chosen from the reference meshes built
by tests/test_mesh_generation.py and leave a safety margin above the values
those meshes reach.
"""

from dataclasses import dataclass

import numpy as np
import netCDF4

from vtxmpasmeshes import mesh_geometry as geo
from vtxmpasmeshes.mesh_config import ring_resolution

N_BDY_LAYERS = 7          # MPAS: 5 relaxation + 2 specified layers
MIN_DVDC = 0.12           # min dvEdge/dcEdge; smaller Voronoi edges hurt MPAS stability
REL_TOL = 1e-6            # exact geometric identities, up to round-off
MAX_CVT_OFFSET = 0.15     # empirical: |centroid - generator| / sqrt(areaCell)
MAX_GRADING = 1.5         # empirical: max size ratio of neighbouring cells
SIZE_TOL = 0.25           # empirical: |cell size / target - 1| allowed per cell
SIZE_MEDIAN_TOL = 0.10    # empirical: |median(cell size / target) - 1|
SIZE_FRACTION = 0.95      # fraction of cells that must be within SIZE_TOL


@dataclass
class CheckResult:
    name: str
    ok: bool
    severity: str
    message: str


def _res(name, ok, message, severity='error'):
    return CheckResult(name, bool(ok), severity, message)


class Mesh:
    """Arrays of an MPAS grid file, 0-based connectivity (-1 = missing)."""

    def __init__(self, grid_file):
        with netCDF4.Dataset(grid_file) as nc:
            nc.set_auto_mask(False)
            self.attrs = {k: nc.getncattr(k) for k in nc.ncattrs()}
            self.v = {k: nc.variables[k][:] for k in nc.variables}
            self.dims = {k: len(d) for k, d in nc.dimensions.items()}
        self.R = float(self.attrs.get('sphere_radius', 1.0))
        self.nCells = self.dims['nCells']
        self.nEdges = self.dims['nEdges']
        self.nVertices = self.dims['nVertices']
        self.regional = 'bdyMaskCell' in self.v
        ne = self.v['nEdgesOnCell'].astype(int)
        self.nEdgesOnCell = ne
        maxe = self.v['cellsOnCell'].shape[1]
        self.valid = np.arange(maxe)[None, :] < ne[:, None]

        def idx(name, n):
            a = self.v[name].astype(np.int64) - 1
            a[(a < 0) | (a >= n)] = -1
            return a
        self.coc = idx('cellsOnCell', self.nCells)
        self.eoc = idx('edgesOnCell', self.nEdges)
        self.voc = idx('verticesOnCell', self.nVertices)
        self.coe = idx('cellsOnEdge', self.nCells)
        self.voe = idx('verticesOnEdge', self.nVertices)
        self.cov = idx('cellsOnVertex', self.nCells)
        self.eov = idx('edgesOnVertex', self.nEdges)
        self.xc = np.stack([self.v['xCell'], self.v['yCell'], self.v['zCell']], 1)
        self.xv = np.stack([self.v['xVertex'], self.v['yVertex'], self.v['zVertex']], 1)
        self.xe = np.stack([self.v['xEdge'], self.v['yEdge'], self.v['zEdge']], 1)
        self.lat = np.degrees(self.v['latCell'])
        self.lon = geo.wrap_lon(np.degrees(self.v['lonCell']))
        # physical lengths in km (MPAS-Tools meshes are on the unit sphere)
        self.km = geo.EARTH_RADIUS_KM / self.R


def _arc(a, b, R):
    """Great-circle distance between rows of a and b on a sphere of radius R."""
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    dot = np.einsum('ij,ij->i', a, b)
    return R * np.arctan2(cross, dot)


# --------------------------------------------------------------------------
# 1. MPAS Voronoi structure
# --------------------------------------------------------------------------
def check_finite(m):
    bad = [k for k, a in m.v.items()
           if np.issubdtype(a.dtype, np.floating) and not np.all(np.isfinite(a))]
    return _res('finite values', not bad,
                'all floating-point fields are finite' if not bad
                else f'NaN/inf in: {", ".join(bad)}')


def check_on_sphere(m):
    worst = max(np.max(np.abs(np.linalg.norm(x, axis=1) / m.R - 1))
                for x in (m.xc, m.xv, m.xe))
    return _res('points on the sphere', worst < 1e-9,
                f'max | |x|/R - 1 | = {worst:.2e}')


def check_latlon_consistent(m):
    errs = []
    for x, la, lo in ((m.xc, 'latCell', 'lonCell'), (m.xv, 'latVertex', 'lonVertex')):
        lat = np.arcsin(np.clip(x[:, 2] / m.R, -1, 1))
        lon = np.arctan2(x[:, 1], x[:, 0])
        dlon = np.angle(np.exp(1j * (lon - m.v[lo])))
        errs.append(max(np.max(np.abs(lat - m.v[la])), np.max(np.abs(dlon))))
    worst = max(errs)
    return _res('lat/lon match x,y,z', worst < 1e-8, f'max angle mismatch {worst:.2e} rad')


def check_connectivity(m):
    problems = []
    ne = m.nEdgesOnCell
    maxe = m.coc.shape[1]
    if ne.min() < 3 or ne.max() > maxe:
        problems.append(f'nEdgesOnCell outside 3..{maxe} ({ne.min()}..{ne.max()})')
    if m.cov.shape[1] != 3:
        problems.append(f'vertexDegree is {m.cov.shape[1]}, MPAS needs 3')
    i = np.repeat(np.arange(m.nCells)[:, None], maxe, 1)[m.valid]
    e = m.eoc[m.valid]
    v = m.voc[m.valid]
    c = m.coc[m.valid]
    if (e < 0).any() or (v < 0).any():
        problems.append('edgesOnCell/verticesOnCell reference missing entries')
    else:
        if not np.all((m.coe[e, 0] == i) | (m.coe[e, 1] == i)):
            problems.append('a cell is not in cellsOnEdge of one of its edges')
        other = np.where(m.coe[e, 0] == i, m.coe[e, 1], m.coe[e, 0])
        if not np.all(other == c):
            problems.append('cellsOnCell[j] is not the neighbour across edgesOnCell[j]')
        if not np.all((m.cov[v] == i[:, None]).any(1)):
            problems.append('a cell is not in cellsOnVertex of one of its vertices')
        if not np.all((m.voe[e] == v[:, None]).any(1)):
            problems.append('verticesOnCell[j] is not an end of edgesOnCell[j]')
    if (m.voe < 0).any():
        problems.append('an edge has a missing vertex')
    else:
        for k in (0, 1):
            vk = m.voe[:, k]
            if not np.all((m.eov[vk] == np.arange(m.nEdges)[:, None]).any(1)):
                problems.append('an edge is not in edgesOnVertex of its vertex')
                break
    if ((m.coe < 0).all(1)).any():
        problems.append('an edge has no cell')
    missing_neighbours = int((c < 0).sum())
    if not m.regional and missing_neighbours:
        problems.append(f'{missing_neighbours} missing neighbours in a global mesh')
    return _res('connectivity', not problems,
                'cells, edges and vertices reference each other consistently'
                if not problems else '; '.join(problems))


def check_euler(m):
    chi = m.nVertices - m.nEdges + m.nCells
    want = 1 if m.regional else 2
    shape = 'a single region without holes' if m.regional else 'a sphere'
    return _res('topology (V - E + F)', chi == want,
                f'V - E + F = {chi}, expected {want} for {shape}')


def check_positive(m):
    bad = [k for k in ('dcEdge', 'dvEdge', 'areaCell', 'areaTriangle')
           if not np.all(m.v[k] > 0)]
    return _res('positive lengths and areas', not bad,
                'dcEdge, dvEdge, areaCell, areaTriangle > 0' if not bad
                else f'non-positive values in {", ".join(bad)}')


def check_dvdc(m):
    ratio = m.v['dvEdge'] / m.v['dcEdge']
    worst = float(np.min(ratio))
    return _res('dvEdge/dcEdge', worst >= MIN_DVDC,
                f'min dvEdge/dcEdge = {worst:.3f} (MPAS needs >= {MIN_DVDC})')


def check_voronoi(m):
    """Vertices are circumcentres: equidistant from their cells; dc/dv are arcs."""
    problems = []
    d = []
    for k in range(3):
        ck = m.cov[:, k]
        ok = ck >= 0
        dk = np.full(m.nVertices, np.nan)
        dk[ok] = _arc(m.xv[ok], m.xc[ck[ok]], m.R)
        d.append(dk)
    d = np.stack(d, 1)
    spread = (np.nanmax(d, 1) - np.nanmin(d, 1)) / np.nanmean(d, 1)
    if np.nanmax(spread) > REL_TOL:
        problems.append(f'vertex not equidistant from its cells (rel. spread {np.nanmax(spread):.1e})')
    both = (m.coe >= 0).all(1)
    dc = _arc(m.xc[m.coe[both, 0]], m.xc[m.coe[both, 1]], m.R)
    err_dc = np.max(np.abs(dc / m.v['dcEdge'][both] - 1))
    if err_dc > REL_TOL:
        problems.append(f'dcEdge differs from cell-centre distance (rel. {err_dc:.1e})')
    dv = _arc(m.xv[m.voe[:, 0]], m.xv[m.voe[:, 1]], m.R)
    err_dv = np.max(np.abs(dv / m.v['dvEdge'] - 1))
    if err_dv > REL_TOL:
        problems.append(f'dvEdge differs from vertex distance (rel. {err_dv:.1e})')
    # orthogonality: Voronoi edge is perpendicular to the line joining its cells
    a = m.xc[m.coe[both, 1]] - m.xc[m.coe[both, 0]]
    b = m.xv[m.voe[both, 1]] - m.xv[m.voe[both, 0]]
    cos = np.abs(np.einsum('ij,ij->i', a, b)) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    if np.max(cos) > 1e-5:
        problems.append(f'edge not orthogonal to cell-centre line (|cos| up to {np.max(cos):.1e})')
    return _res('Voronoi property', not problems,
                'vertices are circumcentres, edges orthogonal, dcEdge/dvEdge exact'
                if not problems else '; '.join(problems))


def check_areas(m):
    problems = []
    kite = m.v['kiteAreasOnVertex']
    err_t = np.max(np.abs(kite.sum(1) / m.v['areaTriangle'] - 1))
    if err_t > REL_TOL:
        problems.append(f'kite areas do not sum to areaTriangle (rel. {err_t:.1e})')
    acc = np.zeros(m.nCells)
    for k in range(3):
        ck = m.cov[:, k]
        ok = ck >= 0
        np.add.at(acc, ck[ok], kite[ok, k])
    complete = (m.cov[m.voc.clip(0)] >= 0).all(2)
    complete = np.where(m.valid, complete, True).all(1)
    err_c = np.max(np.abs(acc[complete] / m.v['areaCell'][complete] - 1))
    if err_c > REL_TOL:
        problems.append(f'kite areas do not sum to areaCell (rel. {err_c:.1e})')
    if not m.regional:
        sphere = 4 * np.pi * m.R ** 2
        for k in ('areaCell', 'areaTriangle'):
            err = abs(m.v[k].sum() / sphere - 1)
            if err > REL_TOL:
                problems.append(f'sum of {k} differs from the sphere area (rel. {err:.1e})')
    return _res('areas', not problems,
                'kite, cell, triangle and total areas are consistent'
                if not problems else '; '.join(problems))


def cvt_offset(m):
    """|centroid - generator| / sqrt(areaCell) for every cell."""
    maxe = m.voc.shape[1]
    c = m.xc
    acc = np.zeros_like(c)
    for j in range(maxe):
        jn = (j + 1) % np.maximum(m.nEdgesOnCell, 1)
        ok = m.valid[:, j]
        va = m.xv[m.voc[:, j].clip(0)]
        vb = m.xv[m.voc[np.arange(m.nCells), jn].clip(0)]
        area = 0.5 * np.linalg.norm(np.cross(va - c, vb - c), axis=1)
        acc += np.where(ok[:, None], area[:, None] * (c + va + vb) / 3.0, 0.0)
    centroid = acc / np.linalg.norm(acc, axis=1)[:, None] * m.R
    return np.linalg.norm(centroid - c, axis=1) / np.sqrt(m.v['areaCell'])


def check_cvt(m):
    off = cvt_offset(m)
    worst = float(np.max(off))
    return _res('near-centroidal cells', worst <= MAX_CVT_OFFSET,
                f'max |centroid - generator| / sqrt(area) = {worst:.3f} '
                f'(limit {MAX_CVT_OFFSET})', severity='warning')


def grading(m):
    size = np.sqrt(m.v['areaCell'])
    i = np.repeat(np.arange(m.nCells)[:, None], m.coc.shape[1], 1)[m.valid]
    c = m.coc[m.valid]
    ok = c >= 0
    return size[i[ok]] / size[c[ok]]


def check_grading(m):
    worst = float(np.max(grading(m)))
    return _res('smooth grading', worst <= MAX_GRADING,
                f'max size ratio of neighbouring cells = {worst:.3f} (limit {MAX_GRADING})',
                severity='warning')


# --------------------------------------------------------------------------
# 2. Limited-area structure
# --------------------------------------------------------------------------
def check_boundary_layers(m):
    if not m.regional:
        return None
    problems = []
    mask = m.v['bdyMaskCell'].astype(int)
    values = set(np.unique(mask).tolist())
    want = set(range(N_BDY_LAYERS + 1))
    if values != want:
        problems.append(f'bdyMaskCell values {sorted(values)}, expected 0..{N_BDY_LAYERS}')
    nb = np.where(m.coc >= 0, mask[m.coc.clip(0)], -1)
    nb = np.where(m.valid, nb, -2)              # -2 = not an edge of the cell
    missing = (nb == -1).any(1)
    if (missing & (mask != N_BDY_LAYERS)).any():
        problems.append(f'cells with missing neighbours outside layer {N_BDY_LAYERS}')
    present = np.where(nb >= 0, nb, 99)
    lowest = present.min(1)
    highest = np.where(nb >= 0, nb, -99).max(1)
    interior = mask == 0
    if (highest[interior] > 1).any():
        problems.append('interior cells touch boundary layers other than layer 1')
    k = mask[~interior]
    if (lowest[~interior] != k - 1).any():
        problems.append('a boundary-layer cell does not touch the previous layer')
    if (np.abs(highest[~interior] - k) > 1).any():
        problems.append('neighbouring cells skip a boundary layer')
    for name in ('bdyMaskEdge', 'bdyMaskVertex'):
        if name not in m.v:
            problems.append(f'{name} missing')
        elif not set(np.unique(m.v[name]).tolist()) <= want:
            problems.append(f'{name} has values outside 0..{N_BDY_LAYERS}')
    return _res('boundary layers', not problems,
                f'{N_BDY_LAYERS} complete, nested boundary layers'
                if not problems else '; '.join(problems))


def check_graph_file(m, graph_file):
    lines = open(graph_file).read().split('\n')
    header = lines[0].split()
    both = int((m.coe >= 0).all(1).sum())
    problems = []
    if [int(x) for x in header[:2]] != [m.nCells, both]:
        problems.append(f'header {header[:2]}, expected [{m.nCells}, {both}]')
    body = lines[1:1 + m.nCells]
    if len(body) != m.nCells:
        problems.append(f'{len(body)} cell lines, expected {m.nCells}')
    else:
        for i, line in enumerate(body):
            want = m.coc[i, :m.nEdgesOnCell[i]]
            want = (want[want >= 0] + 1).tolist()
            if [int(x) for x in line.split()] != want:
                problems.append(f'line for cell {i + 1} does not match cellsOnCell')
                break
    return _res('graph.info', not problems,
                'matches the mesh connectivity' if not problems else '; '.join(problems))


def check_partition_file(m, partition_file, nparts):
    parts = np.loadtxt(partition_file, dtype=int, ndmin=1)
    problems = []
    if parts.size != m.nCells:
        problems.append(f'{parts.size} entries, expected {m.nCells}')
    elif parts.min() < 0 or parts.max() >= nparts:
        problems.append(f'partition ids outside 0..{nparts - 1}')
    elif len(np.unique(parts)) != nparts:
        problems.append(f'only {len(np.unique(parts))} of {nparts} partitions used')
    return _res('partition file', not problems,
                f'{nparts} partitions covering all cells' if not problems
                else '; '.join(problems))


# --------------------------------------------------------------------------
# 3. Generator assumptions
# --------------------------------------------------------------------------
def cell_size_km(m):
    """Mean distance to the neighbouring cell centres (JIGSAW's target length)."""
    dc = np.where(m.valid & (m.coc >= 0), m.v['dcEdge'][m.eoc.clip(0)], np.nan)
    return np.nanmean(dc, 1) * m.km


def target_km(m, p):
    n_layers = p.get('n_layers', 0)
    return geo.target_resolution(m.lat, m.lon, p, ring_resolution(p), n_layers)


def check_cell_size(m, p):
    ratio = cell_size_km(m) / target_km(m, p)
    ok_cells = np.abs(ratio - 1) <= SIZE_TOL
    frac = float(np.mean(ok_cells))
    med = float(np.median(ratio))
    ok = frac >= SIZE_FRACTION and abs(med - 1) <= SIZE_MEDIAN_TOL
    return _res('cell size follows the requested map', ok,
                f'median size/target = {med:.3f}, {100 * frac:.1f}% of cells within '
                f'±{100 * SIZE_TOL:.0f}% (need ≥{100 * SIZE_FRACTION:.0f}%)',
                severity='warning')


def check_ring(m, p):
    """
    Boundary-layer cells must lie in the constant ring beyond the cut line,
    where the target is exactly the ring size. Layer-1 cells may straddle the
    cut line by less than half a cell; their actual size must still be close
    to the ring size.
    """
    if not m.regional:
        return None
    ring = ring_resolution(p)
    bdy = m.v['bdyMaskCell'] > 0
    d = geo.distance_beyond_cut(m.lat[bdy], m.lon[bdy], p)
    t = target_km(m, p)[bdy]
    size = cell_size_km(m)[bdy] / ring
    beyond = d > 0
    worst = float(np.max(np.abs(t[beyond] / ring - 1)))
    inside = float(max(0.0, -np.min(d)))
    problems = []
    if worst > REL_TOL:
        problems.append(f'target size beyond the cut deviates {100 * worst:.2g}% from the ring size')
    if inside > 0.5 * ring:
        problems.append(f'boundary-layer cells up to {inside:.0f} km inside the cut line')
    if np.any(np.abs(size - 1) > SIZE_TOL):
        problems.append('actual boundary-layer cell sizes deviate more than '
                        f'{100 * SIZE_TOL:.0f}% from the ring size')
    return _res('boundary layers at uniform resolution', not problems,
                f'all {int(bdy.sum())} boundary-layer cells within the {ring:g} km ring '
                f'(at most {inside:.0f} km inside the cut line); actual sizes '
                f'{size.min():.2f}-{size.max():.2f} x {ring:g} km'
                if not problems else '; '.join(problems))


def _cut_area_perimeter(p):
    if p['cut'] == 'circle':
        a = p['cut_radius'] / geo.EARTH_RADIUS_KM
        return (2 * np.pi * geo.EARTH_RADIUS_KM ** 2 * (1 - np.cos(a)),
                2 * np.pi * geo.EARTH_RADIUS_KM * np.sin(a))
    A, B = p['cut_a_km'], p['cut_b_km']
    h = ((A - B) / (A + B)) ** 2
    perim = np.pi * (A + B) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
    return np.pi * A * B, perim


def check_cut_kept(m, p):
    """The interior covers the requested cut; boundary layers lie outside it."""
    if not m.regional:
        return None
    ring = ring_resolution(p)
    mask = m.v['bdyMaskCell']
    d = geo.distance_beyond_cut(m.lat, m.lon, p)
    problems = []
    if np.max(d[mask == 0]) > 1.5 * ring:
        problems.append(f'interior cells up to {np.max(d[mask == 0]):.0f} km outside the cut')
    if np.min(d[mask > 0]) < -1.5 * ring:
        problems.append(f'boundary-layer cells up to {-np.min(d[mask > 0]):.0f} km inside the cut')
    area, perim = _cut_area_perimeter(p)
    interior_area = float(np.sum(m.v['areaCell'][mask == 0])) * m.km ** 2
    if interior_area < area - perim * ring:
        problems.append(f'interior area {interior_area:.4g} km² is smaller than the cut '
                        f'({area:.4g} km²)')
    return _res('cut region kept', not problems,
                f'interior {interior_area:.4g} km² for a cut of {area:.4g} km²'
                if not problems else '; '.join(problems))


# --------------------------------------------------------------------------
def check_mesh(grid_file, p=None, graph_file=None, partition_file=None, nparts=None):
    """Run every applicable check; returns a list of CheckResult."""
    m = Mesh(grid_file)
    checks = [check_finite, check_on_sphere, check_latlon_consistent,
              check_connectivity, check_euler, check_positive, check_dvdc,
              check_voronoi, check_areas, check_cvt, check_grading,
              check_boundary_layers]
    results = [c(m) for c in checks]
    if graph_file:
        results.append(check_graph_file(m, graph_file))
    if partition_file:
        results.append(check_partition_file(m, partition_file, nparts))
    if p is not None:
        results += [check_cell_size(m, p), check_ring(m, p), check_cut_kept(m, p)]
    return [r for r in results if r is not None]


def has_errors(results):
    return any(not r.ok and r.severity == 'error' for r in results)


def report(results):
    lines = []
    for r in results:
        tag = 'OK  ' if r.ok else ('FAIL' if r.severity == 'error' else 'WARN')
        lines.append(f'  [{tag}] {r.name}: {r.message}')
    return '\n'.join(lines)
