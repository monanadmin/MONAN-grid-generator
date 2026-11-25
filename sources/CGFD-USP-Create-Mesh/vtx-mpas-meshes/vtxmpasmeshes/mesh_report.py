"""
Per-mesh plot and report.

write_report() makes, for one generated mesh:
  <name>.png          the mesh, cells coloured by their size, with the
                      refinement, cut and boundary-ring outlines
  <name>.report.html  one self-contained page (images embedded): inputs,
                      the mesh plot, every check result with what it means
                      and its references, drawings of what the checks test,
                      and distributions of the mesh variables

p (the generation parameters) may be None, for a mesh made by another tool:
then the parts that compare with the requested resolution map (outlines,
actual/requested size, input table) are left out and the maps are centred on
the finest cells.
"""
import base64
import datetime
import html
import io
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from vtxmpasmeshes import mesh_checks as mc
from vtxmpasmeshes import mesh_geometry as geo
from vtxmpasmeshes import mesh_report_figures as schematic
from vtxmpasmeshes.mesh_config import ring_resolution

NAVY, GREEN, ORANGE, GREY, LIGHT = '#14213D', '#1B6F57', '#C8601A', '#8A94A6', '#E8EDEE'
LAYER_COLOURS = {0: '#E8EDEE', 1: '#9ED9C3', 2: '#6CC3A3', 3: '#3FA684', 4: '#2A8A6C',
                 5: '#1B6F57', 6: '#F0A56B', 7: '#C8601A'}

# What each check tests and why (shown in the report). Keys = CheckResult.name.
EXPLAIN = {
    'finite values': ('MPAS structure', 'No NaN or infinity in any field.',
                      'A corrupted geometry field silently breaks every operator.', ''),
    'points on the sphere': ('MPAS structure', 'Cell, edge and vertex positions lie on the sphere.',
                             'Arc lengths and areas assume points on a sphere of radius sphere_radius.', '4'),
    'lat/lon match x,y,z': ('MPAS structure', 'Latitude/longitude agree with the Cartesian positions.',
                            'Latitude sets the Coriolis parameter and physics; positions set the geometry.', '4'),
    'connectivity': ('MPAS structure', 'Cells, edges and vertices refer to each other consistently, in MPAS order.',
                     'Divergence, gradient, curl and the TRiSK reconstruction are sums over these index arrays.', '2, 4'),
    'topology (V - E + F)': ('MPAS structure', 'Euler characteristic: 2 for a sphere, 1 for one regional domain.',
                             'Detects holes or detached pieces.', ''),
    'positive lengths and areas': ('MPAS structure', 'dcEdge, dvEdge, areaCell, areaTriangle > 0.',
                                   'Operators divide by these quantities.', '2'),
    'dvEdge/dcEdge': ('MPAS structure', 'No Voronoi edge is tiny compared with its cell-to-cell distance.',
                      'Near-degenerate edges degrade operator accuracy and stability. The 0.12 limit is a '
                      'heuristic, not taken from a paper.', ''),
    'Voronoi property': ('MPAS structure', 'Vertices are circumcentres; edges are orthogonal to the cell-to-cell '
                         'line; dcEdge and dvEdge are the true arcs.',
                         'The C-grid and TRiSK (energy conservation, stationary geostrophic modes) are derived for '
                         'orthogonal Voronoi/Delaunay pairs.', '2, 3, 5'),
    'areas': ('MPAS structure', 'Kite areas add up to triangle and cell areas; a global mesh covers 4πR².',
              'TRiSK maps quantities between cells and vertices with these kites; conservation relies on them.',
              '2, 4'),
    'near-centroidal cells': ('MPAS structure', 'Each generator is close to its cell centroid.',
                              'MPAS meshes are meant to be centroidal Voronoi tessellations. The 0.15 limit is '
                              'empirical.', '1, 6, 7'),
    'smooth grading': ('MPAS structure', 'Size ratio of neighbouring cells.',
                       'MPAS does not require smooth grading; steep transitions are valid but lower local '
                       'accuracy and can affect waves and scale-dependent physics, so check the fields near '
                       'them. The 1.5 limit is empirical.', '1, 8, 9'),
    'boundary layers': ('Limited area', 'bdyMaskCell has all 7 nested layers; missing neighbours only in layer 7.',
                        'Regional MPAS-A uses 5 relaxation + 2 specified layers.', '10, 11'),
    'graph.info': ('Limited area', 'The graph file matches the cell neighbours.',
                   'MPAS splits the domain across MPI tasks with METIS from this graph.', '12'),
    'partition file': ('Limited area', 'Every cell belongs to one of the N partitions.',
                       'Each cell must belong to exactly one MPI task.', '12'),
    'cell size follows the requested map': ('Generator', 'Cell spacing vs the requested size.',
                                            'JIGSAW’s size function is the target cell-centre spacing.', '5'),
    'boundary layers at uniform resolution': ('Generator', 'Boundary-layer cells lie in the constant-size ring '
                                              'beyond the cut.',
                                              'Design choice of this generator: the lateral boundary zone sits in '
                                              'uniform cells.', '10'),
    'cut region kept': ('Generator', 'The interior covers the requested domain.',
                        'The domain asked for is fully inside the model interior.', ''),
}

REFERENCES = [
    'Skamarock, W. C., et al. (2012). A multiscale nonhydrostatic atmospheric model using centroidal Voronoi '
    'tesselations and C-grid staggering. Mon. Wea. Rev., 140, 3090–3105. doi:10.1175/MWR-D-11-00215.1',
    'Ringler, T. D., J. Thuburn, J. B. Klemp and W. C. Skamarock (2010). A unified approach to energy conservation '
    'and potential vorticity dynamics for arbitrarily-structured C-grids. J. Comput. Phys., 229, 3065–3090. '
    'doi:10.1016/j.jcp.2009.12.007',
    'Thuburn, J., T. D. Ringler, W. C. Skamarock and J. B. Klemp (2009). Numerical representation of geostrophic '
    'modes on arbitrarily structured C-grids. J. Comput. Phys., 228, 8321–8335. doi:10.1016/j.jcp.2009.08.006',
    'MPAS Mesh Specification, version 1.0 (2015), NCAR/MMM. '
    'https://www2.mmm.ucar.edu/projects/mpas/mpas_website_linked_files/mpas_meshspec.pdf',
    'Engwirda, D. (2017). JIGSAW-GEO (1.0). Geosci. Model Dev., 10, 2117–2140. doi:10.5194/gmd-10-2117-2017',
    'Ringler, T., L. Ju and M. Gunzburger (2008). A multiresolution method for climate system modeling: application '
    'of spherical centroidal Voronoi tessellations. Ocean Dynamics, 58, 475–498. doi:10.1007/s10236-008-0157-2',
    'Du, Q., V. Faber and M. Gunzburger (1999). Centroidal Voronoi tessellations: applications and algorithms. '
    'SIAM Review, 41, 637–676. doi:10.1137/S0036144599352836',
    'Ringler, T., et al. (2011). Exploring a multiresolution modeling approach within the shallow-water equations. '
    'Mon. Wea. Rev., 139, 3348–3368. doi:10.1175/MWR-D-10-05049.1',
    'Fowler, L. D., et al. (2016). Analyzing the Grell–Freitas convection scheme from hydrostatic to '
    'nonhydrostatic scales within a global model. Mon. Wea. Rev., 144, 2285–2306. doi:10.1175/MWR-D-15-0311.1',
    'Skamarock, W. C., M. G. Duda, S. Ha and S.-H. Park (2018). Limited-area atmospheric modeling using an '
    'unstructured mesh. Mon. Wea. Rev., 146, 3445–3460. doi:10.1175/MWR-D-18-0155.1',
    'MPAS-Limited-Area tool: https://github.com/MiCurry/MPAS-Limited-Area',
    'Karypis, G. and V. Kumar (1998). A fast and high quality multilevel scheme for partitioning irregular graphs. '
    'SIAM J. Sci. Comput., 20, 359–392 (METIS).',
]


# ---------------------------------------------------------------------------
def _png(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


def _b64(png_bytes):
    return 'data:image/png;base64,' + base64.b64encode(png_bytes).decode()


def _unwrap(lon, lon0):
    return lon0 + geo.wrap_lon(np.asarray(lon) - lon0)


def _centre(m, p):
    """(lat, lon) the maps are centred on: the requested centre, or else the finest cells."""
    if p is not None:
        return p['lat'], p['lon']
    size = mc.cell_size_km(m)
    if size.max() > 1.05 * size.min():
        fine = size <= np.percentile(size, 1)
    else:                                   # uniform: centre of the whole mesh
        fine = np.ones(m.nCells, bool)
    v = m.xc[fine].sum(0)
    if np.linalg.norm(v) < 1e-6 * m.R * fine.sum():   # uniform global mesh: any centre will do
        return 0.0, 0.0
    v = v / np.linalg.norm(v)
    return (float(np.degrees(np.arcsin(np.clip(v[2], -1, 1)))),
            float(np.degrees(np.arctan2(v[1], v[0]))))


def _fine_extent(m, lon0):
    """Map extent around the finer cells (None if the mesh is nearly uniform)."""
    size = mc.cell_size_km(m)
    if size.max() < 1.3 * size.min():
        return None
    sel = size <= size.min() + 0.3 * (size.max() - size.min())
    lat = m.lat[sel]
    lon = _unwrap(m.lon[sel], lon0)
    pad = 3
    return (lon.min() - pad, lon.max() + pad, max(lat.min() - pad, -90), min(lat.max() + pad, 90))


def _cell_polygons(m, lon0):
    """Cell polygons in (lon, lat) degrees, longitudes continuous around lon0."""
    latv = np.degrees(m.v['latVertex'])
    lonv = _unwrap(np.degrees(m.v['lonVertex']), lon0)
    polys, keep = [], []
    for i in range(m.nCells):
        vs = m.voc[i, :m.nEdgesOnCell[i]]
        lo, la = lonv[vs], latv[vs]
        if lo.max() - lo.min() > 180:          # cell split by the far-side seam
            continue
        polys.append(np.column_stack([lo, la]))
        keep.append(i)
    return polys, np.array(keep, dtype=int)


def _outlines(p):
    """Refinement, cut and boundary-ring outlines as lists of (label, lat, lon, style)."""
    out, t = [], np.linspace(0, 2 * np.pi, 361)
    r = p['refinement']
    if r == 'circle':
        for rad, lab in ((p['inner_radius'], 'fine circle'), (p['outer_radius'], 'transition ends')):
            la, lo = geo.destination(p['lat'], p['lon'], rad, np.degrees(t))
            out.append((lab, la, lo, dict(color=GREEN, lw=1.5, ls='-' if lab == 'fine circle' else '--')))
    elif r == 'ellipse':
        for s, lab in ((1.0, 'fine ellipse'), (1 + p['delta'], 'transition ends')):
            la, lo = geo.local_xy_to_latlon(s * p['a_km'] * np.cos(t), s * p['b_km'] * np.sin(t),
                                            p['lat'], p['lon'], p['angle'])
            out.append((lab, la, lo, dict(color=GREEN, lw=1.5, ls='-' if s == 1 else '--')))
    if p['cut'] != 'global':
        la, lo = geo.cut_boundary(p, n=361)
        out.append(('cut', la, lo, dict(color=ORANGE, lw=2, ls='-')))
        buf = p['n_layers'] * ring_resolution(p)
        if p['cut'] == 'circle':
            la, lo = geo.destination(p['lat'], p['lon'], p['cut_radius'] + buf, np.degrees(t))
        else:
            la, lo = geo.local_xy_to_latlon((p['cut_a_km'] + buf) * np.cos(t), (p['cut_b_km'] + buf) * np.sin(t),
                                            p['lat'], p['lon'], p['cut_angle'])
        out.append(('end of uniform ring', la, lo, dict(color=GREY, lw=1, ls=':')))
    return out


def _coastlines(ax, extent_deg):
    """Add coastlines if cartopy can provide them (needs Natural Earth data or internet)."""
    try:
        import cartopy.feature as cfeature
        res = '50m' if extent_deg < 60 else '110m'
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', res),
                       facecolor='none', edgecolor='k', linewidth=0.5, zorder=3)
        return True
    except Exception:
        return False


def plot_mesh(m, p, values, label, extent=None, cmap='viridis_r', vmin=None, vmax=None, title=None,
              outlines=True):
    """Map of the cells coloured by `values`. extent=(lon_min, lon_max, lat_min, lat_max) or None."""
    import cartopy.crs as ccrs
    lon0 = _centre(m, p)[1]
    polys, keep = _cell_polygons(m, lon0)
    fig = plt.figure(figsize=(10, 6.2))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=lon0))
    data = ccrs.PlateCarree()
    pc = PolyCollection(polys, array=np.asarray(values)[keep], cmap=cmap, transform=data,
                        edgecolors='face' if m.nCells > 20000 else (0, 0, 0, 0.25),
                        linewidths=0.15, rasterized=True)
    pc.set_clim(vmin if vmin is not None else np.nanmin(values), vmax if vmax is not None else np.nanmax(values))
    ax.add_collection(pc)
    if extent is None:
        ax.set_global()
        span = 360
    else:
        ax.set_extent(extent, crs=data)
        span = max(extent[1] - extent[0], extent[3] - extent[2])
    coast = _coastlines(ax, span)
    if outlines and p is not None:
        for lab, la, lo, st in _outlines(p):
            ax.plot(_unwrap(lo, lon0), la, transform=data, zorder=4, label=lab, **st)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='lower left', fontsize=8, framealpha=0.85)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='grey', alpha=0.6)
    gl.top_labels = gl.right_labels = False
    cb = fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(label)
    ax.set_title(title or '')
    return fig, coast


def _regional_extent(m, p):
    lat = np.degrees(m.v['latCell'])
    lon = _unwrap(np.degrees(m.v['lonCell']), _centre(m, p)[1])
    pad = 1.5
    return (lon.min() - pad, lon.max() + pad, max(lat.min() - pad, -90), min(lat.max() + pad, 90))


def _refinement_extent(p):
    if p['refinement'] == 'uniform':
        return None
    la, lo = geo.refinement_outer_contour(p, n=180)
    lo = _unwrap(lo, p['lon'])
    pad = 3
    return (lo.min() - pad, lo.max() + pad, max(la.min() - pad, -90), min(la.max() + pad, 90))


# ---------------------------------------------------------------------------
def _fig_boundary_layers(m, p):
    mask = m.v['bdyMaskCell'].astype(int)
    lat0, lon0 = _centre(m, p)
    latv = np.degrees(m.v['latVertex'])
    lonv = np.degrees(m.v['lonVertex'])
    polys = []
    for i in range(m.nCells):
        vs = m.voc[i, :m.nEdgesOnCell[i]]
        x, y = geo.local_xy(latv[vs], lonv[vs], lat0, lon0, 0.0)
        polys.append(np.column_stack([x, y]))
    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    ax.add_collection(PolyCollection(polys, facecolors=[LAYER_COLOURS[int(k)] for k in mask],
                                     edgecolors='white', linewidths=0.3))
    if p is not None:
        blat, blon = geo.cut_boundary(p, n=361)
        bx, by = geo.local_xy(blat, blon, lat0, lon0, 0.0)
        ax.plot(bx, by, color=NAVY, lw=2, ls='--', label='cut line')
    allxy = np.concatenate(polys)
    lim = np.abs(allxy).max() * 1.03
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    handles = [Patch(fc=LAYER_COLOURS[0], label='0  interior')] + \
              [Patch(fc=LAYER_COLOURS[k], label=f'{k}  relaxation zone') for k in range(1, 6)] + \
              [Patch(fc=LAYER_COLOURS[k], label=f'{k}  specified zone') for k in (6, 7)] + \
              ([plt.Line2D([], [], color=NAVY, ls='--', label='cut line')] if p is not None else [])
    ax.legend(handles=handles, title='bdyMaskCell', loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.set_xlabel('km east of centre'); ax.set_ylabel('km north of centre')
    ax.set_title('Boundary layers of this mesh')
    return _png(fig)


def _hist(ax, data, title, xlabel, lines=(), bins=60, logy=False, colour=GREEN):
    data = np.asarray(data)[np.isfinite(data)]
    ax.hist(data, bins=bins, color=colour, alpha=0.85)
    for x, lab, c in lines:
        ax.axvline(x, color=c, lw=1.5, ls='--')
        ax.text(x, ax.get_ylim()[1] * 0.97, ' ' + lab, color=c, fontsize=8, va='top')
    if logy:
        ax.set_yscale('log')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('cells' if 'edge' not in xlabel.lower() else 'edges')


def _distributions(m, p, part_file, nparts):
    """List of (title, png bytes, caption)."""
    figs = []
    size = mc.cell_size_km(m)
    target = mc.target_km(m, p) if p is not None else None
    lat0, lon0 = _centre(m, p)
    dist = geo.haversine_km(m.lat, m.lon, lat0, lon0)
    regional = m.regional
    mask = m.v['bdyMaskCell'].astype(int) if regional else np.zeros(m.nCells, int)

    if target is not None:
        ratio = size / target
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
        lo, hi = np.nanmin([size.min(), target.min()]), np.nanmax([size.max(), target.max()])
        bins = np.linspace(lo, hi, 60)
        axs[0].hist(target, bins=bins, color=GREY, alpha=0.6, label='requested')
        axs[0].hist(size, bins=bins, color=GREEN, alpha=0.6, label='actual')
        axs[0].set_xlabel('cell size (km)'); axs[0].set_ylabel('cells'); axs[0].legend()
        axs[0].set_title('Cell size: actual vs requested', fontsize=11)
        _hist(axs[1], ratio, 'Actual / requested cell size', 'ratio',
              lines=[(1 - mc.SIZE_TOL, f'-{mc.SIZE_TOL:.0%}', ORANGE),
                     (1 + mc.SIZE_TOL, f'+{mc.SIZE_TOL:.0%}', ORANGE)])
        fig.tight_layout()
        figs.append(('Cell size', _png(fig), 'Cell size = mean distance to the neighbouring cell centres. '
                     'The ratio should cluster around 1; dashed lines are the ±25% tolerance per cell.'))
    else:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        _hist(ax, size, 'Cell size', 'cell size (km)')
        fig.tight_layout()
        figs.append(('Cell size', _png(fig), 'Cell size = mean distance to the neighbouring cell centres '
                     '(no requested resolution map to compare with).'))

    fig, ax = plt.subplots(figsize=(12, 4.6))
    if regional:
        for k in range(8):
            sel = mask == k
            ax.scatter(dist[sel], size[sel], s=4, color=LAYER_COLOURS[k] if k else GREEN,
                       label='interior' if k == 0 else (f'layer {k}' if k in (1, 7) else None))
    else:
        ax.scatter(dist, size, s=3, color=GREEN, label='cells')
    if target is not None:
        o = np.argsort(dist)
        ax.plot(dist[o], target[o], 'k.', ms=1, label='requested')
    ax.set_xlabel('distance from the centre (km)' if p is not None else
                  f'distance from the finest cells ({lat0:.1f}°, {lon0:.1f}°) (km)'); ax.set_ylabel('cell size (km)')
    if not regional:
        ax.set_xscale('symlog', linthresh=1000)
    ax.legend(markerscale=3, fontsize=9)
    ax.set_title('Cell size against distance from the centre', fontsize=11)
    fig.tight_layout()
    figs.append(('Cell size profile', _png(fig), 'Each dot is a cell; black dots are the requested size at that '
                 'cell. For an ellipse the transition appears as a band, because distance from the centre is not '
                 'its natural coordinate.'))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.0))
    _hist(axs[0], m.v['dvEdge'] / m.v['dcEdge'], 'dvEdge / dcEdge', 'ratio (edges)',
          lines=[(mc.MIN_DVDC, f'{mc.MIN_DVDC} limit', ORANGE)])
    _hist(axs[1], mc.cvt_offset(m), 'Generator–centroid offset', 'offset / √area',
          lines=[(mc.MAX_CVT_OFFSET, f'{mc.MAX_CVT_OFFSET} limit', ORANGE)], logy=True)
    _hist(axs[2], mc.grading(m), 'Neighbour size ratio', 'ratio',
          lines=[(mc.MAX_GRADING, f'{mc.MAX_GRADING} limit', ORANGE)], logy=True)
    fig.tight_layout()
    figs.append(('Mesh quality', _png(fig), 'Values beyond the dashed lines fail (dvEdge/dcEdge) or warn '
                 '(centroid offset, neighbour size ratio) in the checks.'))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.0))
    sides, counts = np.unique(m.nEdgesOnCell, return_counts=True)
    axs[0].bar([str(s) for s in sides], counts, color=GREEN)
    for s, c in zip(sides, counts):
        axs[0].text(str(s), c, f'{c}', ha='center', va='bottom', fontsize=8)
    axs[0].set_title('Cell shapes', fontsize=11); axs[0].set_xlabel('number of sides'); axs[0].set_ylabel('cells')
    _hist(axs[1], m.v['areaCell'] * m.km ** 2, 'Cell area', 'area (km²)')
    if regional:
        ks, cs = np.unique(mask, return_counts=True)
        axs[2].bar([str(k) for k in ks], cs, color=[LAYER_COLOURS[int(k)] for k in ks])
        axs[2].set_title('Cells per boundary layer', fontsize=11); axs[2].set_xlabel('bdyMaskCell')
        axs[2].set_yscale('log'); axs[2].set_ylabel('cells')
    elif part_file:
        axs[2].axis('off')
    else:
        axs[2].axis('off')
    fig.tight_layout()
    figs.append(('Cells', _png(fig), 'Most cells are hexagons; pentagons and heptagons are normal where the '
                 'resolution changes and (globally) at icosahedral points.'))

    if part_file and nparts > 1 and os.path.exists(part_file):
        parts = np.loadtxt(part_file, dtype=int, ndmin=1)
        fig, ax = plt.subplots(figsize=(10, 3.6))
        cnt = np.bincount(parts, minlength=nparts)
        ax.bar(np.arange(nparts), cnt, color=GREEN)
        ax.axhline(cnt.mean(), color=ORANGE, ls='--', lw=1)
        ax.set_xlabel('MPI task'); ax.set_ylabel('cells'); ax.set_title('Cells per MPI task', fontsize=11)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        fig.tight_layout()
        figs.append(('Partition', _png(fig), f'{nparts} partitions; largest/mean = {cnt.max() / cnt.mean():.2f} '
                     '(1.00 = perfectly balanced).'))
    return figs


# ---------------------------------------------------------------------------
CSS = """
body{font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;color:#1a1a1a;max-width:1100px;margin:24px auto;
padding:0 16px;line-height:1.45;background:#fff}
h1{color:#14213D;font-size:24px;margin-bottom:4px}h2{color:#14213D;font-size:19px;margin-top:32px;
border-bottom:1px solid #e3e7eb;padding-bottom:4px}h3{color:#14213D;font-size:15px}
.sub{color:#4A5568}.badge{display:inline-block;padding:4px 10px;border-radius:12px;font-weight:600;color:#fff}
.ok{background:#1B6F57}.warn{background:#B5541A}.fail{background:#9B1C1C}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border:1px solid #d5dbe1;
padding:5px 7px;text-align:left;vertical-align:top}th{background:#eef1f3}
td.r-ok{color:#1B6F57;font-weight:600}td.r-warn{color:#B5541A;font-weight:600}td.r-fail{color:#9B1C1C;font-weight:600}
img{max-width:100%;height:auto;display:block;margin:8px auto}figure{margin:12px 0}
figcaption{font-size:13px;color:#4A5568;font-style:italic}code{background:#f1f3f5;padding:1px 4px;border-radius:3px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px}
.stat{border:1px solid #d5dbe1;border-radius:8px;padding:8px 10px}.stat b{display:block;font-size:18px}
.stat small{color:#4A5568}ol.refs{font-size:12.5px}
"""


OUTLINE_CAPTION = (' Green: refinement outlines; orange: cut; grey dotted: end of the uniform ring '
                   'that holds the boundary layers.')


def _status(results):
    fails = [r for r in results if not r.ok and r.severity == 'error']
    warns = [r for r in results if not r.ok and r.severity != 'error']
    if fails:
        return 'fail', f'{len(fails)} check(s) failed'
    if warns:
        return 'warn', f'all required checks passed, {len(warns)} warning(s)'
    return 'ok', f'all {len(results)} checks passed'


def write_report(grid_file, p, results, out_dir, name, graph_file=None, part_file=None,
                 durations=None, nparts=None, made_by='create_regional_mesh.py'):
    """Write <name>.png and <name>.report.html into out_dir. Returns their paths.

    p may be None (mesh not made by this generator); nparts defaults to p['N']."""
    m = mc.Mesh(grid_file)
    lat0, lon0 = _centre(m, p)
    if nparts is None:
        nparts = p.get('N', 1) if p is not None else 1
    size = mc.cell_size_km(m)
    e = html.escape

    # Mesh plot(s)
    smin, smax = np.percentile(size, 0.5), np.percentile(size, 99.5)
    if m.regional:
        fig, coast = plot_mesh(m, p, size, 'cell size (km)', extent=_regional_extent(m, p), vmin=smin, vmax=smax,
                               title=f'{m.nCells:,} cells')
        main_png = _png(fig, dpi=150)
        extra = []
    else:
        fig, coast = plot_mesh(m, p, size, 'cell size (km)', extent=None, vmin=smin, vmax=smax,
                               title=f'{m.nCells:,} cells')
        main_png = _png(fig, dpi=150)
        extra = []
        ext = _refinement_extent(p) if p is not None else _fine_extent(m, lon0)
        if ext is not None:
            fig, _ = plot_mesh(m, p, size, 'cell size (km)', extent=ext, vmin=smin,
                               vmax=np.percentile(size[(geo.haversine_km(m.lat, m.lon, lat0, lon0)
                                                        < 3000)], 99.5),
                               title='Zoom on the refinement')
            extra.append(('Zoom on the refinement', _png(fig, dpi=150)))
    png_path = os.path.join(out_dir, f'{name}.png')
    with open(png_path, 'wb') as f:
        f.write(main_png)

    ratio_png = None
    if p is not None:
        ratio = size / mc.target_km(m, p)
        fig, _ = plot_mesh(m, p, ratio, 'actual / requested cell size', cmap='RdBu_r', vmin=0.75, vmax=1.25,
                           extent=_regional_extent(m, p) if m.regional else _refinement_extent(p),
                           title='Where the cell size departs from the request')
        ratio_png = _png(fig, dpi=130)

    # Schematics (generic) and this mesh's boundary layers
    sch = []
    for fn, cap in ((schematic.fig_connectivity,
                     'Connectivity and the Voronoi property: index arrays listed counter-clockwise; each edge is '
                     'perpendicular to the line joining its two cells; each vertex is the circumcentre of its 3 '
                     'cells.'),
                    (schematic.fig_areas,
                     'Areas: each triangle is split into 3 kites, one per cell; the kites must add up to the '
                     'triangle and to each cell.'),
                    (schematic.fig_quality,
                     'Quality checks: a near-degenerate edge (error), a generator off its centroid (warning), an '
                     'abrupt jump in cell size (warning).')):
        buf = io.BytesIO()
        fn(buf, dpi=110)
        plt.close('all')
        sch.append((buf.getvalue(), cap))
    if m.regional:
        sch.append((_fig_boundary_layers(m, p),
                    'Boundary layers of this mesh: interior (0), relaxation layers 1–5, specified layers 6–7, '
                    'all beyond the cut line.'))

    dists = _distributions(m, p, part_file, nparts)

    # ---- HTML
    cls, text = _status(results)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    rows = []
    for r in results:
        grp, what, why, refs = EXPLAIN.get(r.name, ('', '', '', ''))
        res = 'OK' if r.ok else ('FAIL' if r.severity == 'error' else 'WARN')
        rc = 'r-ok' if r.ok else ('r-fail' if r.severity == 'error' else 'r-warn')
        rows.append(f'<tr><td>{e(grp)}</td><td>{e(r.name)}</td><td class="{rc}">{res}</td>'
                    f'<td>{e(r.message)}</td><td>{e(what)}</td><td>{e(why)}</td><td>{e(refs)}</td></tr>')
    if p is not None:
        params = ('<table><tr><th>Parameter</th><th>Value</th></tr>'
                  + ''.join(f'<tr><td><code>{e(str(k))}</code></td><td>{e(str(v))}</td></tr>'
                            for k, v in p.items()) + '</table>')
    else:
        params = ('<p class="sub">No generation parameters (mesh not made by this generator, or no input '
                  'file given): the comparison with a requested resolution map is left out.</p>')
    files = [os.path.basename(grid_file), os.path.basename(graph_file or '')]
    if part_file:
        files.append(os.path.basename(part_file))
    dur = ''
    if durations:
        dur = ''.join(f'<div class="stat"><b>{v:.1f} s</b><small>{e(k)}</small></div>'
                      for k, v in durations.items())
    stats = (f'<div class="stat"><b>{m.nCells:,}</b><small>cells</small></div>'
             f'<div class="stat"><b>{m.nEdges:,}</b><small>edges</small></div>'
             f'<div class="stat"><b>{m.nVertices:,}</b><small>vertices</small></div>'
             f'<div class="stat"><b>{size.min():.3g}–{size.max():.3g} km</b><small>cell size range</small></div>'
             + dur)
    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
             f'content="width=device-width, initial-scale=1"><title>Mesh report: {e(name)}</title>'
             f'<style>{CSS}</style></head><body>',
             f'<h1>Mesh report</h1><p class="sub"><code>{e(name)}</code><br>Made {now} by '
             f'{e(made_by)}</p>',
             f'<p><span class="badge {cls}">{e(text)}</span></p>',
             f'<div class="grid">{stats}</div>',
             '<h2>Inputs</h2>' + params,
             '<p class="sub">Files: ' + ', '.join(f'<code>{e(f)}</code>' for f in files if f) + '</p>',
             '<h2>Mesh</h2>',
             f'<figure><img src="{_b64(main_png)}" alt="Map of the mesh coloured by cell size">'
             f'<figcaption>Cells coloured by size (mean distance to neighbouring cell centres).'
             f'{OUTLINE_CAPTION if p is not None else ""}'
             f'{"" if coast else " (Coastlines unavailable: cartopy could not load Natural Earth data.)"}'
             f'</figcaption></figure>']
    for title, png in extra:
        parts.append(f'<figure><img src="{_b64(png)}" alt="{e(title)}"><figcaption>{e(title)}</figcaption></figure>')
    if ratio_png is not None:
        parts.append(f'<figure><img src="{_b64(ratio_png)}" alt="Actual over requested cell size">'
                     '<figcaption>Actual / requested cell size: white ≈ as requested; red = larger, '
                     'blue = smaller.</figcaption></figure>')
    parts.append('<h2>Checks</h2><table><tr><th>Group</th><th>Check</th><th>Result</th><th>Measured</th>'
                 '<th>What it tests</th><th>Why</th><th>Refs</th></tr>' + ''.join(rows) + '</table>'
                 '<p class="sub">FAIL = the mesh must not be used (the run stops before partitioning). '
                 'WARN = valid mesh that deviates from what is expected; look at the related distribution below.'
                 '</p>')
    parts.append('<h2>What the checks test</h2>')
    for png, cap in sch:
        parts.append(f'<figure><img src="{_b64(png)}" alt="{e(cap)}"><figcaption>{e(cap)}</figcaption></figure>')
    parts.append('<h2>Distributions</h2>')
    for title, png, cap in dists:
        parts.append(f'<h3>{e(title)}</h3><figure><img src="{_b64(png)}" alt="{e(title)}">'
                     f'<figcaption>{e(cap)}</figcaption></figure>')
    parts.append('<h2>References</h2><ol class="refs">' + ''.join(f'<li>{e(r)}</li>' for r in REFERENCES)
                 + '</ol></body></html>')
    html_path = os.path.join(out_dir, f'{name}.report.html')
    with open(html_path, 'w') as f:
        f.write('\n'.join(parts))
    return png_path, html_path
