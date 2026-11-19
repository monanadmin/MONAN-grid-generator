"""
Geometry shared by the resolution map, the regional cut and the mesh checks.

Conventions
-----------
- Great-circle distances use the MPAS Earth radius (6371.229 km), the same
  radius MPAS-Limited-Area uses for circular cuts.
- Ellipses (refinement and cut) are defined in a local tangent plane at the
  mesh center: x = R cos(lat_c) dlon (east), y = R dlat (north), R = 6371 km,
  rotated by `angle` degrees counter-clockwise from east. The same mapping is
  used to build the elliptical cut, so the cut follows exactly the contour the
  resolution map uses.
- All distances in km, angles in degrees.
"""

import numpy as np

EARTH_RADIUS_KM = 6371.229        # MPAS / MPAS-Limited-Area
LOCAL_PLANE_RADIUS_KM = 6371.0    # local plane of the ellipse definitions
FAR_FIELD_SLOPE = 0.05            # km of cell size per km beyond the ring
FAR_FIELD_MAX_RES = 1000.0        # coarsest far-field cell size (km)


def haversine_km(lat, lon, lat_ref, lon_ref):
    """Great-circle distance (km) from (lat_ref, lon_ref) to (lat, lon)."""
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat_ref), np.radians(lon_ref)
    a = (np.sin((lat - lat0) / 2.0) ** 2
         + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def destination(lat_ref, lon_ref, distance_km, bearing_deg):
    """Points at a great-circle distance and bearing (from north) of a center."""
    d = np.asarray(distance_km) / EARTH_RADIUS_KM
    b = np.radians(bearing_deg)
    lat0, lon0 = np.radians(lat_ref), np.radians(lon_ref)
    lat = np.arcsin(np.sin(lat0) * np.cos(d) + np.cos(lat0) * np.sin(d) * np.cos(b))
    lon = lon0 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(lat0),
                            np.cos(d) - np.sin(lat0) * np.sin(lat))
    return np.degrees(lat), wrap_lon(np.degrees(lon))


def wrap_lon(lon):
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0


def local_xy(lat, lon, lat_ref, lon_ref, angle_deg):
    """Rotated local-plane coordinates (km) of (lat, lon)."""
    dlon = np.radians(wrap_lon(np.asarray(lon) - lon_ref))
    dx = LOCAL_PLANE_RADIUS_KM * np.cos(np.radians(lat_ref)) * dlon
    dy = LOCAL_PLANE_RADIUS_KM * np.radians(np.asarray(lat) - lat_ref)
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    return c * dx + s * dy, -s * dx + c * dy


def local_xy_to_latlon(x, y, lat_ref, lon_ref, angle_deg):
    """Inverse of local_xy."""
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    dx = c * x - s * y
    dy = s * x + c * y
    lat = lat_ref + np.degrees(dy / LOCAL_PLANE_RADIUS_KM)
    lon = lon_ref + np.degrees(dx / (LOCAL_PLANE_RADIUS_KM * np.cos(np.radians(lat_ref))))
    return lat, wrap_lon(lon)


def elliptical_radius(lat, lon, lat_ref, lon_ref, a_km, b_km, angle_deg):
    """Normalised elliptical radius phi (1 on the ellipse) and plane distance r."""
    x, y = local_xy(lat, lon, lat_ref, lon_ref, angle_deg)
    return np.hypot(x / a_km, y / b_km), np.hypot(x, y)


# --------------------------------------------------------------------------
# Refinement: target cell size before the cut-dependent far field
# --------------------------------------------------------------------------
def refinement_resolution(lat, lon, p):
    """Target cell size (km) of the refinement at points (lat, lon)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    r = p['refinement']
    if r == 'uniform':
        return np.full(np.broadcast(lat, lon).shape, float(p['high_res']))
    if r == 'circle':
        d = haversine_km(lat, lon, p['lat'], p['lon'])
        return np.interp(d, [0.0, p['inner_radius'], p['outer_radius']],
                         [p['high_res'], p['high_res'], p['low_res']])
    if r == 'ellipse':
        phi, _ = elliptical_radius(lat, lon, p['lat'], p['lon'],
                                   p['a_km'], p['b_km'], p['angle'])
        t = np.clip((phi - 1.0) / p['delta'], 0.0, 1.0)
        return p['high_res'] + (p['low_res'] - p['high_res']) * t
    raise ValueError(f'unknown refinement {r!r}')


def refinement_outer_contour(p, n=720):
    """
    Points where the refinement reaches low_res (end of the transition).
    None for uniform refinement (it has no transition).
    """
    r = p['refinement']
    if r == 'uniform':
        return None
    if r == 'circle':
        return destination(p['lat'], p['lon'], p['outer_radius'],
                           np.linspace(0.0, 360.0, n, endpoint=False))
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    s = 1.0 + p['delta']
    return local_xy_to_latlon(s * p['a_km'] * np.cos(t), s * p['b_km'] * np.sin(t),
                              p['lat'], p['lon'], p['angle'])


# --------------------------------------------------------------------------
# Cut
# --------------------------------------------------------------------------
def _ellipse_perimeter(a, b):
    h = ((a - b) / (a + b)) ** 2
    return np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))


def distance_beyond_cut(lat, lon, p, sample_km=None):
    """
    Shortest distance (km) to the cut boundary: negative inside, positive outside.
    circle : great-circle distance minus cut_radius
    ellipse: distance in the local plane to the cut ellipse, from a KD-tree
             over boundary points spaced sample_km apart (default: 1/20 of
             the ring resolution or less), so the ring of boundary layers has
             the same width all around, also where the ellipse is elongated.
    """
    if p['cut'] == 'circle':
        return haversine_km(lat, lon, p['lat'], p['lon']) - p['cut_radius']
    if p['cut'] == 'ellipse':
        from scipy.spatial import cKDTree
        a, b = p['cut_a_km'], p['cut_b_km']
        if sample_km is None:
            ring = p['low_res'] if p['refinement'] != 'uniform' else p['high_res']
            sample_km = ring / 20.0
        n = int(max(1440, np.ceil(_ellipse_perimeter(a, b) / sample_km)))
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        tree = cKDTree(np.column_stack([a * np.cos(t), b * np.sin(t)]))
        x, y = local_xy(lat, lon, p['lat'], p['lon'], p['cut_angle'])
        x, y = np.broadcast_arrays(x, y)
        dist, _ = tree.query(np.column_stack([x.ravel(), y.ravel()]))
        inside = (np.hypot(x / a, y / b) < 1.0).ravel()
        return np.where(inside, -dist, dist).reshape(x.shape)
    raise ValueError(f"no cut boundary for cut={p['cut']!r}")


def cut_boundary(p, n=360):
    """Boundary of the cut as (lat, lon) arrays, counter-clockwise."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    if p['cut'] == 'circle':
        # bearings decreasing = counter-clockwise seen from above
        return destination(p['lat'], p['lon'], p['cut_radius'], -np.degrees(t))
    if p['cut'] == 'ellipse':
        return local_xy_to_latlon(p['cut_a_km'] * np.cos(t), p['cut_b_km'] * np.sin(t),
                                  p['lat'], p['lon'], p['cut_angle'])
    raise ValueError(f"no cut boundary for cut={p['cut']!r}")


def target_resolution(lat, lon, p, ring_res, n_layers):
    """
    Final target cell size (km): the refinement, plus for regional cuts a
    constant ring of n_layers * ring_res beyond the cut (where the MPAS
    boundary layers sit) and a far field coarsening up to FAR_FIELD_MAX_RES.
    """
    res = refinement_resolution(lat, lon, p)
    if p['cut'] == 'global':
        return res
    d = distance_beyond_cut(lat, lon, p)
    buffer = n_layers * ring_res
    far = np.minimum(ring_res + FAR_FIELD_SLOPE * (d - buffer),
                     max(FAR_FIELD_MAX_RES, ring_res))
    res = np.where(d > 0, ring_res, res)
    return np.where(d > buffer, far, res)


def _ceil(x):
    """Round up to whole km, ignoring round-off (600.0000001 -> 600)."""
    return float(np.ceil(x - 1e-6))


def check_cut_encloses_refinement(p, ring_res, rel_tol=1e-6):
    """
    The cut must lie entirely in the constant ring resolution, i.e. outside the
    refinement's transition zone, so the MPAS boundary layers see a uniform
    mesh. Returns None if fine, else an error message with the minimum valid
    cut size.
    """
    if p['cut'] == 'global':
        return None
    blat, blon = cut_boundary(p, n=1440)
    res = refinement_resolution(blat, blon, p)
    if np.all(np.abs(res - ring_res) <= rel_tol * ring_res):
        return None
    contour = refinement_outer_contour(p)
    if p['cut'] == 'circle':
        need = float(np.max(haversine_km(contour[0], contour[1], p['lat'], p['lon'])))
        hint = f'use cut_radius >= {_ceil(need):g} km'
    else:
        phi, _ = elliptical_radius(contour[0], contour[1], p['lat'], p['lon'],
                                   p['cut_a_km'], p['cut_b_km'], p['cut_angle'])
        s = float(np.max(phi))
        hint = (f'scale the cut ellipse by at least {s:.3f}: cut_a_km >= '
                f"{_ceil(s * p['cut_a_km']):g}, cut_b_km >= {_ceil(s * p['cut_b_km']):g}")
    return ('the cut passes through the refinement or its transition zone, so '
            'the MPAS boundary layers would not be at uniform resolution '
            f'({ring_res:g} km); {hint}')
