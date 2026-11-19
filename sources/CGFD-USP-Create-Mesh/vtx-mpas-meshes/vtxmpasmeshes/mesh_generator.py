import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import xarray as xr
import numpy as np
from scipy.interpolate import interp1d

from mpas_tools.mesh.creation.jigsaw_to_netcdf import jigsaw_to_netcdf
from mpas_tools.mesh.conversion import convert
from mpas_tools.io import write_netcdf

from vtxmpasmeshes.jigsaw_generator import jigsaw_gen_sph_grid
from vtxmpasmeshes.mpas_plots import view_resolution_map, \
    view_mpas_regional_mesh
from vtxmpasmeshes.dataset_utilities import distance_latlon_matrix
from vtxmpasmeshes import mesh_geometry as geo
from vtxmpasmeshes.mesh_config import ring_resolution

# sources/MPAS-Limited-Area, independent of the current working directory
PATH_LIMITED_AREA = str(Path(__file__).resolve().parents[3] / 'MPAS-Limited-Area')


def apply_resolution_at_distance(distances, ref_points, ref_resolutions):
    f = interp1d(ref_points, ref_resolutions, bounds_error=False,
                 fill_value='extrapolate')
    resolution = xr.apply_ufunc(f, distances)
    return resolution


def doughnut_variable_resolution(**kwargs):

    # Setting parameters to the defaults if not passed as arguments
    defaults = {'lowresolution': 25,
                'highresolution': 3,
                'size': 40,
                'margin': 100,
                'final_res_dist': 1000}

    for name, default in defaults.items():
        kwag = kwargs.get(name, None)
        if kwag is None:
            kwargs.update({name: default})

    # Inner Circle
    # ------------------------------
    # From distance 0 to <size> -> constant highresolution
    d0, r0 = 0., kwargs['highresolution']
    d1, r1 = kwargs['size'], kwargs['highresolution']

    # 1st Variable Resolution Ring
    # -------------------------------
    # From <size> to <radius> = <size>+<margin> -> linear increase
    # from highresolution to lowresolution
    kwargs['radius'] = kwargs['size'] + kwargs['margin']
    slope = (kwargs['lowresolution'] - kwargs['highresolution']) / kwargs['margin']
    d2, r2 = kwargs['radius'], kwargs['lowresolution']

    # Fixed Low Resolution ring
    # -------------------------------
    # From <radius> until <border>=<radius>+<buffer> -> constant lowresolution
    #  - where the <buffer>=10 * lowresolution
    kwargs['buffer'] = kwargs['num_boundary_layers'] * kwargs['lowresolution']
    kwargs['border'] = kwargs['radius'] + kwargs['buffer']
    d3, r3 = kwargs['border'], kwargs['lowresolution']

    # Crazy increase in cell size ring
    # --------------------------------
    # From <border> to <xmax> -> linear increase with the same <slope>
    # than before until we reach the reoslution <final_res_dist>
    #  - <xmax> can be computed
    slope_xmax = 0.05
    deltares = kwargs['final_res_dist'] - kwargs['lowresolution']
    xmax = kwargs['border'] + deltares / slope_xmax
    d4, r4 = xmax, kwargs['final_res_dist']
    d5, r5 = 2*xmax, kwargs['final_res_dist']

    # Those are the points I fix
    dists = np.array([d0, d1, d2, d3, d4, d5])
    resol = np.array([r0, r1, r2, r3, r4, r5])

    return dists, resol, kwargs

def doughnut_variable_resolution_global(**kwargs):

    # Setting parameters to the defaults if not passed as arguments
    defaults = {'lowresolution': 25,
                'highresolution': 3,
                'size': 40,
                'margin': 100,
                'final_res_dist': 1000}

    for name, default in defaults.items():
        kwag = kwargs.get(name, None)
        if kwag is None:
            kwargs.update({name: default})

    # Inner Circle
    # ------------------------------
    # From distance 0 to <size> -> constant highresolution
    d0, r0 = 0., kwargs['highresolution']
    d1, r1 = kwargs['size'], kwargs['highresolution']

    # 1st Variable Resolution Ring
    # -------------------------------
    # From <size> to <radius> = <size>+<margin> -> linear increase
    # from highresolution to lowresolution
    kwargs['radius'] = kwargs['size'] + kwargs['margin']
    slope = (kwargs['lowresolution'] - kwargs['highresolution']) / kwargs['margin']
    d2, r2 = kwargs['radius'], kwargs['lowresolution']

    # Fixed Low Resolution ring
    # -------------------------------
    # From <radius> until <border>=<radius>+<buffer> -> constant lowresolution
    #  - where the <buffer>=10 * lowresolution
    kwargs['buffer'] = kwargs['num_boundary_layers'] * kwargs['lowresolution']
    kwargs['border'] = kwargs['radius'] + kwargs['buffer']
    d3, r3 = kwargs['border'], kwargs['lowresolution']

    # Those are the points I fix
    dists = np.array([d0, d1, d2, d3])
    resol = np.array([r0, r1, r2, r3])

    return dists, resol, kwargs

def constant_resolution(**kwargs):

    # Setting parameters to the defaults if not passed as arguments
    defaults = {'lowresolution': 25,
                'highresolution': 3,
                'size': 40,
                'margin': 100,
                'final_res_dist': 1000}

    for name, default in defaults.items():
        kwag = kwargs.get(name, None)
        if kwag is None:
            kwargs.update({name: default})

    # Inner Circle
    # ------------------------------
    # From distance 0 to <size> -> constant highresolution
    d0, r0 = 0., kwargs['highresolution']
    d1, r1 = kwargs['size'], kwargs['highresolution']
    
    # Boundary layers around it
    # ------------------------------
    kwargs['radius'] = kwargs['size']
    kwargs['buffer'] = kwargs['num_boundary_layers'] * kwargs['highresolution']
    kwargs['border'] = kwargs['radius'] + kwargs['buffer']
    d2, r2 = kwargs['border'], kwargs['highresolution']

    # Increase in resolution until final_res_dist
    # ------------------------------
    slope_xmax = 0.05
    deltares = kwargs['final_res_dist'] - kwargs['highresolution']
    xmax = kwargs['border'] + deltares / slope_xmax
    d3, r3 = xmax, kwargs['final_res_dist']
    d4, r4 = 2*xmax, kwargs['final_res_dist']

    # Those are the points I fix
    dists = np.array([d0, d1, d2, d3, d4])
    resol = np.array([r0, r1, r2, r3, r4])

    return dists, resol, kwargs

def constant_resolution_global(**kwargs):

    # Setting parameters to the defaults if not passed as arguments
    defaults = {'lowresolution': 25,
                'highresolution': 3,
                'size': 40,
                'margin': 100,
                'final_res_dist': 1000}

    for name, default in defaults.items():
        kwag = kwargs.get(name, None)
        if kwag is None:
            kwargs.update({name: default})

    # Inner Circle
    # ------------------------------
    # From distance 0 to <size> -> constant highresolution
    d0, r0 = 0., kwargs['highresolution']
    d1, r1 = kwargs['size'], kwargs['highresolution']

    # Boundary layers around it
    # ------------------------------
    kwargs['radius'] = kwargs['size']
    kwargs['buffer'] = kwargs['num_boundary_layers'] * kwargs['highresolution']
    kwargs['border'] = kwargs['radius'] + kwargs['buffer']
    d2, r2 = kwargs['border'], kwargs['highresolution']

    # Those are the points I fix
    dists = np.array([d0, d1, d2])
    resol = np.array([r0, r1, r2])

    return dists, resol, kwargs

def ellipse_variable_resolution(lat_values, lon_values, **kwargs):
    """
    Create a resolution map (2D, lat x lon) where the refined region is an
    ellipse instead of a circle.

    Parameters provided via kwargs (defaults set to be compatible with other functions):
      - lowresolution, highresolution: outer and inner target resolutions (km)
      - size: used as default semi-major axis if a_km not set (km)
      - a_km: semi-major axis length in km (if not set, defaults to 'size')
      - b_km: semi-minor axis length in km (if not set, defaults to 'size')
      - angle: rotation angle in degrees CCW from east for the ellipse
      - delta: transition width in normalized ellipse units (unitless, typical 0.05-0.2)
      - margin: extra margin added to radius computation (km)
      - num_boundary_layers: used to compute buffer like other functions
      - lat_ref, lon_ref: ellipse center (degrees)
    Returns:
      - size_map: 2D numpy array with shape (nlat, nlon) with resolution in km
      - kwargs: updated kwargs with keys like 'radius' and 'border'
    """
    # defaults
    defaults = {'lowresolution': 25,
                'highresolution': 3,
                'size': 40,
                'margin': 100,
                'final_res_dist': 1000,
                'angle': 0.0,
                'delta': 0.12,
                'a_km': None,
                'b_km': None,
                }

    for name, default in defaults.items():
        if kwargs.get(name, None) is None:
            kwargs[name] = default

    # fallback a/b to 'size' if not provided
    a_km = kwargs.get('a_km') or kwargs.get('size')
    b_km = kwargs.get('b_km') or kwargs.get('size')
    angle_deg = kwargs.get('angle', 0.0)
    delta = kwargs.get('delta', 0.12)

    if a_km <= 0 or b_km <= 0:
        raise ValueError("Ellipse semi-axes a_km and b_km must be > 0")
    if delta <= 0:
        raise ValueError("delta must be > 0 (use small value for sharp transition)")

    # Build 2D lat/lon mesh (lat_values and lon_values are 1D arrays of coords)
    lats = np.asarray(lat_values)
    lons = np.asarray(lon_values)
    Lon, Lat = np.meshgrid(lons, lats)  # shapes (nlat, nlon)

    # center / reference
    lat_ref = kwargs.get('lat_ref', 0.0)
    lon_ref = kwargs.get('lon_ref', 0.0)
    kwargs['lat_ref'] = lat_ref
    kwargs['lon_ref'] = lon_ref

    print ("building ellipse variable resolution map with parameters:")
    print (f"  lowres={kwargs.get('lowresolution')} km, highres={kwargs.get('highresolution')} km")
    print (f"  ellipse a={a_km} km, b={b_km} km, angle={angle_deg} deg, centered at ({lat_ref} deg, {lon_ref} deg)")

    # Earth radius in km
    earth_radius_km = kwargs.get('earth_radius_km', 6371.0)

    # Convert degrees to radians
    lat = np.radians(Lat)
    lon = np.radians(Lon)
    latc = np.radians(lat_ref)
    lonc = np.radians(lon_ref)

    # Normalize dlon to [-pi,pi] to facilitate calculations
    ## Get distance to reference longitude
    dlon = lon - lonc
    ## Shift range to [0,2pi], then wrap values to [-pi,pi]
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi

    # Local tangent-plane (East, North) approximation (in km)
    dx = earth_radius_km * np.cos(latc) * dlon
    dy = earth_radius_km * (lat - latc)

    # Rotate coordinates by -angle to align dx and dy with ellipse axes
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    dx_local =  c * dx + s * dy
    dy_local = -s * dx + c * dy

    # Normalized elliptical coordinate phi = sqrt((x/a)^2 + (y/b)^2)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.hypot(dx_local / a_km, dy_local / b_km)

    # Linear transition between phi=1 and phi=1+delta
    ## Min and max grid-spacing
    d_min = kwargs.get('highresolution')
    d_max = kwargs.get('lowresolution')
    ## Slope of linear transition
    slope = (d_max - d_min)/delta
    ## Calculate grid-spacing (resolution) map
    ### Initialize resol_map with zeros (same shape as phi)
    resol_map = np.zeros_like(phi)
    ### Condition 1: phi < 1
    resol_map[phi < 1] = d_min
    ### Condition 2: 1 <= phi < 1 + delta
    mask_middle = (phi >= 1) & (phi < 1 + delta)
    resol_map[mask_middle] = d_min + slope * (phi[mask_middle] - 1)
    ### Condition 3: phi >= 1 + delta
    resol_map[phi >= 1 + delta] = d_max

    # Update radius/border attributes in kwargs to be consistent with other functions
    # Use max semi-axis as representative radius for downstream tools that expect a scalar radius
    kwargs['radius'] = max(a_km, b_km) + kwargs.get('margin', 100)
    kwargs['buffer'] = kwargs.get('num_boundary_layers', 0) * kwargs.get('lowresolution', 25)
    kwargs['border'] = kwargs['radius'] + kwargs['buffer']

    return resol_map, kwargs

def ellipse_circle_variable_resolution(lat_values, lon_values, **kwargs):
    """
    Generate a resolution mesh with the following zones:
    - Outside Circle/Outside Ellipse: Low resolution
    - Outside Circle/Outside Ellipse Transition: Linear transition from low to mid resolution
    - Inside Circle/Outside Ellipse: Mid resolution
    - Inside Circle/Outside Ellipse Transition: Linear transition from mid to high resolution
    - Inside Circle/Inside Ellipse: High resolution

    Parameters:
      - lowresolution, midresolution, highresolution: Resolutions (km)
      - inner_radius, outer_radius: Circle transition radii (km)
      - a_km, b_km: Ellipse semi-major and semi-minor axes (km)
      - angle: Ellipse rotation angle (degrees CCW from east)
      - delta: Transition width for ellipse (unitless, typical 0.05-0.2)
      - lat_ref, lon_ref: Ellipse center (degrees)
      - lat_circle_ref, lon_circle_ref: Circle center (degrees)
    Returns:
      - resolution_map: 2D numpy array with resolution values
    """
    # Defaults
    defaults = {
        'lowresolution': 1000,
        'midresolution': 15,
        'highresolution': 3,
        'inner_radius': 5000, #3700,
        'outer_radius': 7000, #5700,
        'a_km': 30,
        'b_km': 20,
        'angle': 0.0,
        'delta': 0.1,
        'lat_ref': 0.0,
        'lon_ref': 0.0,
        'lat_circle_ref': -15.599,
        'lon_circle_ref': -56.101,
        'earth_radius_km': 6371.0,
    }

    for name, default in defaults.items():
        if kwargs.get(name, None) is None:
            kwargs[name] = default

    # Extract parameters
    lowres = kwargs['lowresolution']
    midres = kwargs['midresolution']
    highres = kwargs['highresolution']
    inner_radius = kwargs['inner_radius']
    outer_radius = kwargs['outer_radius']
    a_km = kwargs['a_km']
    b_km = kwargs['b_km']
    angle_deg = kwargs['angle']
    delta = kwargs['delta']
    lat_ref = kwargs['lat_ref']
    lon_ref = kwargs['lon_ref']
    lat_circle_ref = kwargs['lat_circle_ref']
    lon_circle_ref = kwargs['lon_circle_ref']
    earth_radius_km = kwargs['earth_radius_km']

    print ("building ellipse-circle variable resolution map with parameters:")
    print (f"  lowres={lowres} km, midres={midres} km, highres={highres} km")
    print (f"  circle inner_radius={inner_radius} km, outer_radius={outer_radius} km, centered at ({lat_circle_ref} deg, {lon_circle_ref} deg)")
    print (f"  ellipse a={a_km} km, b={b_km} km, angle={angle_deg} deg, centered at ({lat_ref} deg, {lon_ref} deg)")

    # Build 2D lat/lon mesh
    lats = np.asarray(lat_values)
    lons = np.asarray(lon_values)
    Lon, Lat = np.meshgrid(lons, lats)

    # Convert degrees to radians
    lat = np.radians(Lat)
    lon = np.radians(Lon)
    latc_circle = np.radians(lat_circle_ref)
    lonc_circle = np.radians(lon_circle_ref)
    latc_ellipse = np.radians(lat_ref)
    lonc_ellipse = np.radians(lon_ref)

    # Circle distance calculation
    dlon_circle = (lon - lonc_circle + np.pi) % (2.0 * np.pi) - np.pi
    dx_circle = earth_radius_km * np.cos(latc_circle) * dlon_circle
    dy_circle = earth_radius_km * (lat - latc_circle)
    dist_circle = np.sqrt(dx_circle**2 + dy_circle**2)

    # Ellipse distance calculation
    dlon_ellipse = (lon - lonc_ellipse + np.pi) % (2.0 * np.pi) - np.pi
    dx_ellipse = earth_radius_km * np.cos(latc_ellipse) * dlon_ellipse
    dy_ellipse = earth_radius_km * (lat - latc_ellipse)
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    dx_local = c * dx_ellipse + s * dy_ellipse
    dy_local = -s * dx_ellipse + c * dy_ellipse
    phi = np.hypot(dx_local / a_km, dy_local / b_km)

    # Initialize resolution map
    resolution_map = np.zeros_like(phi)

    # Outside Circule/Outside Ellipse
    resolution_map[dist_circle >= outer_radius] = lowres

    # Outside Circle/Outside Ellipse Transition
    mask_transition_outer = (dist_circle >= inner_radius) & (dist_circle < outer_radius)
    slope_outer = (lowres - midres) / (outer_radius - inner_radius)
    resolution_map[mask_transition_outer] = midres + slope_outer * (dist_circle[mask_transition_outer] - inner_radius)

    # Inside Circle/Outside Ellipse
    mask_inside_circle = (dist_circle < inner_radius) & (phi >= 1 + delta)
    resolution_map[mask_inside_circle] = midres

    # Inside Circle/Outside Ellipse Transition
    mask_transition_inner = (dist_circle < inner_radius) & (phi >= 1) & (phi < 1 + delta)
    slope_inner = (midres - highres) / delta
    resolution_map[mask_transition_inner] = highres + slope_inner * (phi[mask_transition_inner] - 1)

    # Inside Circle/Inside Ellipse
    mask_inside_ellipse = phi < 1
    resolution_map[mask_inside_ellipse] = highres

    # Update radius/border attributes in kwargs to be consistent with other functions
    # Use max semi-axis as representative radius for downstream tools that expect a scalar radius
    kwargs['radius'] = max(a_km, b_km) + kwargs.get('margin', 100)
    kwargs['buffer'] = kwargs.get('num_boundary_layers', 0) * kwargs.get('lowresolution', 25)
    kwargs['border'] = kwargs['radius'] + kwargs['buffer']

    return resolution_map, kwargs

def ellipse_variable_resolution_smoothstep(lat_values, lon_values, **kwargs):
    """
    Create a resolution map (2D, lat x lon) where the refined region is an
    ellipse instead of a circle.

    Parameters provided via kwargs (defaults set to be compatible with other functions):
      - lowresolution, highresolution: outer and inner target resolutions (km)
      - size: used as default semi-major axis if a_km not set (km)
      - a_km: semi-major axis length in km (if not set, defaults to 'size')
      - b_km: semi-minor axis length in km (if not set, defaults to 'size')
      - angle: rotation angle in degrees CCW from east for the ellipse
      - delta: transition width in normalized ellipse units (unitless, typical 0.05-0.2)
      - margin: extra margin added to radius computation (km)
      - num_boundary_layers: used to compute buffer like other functions
      - lat_ref, lon_ref: ellipse center (degrees)
    Returns:
      - size_map: 2D numpy array with shape (nlat, nlon) with resolution in km
      - kwargs: updated kwargs with keys like 'radius' and 'border'
    """
    # defaults
    defaults = {'lowresolution': 25,
                'highresolution': 3,
                'size': 40,
                'margin': 100,
                'final_res_dist': 1000,
                'angle': 0.0,
                'delta': 0.12,
                'a_km': None,
                'b_km': None,
                }

    for name, default in defaults.items():
        if kwargs.get(name, None) is None:
            kwargs[name] = default

    # fallback a/b to 'size' if not provided
    a_km = kwargs.get('a_km') or kwargs.get('size')
    b_km = kwargs.get('b_km') or kwargs.get('size')
    angle_deg = kwargs.get('angle', 0.0)
    delta = kwargs.get('delta', 0.12)

    if a_km <= 0 or b_km <= 0:
        raise ValueError("Ellipse semi-axes a_km and b_km must be > 0")
    if delta <= 0:
        raise ValueError("delta must be > 0 (use small value for sharp transition)")

    # Build 2D lat/lon mesh (lat_values and lon_values are 1D arrays of coords)
    lats = np.asarray(lat_values)
    lons = np.asarray(lon_values)
    Lon, Lat = np.meshgrid(lons, lats)  # shapes (nlat, nlon)

    # center / reference
    lat_ref = kwargs.get('lat_ref', 0.0)
    lon_ref = kwargs.get('lon_ref', 0.0)
    kwargs['lat_ref'] = lat_ref
    kwargs['lon_ref'] = lon_ref

    # Earth radius in km
    earth_radius_km = kwargs.get('earth_radius_km', 6371.0)

    # Convert degrees to radians
    lat = np.radians(Lat)
    lon = np.radians(Lon)
    latc = np.radians(lat_ref)
    lonc = np.radians(lon_ref)

    # dlon normalized [-pi,pi]
    dlon = lon - lonc
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi

    # Local tangent-plane (East, North) approximation (in km)
    dx = earth_radius_km * np.cos(latc) * dlon   # east
    dy = earth_radius_km * (lat - latc)          # north

    # Rotate coordinates by -angle to align ellipse axes with x/y
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    x_local =  c * dx + s * dy
    y_local = -s * dx + c * dy

    # Normalized elliptical coordinate phi = sqrt((x/a)^2 + (y/b)^2)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.hypot(x_local / a_km, y_local / b_km)

    # Smooth transition using smoothstep between phi=1 and phi=1+delta
    t = (phi - 1.0) / delta
    t_clamped = np.clip(t, 0.0, 1.0)
    smooth = 3.0 * t_clamped**2 - 2.0 * t_clamped**3

    s_min = kwargs.get('highresolution')
    s_max = kwargs.get('lowresolution')

    size_map = s_min + (s_max - s_min) * smooth

    # enforce exact inside/outside values
    inside_mask = (phi <= 1.0)
    size_map[inside_mask] = s_min
    outside_mask = (phi >= 1.0 + delta)
    size_map[outside_mask] = s_max

    # Update radius/border attributes in kwargs to be consistent with other functions
    # Use max semi-axis as representative radius for downstream tools that expect a scalar radius
    kwargs['radius'] = max(a_km, b_km) + kwargs.get('margin', 100)
    kwargs['buffer'] = kwargs.get('num_boundary_layers', 0) * kwargs.get('lowresolution', 25)
    kwargs['border'] = kwargs['radius'] + kwargs['buffer']

    return size_map, kwargs


def variable_resolution_latlonmap(p, with_distance=False):
    """
    Target cell-size map (km) on a global lat-lon grid.

    p: validated parameters from mesh_config.resolve (refinement x cut).
    The refinement sets the cell size near the center; for regional cuts a
    constant ring of n_layers * ring_res beyond the cut holds the MPAS
    boundary layers, followed by a far field that coarsens up to 1000 km
    (see mesh_geometry.target_resolution).
    """
    print('\n>> Creating a variable resolution map')
    print(f"\trefinement={p['refinement']}, cut={p['cut']}")

    if p['refinement'] == 'uniform' and p['cut'] == 'global':
        # Constant field: a coarse grid describes it exactly.
        dist_degrees = 1.0
    else:
        # Grid spacing of the finest resolution, so no detail is smoothed out.
        dist_degrees = p['high_res'] / 110.
    nlat = int(180. / dist_degrees) + 1
    nlon = int(360. / dist_degrees) + 1
    print(f'\tlat-lon grid: {nlat} x {nlon} points')

    ds = xr.Dataset(coords={'lat': np.linspace(-90., 90., nlat),
                            'lon': np.linspace(-180., 180., nlon)})
    lat2d = ds['lat'].values[:, None]
    lon2d = ds['lon'].values[None, :]

    ring_res = ring_resolution(p)
    n_layers = p.get('n_layers', 0)
    res = geo.target_resolution(lat2d, lon2d, p, ring_res, n_layers)
    ds['resolution'] = xr.DataArray(data=res, dims=('lat', 'lon'))
    if with_distance:
        ds['distance'] = xr.DataArray(
            data=geo.haversine_km(lat2d, lon2d, p['lat'], p['lon']),
            dims=('lat', 'lon'))

    ds.attrs = {k: v for k, v in p.items() if v is not None}
    ds.attrs['ring_resolution'] = ring_res
    if p['cut'] != 'global':
        ds.attrs['ring_width_km'] = n_layers * ring_res
    print(f"\tcell size range: {float(res.min()):.3g} - {float(res.max()):.3g} km")
    return ds


def get_mesh_from_resolution(resolution_ds, basename='./mesh'):
    print('\n>> Generating a MPAS mesh')

    # jigsaw
    print('\n\t .- Jigsaw generation')

    mesh_file = jigsaw_gen_sph_grid(resolution_ds['resolution'].values,
                                    resolution_ds['lon'].values,
                                    resolution_ds['lat'].values,
                                    basename=basename)

    # mpas-tools

    print('\n\t .- Jigsaw to netCDF')
    out_file_triangles = basename + '.triangles.nc'
    jigsaw_to_netcdf(msh_filename=mesh_file,
                     output_name=out_file_triangles,
                     on_sphere=True, sphere_radius=1.0)

    print('\n\t .- Convert to MPAS format')
    out_file_mpas = basename + '.grid.nc'
    out_file_graph_info = basename + ".graph.info"
    write_netcdf(
        convert(xr.open_dataset(out_file_triangles),
                dir=os.path.dirname(basename),
                graphInfoFileName=out_file_graph_info),
        out_file_mpas)

    return out_file_mpas, out_file_graph_info


def cut_circular_region(mpas_global_file,
                        region_radius_meters,
                        regional_grid=None,
                        regional_grid_info=None,
                        lat_cen=0., lon_cen=0.,
                        path_create_region=PATH_LIMITED_AREA,
                        ):

    if not os.path.isdir(path_create_region):
        raise IOError('The path to the MPAS-Limited-Area folder is not '
                      'correct. Pass it to the function '
                      'cut_circular_region_beta or define a correct '
                      'default in variable PATH_LIMITED_AREA (in '
                      'mesh_generator.py.')

    print('\n>> Cutting a circular region')
    print('\t centered at %.4f, %.4f' % (lat_cen, lon_cen))
    print('\t with radius %.1fkm' % float(region_radius_meters/1000))

    if region_radius_meters < 5000:
        raise ValueError('Do you want a %.0fm radius region?'
                         'That looks way too small. Maybe you passed '
                         'the radius in km instead as in meters.'
                         % region_radius_meters)

    if not os.path.exists(mpas_global_file):
        raise IOError('Wanted to use the MPAS global file %s but'
                      'it does not seem to exist.' % mpas_global_file)

    os.system('cp ' + mpas_global_file + ' global.grid.nc')

    with open('points.txt', 'w') as f:
        f.write('Name: circle\n')
        f.write('Type: circle\n')
        f.write('Point: %.4f, %.4f\n' % (lat_cen, lon_cen))
        f.write('radius: %.0f\n' % region_radius_meters)

    # If there are regional files we should erase them
    os.system('rm -f circle.grid.nc')
    os.system('rm -f circle.graph.info')

    # Execute create region
    os.system(path_create_region + '/create_region points.txt ' +
              mpas_global_file)

    if not os.path.exists('circle.grid.nc') or \
            not os.path.exists('circle.graph.info'):
        raise AttributeError('The regions were not generated correctly')

    if regional_grid is not None:
        os.system('cp circle.grid.nc ' + regional_grid)

        if regional_grid_info is None:
            regional_grid_info = regional_grid.replace('.nc', '.graph.info')
    else:
        regional_grid = 'circle.grid.nc'

    if regional_grid_info is not None:
        os.system('cp circle.graph.info ' + regional_grid_info)
    else:
        regional_grid_info = 'circle.graph.info'

    return regional_grid, regional_grid_info


def cut_circular_region_beta(mpas_global_file,
                             region_radius_meters,
                             regional_grid=None,
                             regional_grid_info=None,
                             num_boundary_layers=8,
                             lat_cen=0., lon_cen=0.,
                             path_create_region=PATH_LIMITED_AREA,
                             ):
    if not os.path.isdir(path_create_region):
        raise IOError('The path to the MPAS-Limited-Area folder is not '
                      'correct. Pass it to the function '
                      'cut_circular_region_beta or define a correct '
                      'default in variable PATH_LIMITED_AREA (in '
                      'mesh_generator.py.')

    create_region_exec = path_create_region + '/create_region_no_file'
    
    if not os.path.isfile(create_region_exec):
        raise IOError('Your MPAS-Limited-Area folder does not '
                      'contain the expected executable. Maybe you '
                      'have an old version? ' + create_region_exec)

    print('\n>> Cutting a circular region')
    print('\t centered at %.4f, %.4f' % (lat_cen, lon_cen))
    print('\t with radius %.1fkm' % float(region_radius_meters/1000))

    if region_radius_meters < 5000:
        raise ValueError('Do you want a %.0fm radius region?'
                         'That looks way too small. Maybe you passed '
                         'the radius in km instead as in meters.'
                         % region_radius_meters)

    if not os.path.exists(mpas_global_file):
        raise IOError('Wanted to use the MPAS global file %s but'
                      'it does not seem to exist.' % mpas_global_file)

    # If there are regional files we should erase them
    if regional_grid is None:
        regional_grid = 'circle.grid.nc'
    if regional_grid_info is None:
        regional_grid_info = regional_grid.replace('.nc', '.graph.info')

    os.system('rm -f ' + regional_grid)
    os.system('rm -f ' + regional_grid_info)

    # Execute create region
    os.system(create_region_exec + ' ' + mpas_global_file +
              ' --num_boundary_layers ' + str(num_boundary_layers) +
              ' --oregion ' + regional_grid +
              ' --ograph ' + regional_grid_info +
              ' region.type:circle' +
              ' region.radius:' + str(region_radius_meters) +
              ' region.in_point:' + str(lat_cen) + ',' + str(lon_cen)
              )

    if not os.path.exists(regional_grid) or \
            not os.path.exists(regional_grid_info):
        raise AttributeError('The regions were not generated correctly')

    return regional_grid, regional_grid_info

def write_points_file(p, pts_file):
    """MPAS-Limited-Area points file for the cut (circle, or custom polygon)."""
    with open(pts_file, 'w') as f:
        f.write('Name: region\n')
        if p['cut'] == 'circle':
            f.write('Type: circle\n')
            f.write(f"Point: {p['lat']:.8f}, {p['lon']:.8f}\n")
            f.write(f"radius: {p['cut_radius'] * 1000.0:.3f}\n")
        elif p['cut'] == 'ellipse':
            # Polygon on the same local-plane ellipse the resolution map uses.
            blat, blon = geo.cut_boundary(p)
            f.write('Type: custom\n')
            f.write(f"Point: {p['lat']:.8f}, {p['lon']:.8f}\n")
            for la, lo in zip(blat, blon):
                f.write(f'{la:.8f}, {lo:.8f}\n')
        else:
            raise ValueError(f"no points file for cut={p['cut']!r}")


def cut_region(mpas_global_file, p, work_dir, regional_grid, regional_grid_info,
               path_create_region=PATH_LIMITED_AREA):
    """
    Cut the regional mesh (circle or ellipse, see p['cut']) out of a global
    MPAS mesh with MPAS-Limited-Area, adding p['n_layers'] boundary layers
    (n_layers - 1 are marked; 8 gives the 7 MPAS needs).
    """
    script = os.path.join(path_create_region, 'create_region_n_layers_as_input')
    if not os.path.isfile(script):
        raise IOError(f'MPAS-Limited-Area script not found: {script}')
    if not os.path.exists(mpas_global_file):
        raise IOError(f'global MPAS mesh not found: {mpas_global_file}')

    print(f"\n>> Cutting a {p['cut']} region centered at "
          f"({p['lat']:g}, {p['lon']:g}) with {p['n_layers']} layers")
    pts_file = os.path.join(work_dir, 'region.pts')
    write_points_file(p, pts_file)
    for old in Path(work_dir).glob('region.*'):
        if old.suffix in ('.nc', '.info'):
            old.unlink()

    subprocess.run([sys.executable, script, pts_file,
                    os.path.abspath(mpas_global_file), str(p['n_layers'])],
                   cwd=work_dir, check=True)

    grids = sorted(Path(work_dir).glob('region.*.nc'))
    graph = Path(work_dir) / 'region.graph.info'
    if len(grids) != 1 or not graph.exists():
        raise IOError('MPAS-Limited-Area did not produce the regional grid '
                      f'and graph.info in {work_dir}')
    shutil.move(str(grids[0]), regional_grid)
    shutil.move(str(graph), regional_grid_info)
    return regional_grid, regional_grid_info

def full_generation_process(mpas_grid_file, grid, redo=True,
                            do_plots=True, do_region=False, **kwargs):

    if os.path.isfile(mpas_grid_file) and not redo:
        print(' .. already available')
        return

    graph_info_file = mpas_grid_file.replace('.nc', '.graph.info')
    path_save = os.path.dirname(mpas_grid_file)

    os.system('rm -f ' + mpas_grid_file)
    os.system('rm -f ' + graph_info_file)

    start_time = time.time()
    resolution_ds = variable_resolution_latlonmap(grid, **kwargs)
    duration_resolution = time.time() - start_time
    print(' .. finished finding resolution map: %.3fs\n\n' % duration_resolution)

    border = resolution_ds.attrs['border']
    radius = resolution_ds.attrs['radius']
    lat_ref = resolution_ds.attrs['lat_ref']
    lon_ref = resolution_ds.attrs['lon_ref']

    if do_plots:
        print('Plotting')
        start_time = time.time()
        view_resolution_map(resolution_ds,
                            pdfname=path_save + '/resolution.pdf',
                            list_distances=[
                                #1000,
                                500, border, radius])
        duration_plots = time.time() - start_time
        print(' .. finished doing resolution plots: %.3fs\n\n' % duration_plots)

    start_time = time.time()
    tmp_mesh_file, tmp_graph_info = get_mesh_from_resolution(resolution_ds,
                                                             basename='mesh')
    duration_gen = time.time() - start_time
    print(' .. finished generating global mesh: %.3fs\n\n' % duration_gen)

    mpas_grid_file_tmp = mpas_grid_file + '.tmp'
    if do_region:
        start_time = time.time()
        num_boundary_layers = kwargs.get('num_boundary_layers', 8)
        print('cutting circular region')
        cut_circular_region_beta(tmp_mesh_file, radius*1000,
                                 regional_grid=mpas_grid_file_tmp,
                                 regional_grid_info=graph_info_file,
                                 num_boundary_layers=num_boundary_layers,
                                 lat_cen=lat_ref, lon_cen=lon_ref,
                                 )
        
        duration_region = time.time() - start_time
        print(' .. finished cutting region: %.3fs\n\n' % duration_region)

        if not os.path.isfile(mpas_grid_file_tmp) or \
                not os.path.isfile(graph_info_file):
            raise IOError('The file we had to generate was not generated')
    else:
        duration_region = 0.
        os.system('mv ' + tmp_mesh_file + ' ' + mpas_grid_file_tmp)
        os.system('mv ' + tmp_graph_info + ' ' + graph_info_file)

    # Open dataset and update attributes
    mpas_ds = xr.open_dataset(mpas_grid_file_tmp)
    for name, value in resolution_ds.attrs.items():
        mpas_ds.attrs['vtx-param-' + str(name)] = value
    if do_region:
        mpas_ds.attrs['vtx-region-num_boundary_layers'] = num_boundary_layers
        lowres = mpas_ds.attrs['vtx-param-lowresolution']
        region_border = radius + (num_boundary_layers * lowres) * 0.6
        mpas_ds.attrs['vtx-region_border'] = region_border

    # Update the duration of the steps
    durations_process = {
        'resolution': duration_resolution,
        'generation': duration_gen,
        'region': duration_region,
        'total': duration_resolution + duration_gen + duration_region,
    }
    for name, value in durations_process.items():
        mpas_ds.attrs['vtx-duration-' + name] = '%.2f' % value

    print('\n MESH:\n')
    print(mpas_ds)
    mpas_ds.to_netcdf(mpas_grid_file)

    if not os.path.isfile(mpas_grid_file):
        raise IOError('Could not update attributes of file ' +
                      mpas_grid_file_tmp)

    #os.system('rm -f ' + mpas_grid_file_tmp)

    if do_plots:
        if do_region:
            print('Resolution Plots')
            view_mpas_regional_mesh(mpas_grid_file,
                                    outfile=path_save + '/resolution_mesh.png')

    return mpas_grid_file, graph_info_file

def generate_mesh(p, mpas_grid_file, work_dir, do_plots=False):
    """
    Build the MPAS mesh for validated parameters p (see mesh_config.resolve).

    Writes, next to each other:
      <name>.grid.nc               mesh for the MPAS workflow
      <name>.graph.info            graph for gpmetis
      <name>_for_plotting.grid.nc  same mesh with the vtx-param-* metadata
    Intermediate files go to work_dir. Returns a dict of paths and durations.
    """
    if not mpas_grid_file.endswith('.grid.nc'):
        raise ValueError('mpas_grid_file must end in .grid.nc')
    graph_info_file = mpas_grid_file.replace('.grid.nc', '.graph.info')
    plotting_file = mpas_grid_file.replace('.grid.nc', '_for_plotting.grid.nc')

    ring_res = ring_resolution(p)
    problem = geo.check_cut_encloses_refinement(p, ring_res)
    if problem:
        raise ValueError(problem)

    start_time = time.time()
    resolution_ds = variable_resolution_latlonmap(p, with_distance=do_plots)
    duration_resolution = time.time() - start_time
    print(' .. finished finding resolution map: %.3fs\n' % duration_resolution)

    if do_plots:
        extent = {'global': 2000.,
                  'circle': p.get('cut_radius', 0.),
                  'ellipse': max(p.get('cut_a_km', 0.), p.get('cut_b_km', 0.))}[p['cut']]
        extent += resolution_ds.attrs.get('ring_width_km', 0.)
        view_resolution_map(resolution_ds,
                            pdfname=os.path.join(work_dir, 'resolution.pdf'),
                            list_distances=[extent])

    start_time = time.time()
    mesh_file, mesh_graph = get_mesh_from_resolution(
        resolution_ds, basename=os.path.join(work_dir, 'mesh'))
    duration_gen = time.time() - start_time
    print(' .. finished generating global mesh: %.3fs\n' % duration_gen)

    start_time = time.time()
    if p['cut'] == 'global':
        shutil.move(mesh_file, mpas_grid_file)
        shutil.move(mesh_graph, graph_info_file)
    else:
        cut_region(mesh_file, p, work_dir, mpas_grid_file, graph_info_file)
    duration_region = time.time() - start_time

    # Copy of the mesh carrying the generation parameters as attributes.
    with xr.open_dataset(mpas_grid_file) as mpas_ds:
        mpas_ds = mpas_ds.load()
    for name, value in resolution_ds.attrs.items():
        mpas_ds.attrs['vtx-param-' + str(name)] = value
    durations = {'resolution': duration_resolution, 'generation': duration_gen,
                 'region': duration_region,
                 'total': duration_resolution + duration_gen + duration_region}
    for name, value in durations.items():
        mpas_ds.attrs['vtx-duration-' + name] = '%.2f' % value
    for var in mpas_ds.variables:
        mpas_ds[var].encoding['_FillValue'] = None
    mpas_ds.to_netcdf(plotting_file)

    if do_plots and p['cut'] != 'global':
        view_mpas_regional_mesh(plotting_file,
                                outfile=os.path.join(work_dir, 'resolution_mesh.png'))

    print(f'\n MESH FOR MPAS WORKFLOW: {mpas_grid_file}')
    print(f' MESH FOR PLOTTING:      {plotting_file}')
    return {'grid': mpas_grid_file, 'graph_info': graph_info_file,
            'plotting': plotting_file, 'durations': durations}
