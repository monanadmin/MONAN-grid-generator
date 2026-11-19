"""
Mesh options, input validation and mesh naming.

A mesh is defined by two independent choices:

    refinement = uniform | circle | ellipse   (shape of the high-resolution area)
    cut        = global  | circle | ellipse   (shape of the regional domain, or none)

Every combination is valid. Each choice needs its own parameters; parameters
that the chosen combination does not use may be present and are ignored.
"""

REFINEMENTS = ('uniform', 'circle', 'ellipse')
CUTS = ('global', 'circle', 'ellipse')

# MPAS limited-area meshes have 7 boundary layers (5 relaxation + 2 specified).
# MPAS-Limited-Area marks n_layers - 1 of them, so 8 gives the required 7.
MPAS_N_LAYERS = 8

# What each parameter means (used in error messages).
DESCRIPTIONS = {
    'lat': 'latitude of the mesh center (degrees)',
    'lon': 'longitude of the mesh center (degrees)',
    'refinement': 'refinement shape: ' + ' | '.join(REFINEMENTS),
    'cut': 'regional cut: ' + ' | '.join(CUTS),
    'high_res': 'finest cell size (km)',
    'low_res': 'cell size outside the refinement (km)',
    'inner_radius': 'radius of the high-resolution circle (km)',
    'outer_radius': 'radius where the transition to low_res ends (km)',
    'a_km': 'ellipse semi-axis along the rotated x direction (km)',
    'b_km': 'ellipse semi-axis along the rotated y direction (km)',
    'angle': 'ellipse rotation, degrees counter-clockwise from east',
    'delta': 'ellipse transition width, in units of the normalised elliptical radius',
    'cut_radius': 'radius of the circular cut (km)',
    'cut_a_km': 'semi-axis of the elliptical cut along its rotated x direction (km)',
    'cut_b_km': 'semi-axis of the elliptical cut along its rotated y direction (km)',
    'cut_angle': 'rotation of the elliptical cut, degrees counter-clockwise from east',
    'n_layers': 'MPAS-Limited-Area boundary layers (8 gives the 7 MPAS needs)',
    'N': 'number of MPI partitions for gpmetis',
}

REFINEMENT_KEYS = {
    'uniform': ('high_res',),
    'circle': ('high_res', 'low_res', 'inner_radius', 'outer_radius'),
    'ellipse': ('high_res', 'low_res', 'a_km', 'b_km', 'angle', 'delta'),
}
CUT_KEYS = {
    'global': (),
    'circle': ('cut_radius', 'n_layers'),
    'ellipse': ('cut_a_km', 'cut_b_km', 'cut_angle', 'n_layers'),
}
DEFAULTS = {'angle': 0.0, 'delta': 0.12, 'n_layers': MPAS_N_LAYERS, 'N': 1}

ALL_KEYS = ('lat', 'lon', 'refinement', 'cut', 'high_res', 'low_res',
            'inner_radius', 'outer_radius', 'a_km', 'b_km', 'angle', 'delta',
            'cut_radius', 'cut_a_km', 'cut_b_km', 'cut_angle', 'n_layers', 'N')

LEGACY_TRANSLATION = {
    ('doughnut', 'y'): 'refinement=circle, cut=circle (cut_radius defaults to outer_radius)',
    ('doughnut', 'n'): 'refinement=circle, cut=global',
    ('constant', 'y'): 'refinement=uniform, cut=circle, cut_radius=<your inner_radius>',
    ('constant', 'n'): 'refinement=uniform, cut=global',
    ('ellipse', 'y'): 'refinement=ellipse, cut=ellipse',
    ('ellipse', 'n'): 'refinement=ellipse, cut=global',
}


class MeshConfigError(ValueError):
    """Raised when the mesh inputs are missing, inconsistent or invalid."""


def legacy_error(grid_type=None, do_regional=None):
    """Message for input files that still use grid_type / do_regional."""
    hint = LEGACY_TRANSLATION.get((grid_type, do_regional))
    msg = ('grid_type and do_regional are no longer used. Use refinement '
           '(uniform | circle | ellipse) and cut (global | circle | ellipse) '
           'instead.')
    if hint:
        msg += f' For grid_type={grid_type}, do_regional={do_regional} use: {hint}.'
    return msg


def resolve(raw):
    """
    Validate raw inputs and return the complete parameter dict.

    raw: dict of key -> value (values may be None for "not given").
    Returns (params, ignored) where params has defaults and derived cut sizes
    filled in, and ignored lists the given keys the chosen option does not use.
    Raises MeshConfigError with every problem found.
    """
    given = {k: v for k, v in raw.items() if v is not None}
    errors = []

    if 'grid_type' in given or 'do_regional' in given:
        raise MeshConfigError(legacy_error(given.get('grid_type'),
                                           given.get('do_regional')))

    refinement = given.get('refinement')
    cut = given.get('cut')
    if refinement not in REFINEMENTS:
        errors.append(f"refinement must be one of {', '.join(REFINEMENTS)} "
                      f"(got {refinement!r})")
    if cut not in CUTS:
        errors.append(f"cut must be one of {', '.join(CUTS)} (got {cut!r})")
    if errors:
        raise MeshConfigError('; '.join(errors))

    p = dict(DEFAULTS)
    p.update(given)
    if refinement == 'ellipse' and cut == 'ellipse' and 'cut_angle' not in given:
        p['cut_angle'] = p['angle']
    if cut == 'ellipse' and 'cut_angle' not in p:
        p['cut_angle'] = 0.0

    # Cut sizes that can be derived from the refinement (doughnut-like logic:
    # cut where the transition to low_res ends).
    if cut == 'circle' and 'cut_radius' not in given and refinement == 'circle':
        if 'outer_radius' in given:
            p['cut_radius'] = given['outer_radius']
    if cut == 'ellipse' and refinement == 'ellipse':
        if 'cut_a_km' not in given and 'a_km' in given:
            p['cut_a_km'] = (1.0 + p['delta']) * given['a_km']
        if 'cut_b_km' not in given and 'b_km' in given:
            p['cut_b_km'] = (1.0 + p['delta']) * given['b_km']

    needed = ('lat', 'lon') + REFINEMENT_KEYS[refinement] + CUT_KEYS[cut]
    missing = [k for k in needed if k not in p]
    if missing:
        lines = [f'  {k}: {DESCRIPTIONS[k]}' for k in missing]
        raise MeshConfigError(
            f'missing inputs for refinement={refinement}, cut={cut}:\n'
            + '\n'.join(lines))

    for k in ('high_res', 'low_res', 'outer_radius', 'a_km', 'b_km', 'delta',
              'cut_radius', 'cut_a_km', 'cut_b_km'):
        if k in needed and p[k] <= 0:
            errors.append(f'{k} must be > 0 (got {p[k]})')
    if refinement == 'circle':
        if p['inner_radius'] < 0:
            errors.append(f"inner_radius must be >= 0 (got {p['inner_radius']})")
        if p['inner_radius'] >= p['outer_radius']:
            errors.append('inner_radius must be smaller than outer_radius')
    if refinement in ('circle', 'ellipse') and p['high_res'] >= p['low_res']:
        errors.append('high_res must be smaller than low_res')
    if not -90 <= p['lat'] <= 90:
        errors.append(f"lat must be in [-90, 90] (got {p['lat']})")
    if cut != 'global':
        if int(p['n_layers']) != p['n_layers'] or p['n_layers'] < MPAS_N_LAYERS:
            errors.append(
                f"n_layers must be an integer >= {MPAS_N_LAYERS}: n_layers=N "
                f"gives N-1 boundary layers and MPAS needs 7 (got {p['n_layers']})")
    if int(p['N']) != p['N'] or p['N'] < 1:
        errors.append(f"N must be a positive integer (got {p['N']})")
    if errors:
        raise MeshConfigError('; '.join(errors))

    p['n_layers'] = int(p['n_layers'])
    p['N'] = int(p['N'])
    used = set(needed) | {'refinement', 'cut', 'N'}
    ignored = sorted(k for k in given if k not in used)
    for k in [k for k in p if k not in used]:
        del p[k]
    return p, ignored


def ring_resolution(p):
    """Cell size of the constant ring that holds the MPAS boundary layers."""
    return p['high_res'] if p['refinement'] == 'uniform' else p['low_res']


def _g(v):
    return f'{v:g}'


def mesh_name(p):
    """Mesh name built from the exact parameter values of the chosen option."""
    parts = [f"lat_{_g(p['lat'])}", f"lon_{_g(p['lon'])}", p['refinement']]
    r = p['refinement']
    if r == 'uniform':
        parts.append(f"res_{_g(p['high_res'])}")
    elif r == 'circle':
        parts += [f"ir_{_g(p['inner_radius'])}", f"or_{_g(p['outer_radius'])}",
                  f"hres_{_g(p['high_res'])}", f"lres_{_g(p['low_res'])}"]
    else:
        parts += [f"a_{_g(p['a_km'])}", f"b_{_g(p['b_km'])}",
                  f"ang_{_g(p['angle'])}", f"delta_{_g(p['delta'])}",
                  f"hres_{_g(p['high_res'])}", f"lres_{_g(p['low_res'])}"]
    parts += ['cut', p['cut']]
    if p['cut'] == 'circle':
        parts.append(f"r_{_g(p['cut_radius'])}")
    elif p['cut'] == 'ellipse':
        parts += [f"a_{_g(p['cut_a_km'])}", f"b_{_g(p['cut_b_km'])}",
                  f"ang_{_g(p['cut_angle'])}"]
    if p['cut'] != 'global' and p['n_layers'] != MPAS_N_LAYERS:
        parts.append(f"nl_{p['n_layers']}")
    return '_'.join(parts)
