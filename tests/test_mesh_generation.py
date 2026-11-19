"""
End-to-end and unit tests for MONAN-grid-generator.

Run from the repository root inside the monan-grid conda env:
    pytest tests/ -v
The end-to-end part builds 10 coarse meshes (all 9 refinement x cut
combinations plus one across the date line) and takes a few minutes.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO, SCRIPTS, VTX, needs_mesh_tools
from vtxmpasmeshes import mesh_geometry as geo
from vtxmpasmeshes.mesh_config import MeshConfigError, mesh_name, resolve

CREATE = VTX / 'create_regional_mesh.py'
CENTER = dict(lat=-22.7, lon=-45.0)
CIRCLE = dict(refinement='circle', inner_radius=300, outer_radius=600,
              high_res=40, low_res=80)
ELLIPSE = dict(refinement='ellipse', a_km=600, b_km=400, angle=30, delta=0.5,
               high_res=40, low_res=80)

# Coarse versions of all 9 refinement x cut combinations (+ one at the date line).
CASES = {
    'uniform-global': dict(CENTER, refinement='uniform', high_res=240, cut='global'),
    'uniform-circle': dict(CENTER, refinement='uniform', high_res=60, cut='circle',
                           cut_radius=600, N=4),
    'uniform-ellipse': dict(CENTER, refinement='uniform', high_res=60, cut='ellipse',
                            cut_a_km=800, cut_b_km=500, cut_angle=30, N=4),
    'circle-global': dict(CENTER, refinement='circle', inner_radius=400,
                          outer_radius=1500, high_res=80, low_res=240, cut='global'),
    'circle-circle': dict(CENTER, **CIRCLE, cut='circle', N=4),
    'circle-ellipse': dict(CENTER, **CIRCLE, cut='ellipse', cut_a_km=900,
                           cut_b_km=700, cut_angle=20, N=4),
    'ellipse-global': dict(CENTER, refinement='ellipse', a_km=800, b_km=500, angle=30,
                           delta=2, high_res=80, low_res=240, cut='global'),
    'ellipse-circle': dict(CENTER, **ELLIPSE, cut='circle', cut_radius=1000, N=4),
    'ellipse-ellipse': dict(CENTER, **ELLIPSE, cut='ellipse', N=4),
    'ellipse-ellipse-dateline': dict(lat=55.0, lon=178.0, refinement='ellipse',
                                     a_km=500, b_km=300, angle=-20, delta=0.5,
                                     high_res=40, low_res=80, cut='ellipse', N=4),
}


def cli_args(params):
    args = []
    for k, v in params.items():
        args += [f'--{k}', str(v)]
    return args


def run_create(params, meshes_dir, input_file=None, extra=()):
    cmd = [sys.executable, str(CREATE), '--meshes_dir', str(meshes_dir)]
    if input_file:
        cmd += ['--input_file', str(input_file)]
    return subprocess.run(cmd + cli_args(params) + list(extra),
                          capture_output=True, text=True)


# --------------------------------------------------------------------------
# End-to-end: build every combination once
# --------------------------------------------------------------------------
@pytest.fixture(scope='module')
def built(tmp_path_factory):
    meshes_dir = tmp_path_factory.mktemp('grids')
    input_file = meshes_dir / 'mesh_input_file.txt'
    input_file.write_text('# test input file\n')
    out = {}
    for case, params in CASES.items():
        proc = run_create(params, meshes_dir, input_file)
        p, _ = resolve(params)
        name = mesh_name(p) + '.region'
        out[case] = dict(proc=proc, p=p, name=name, dir=meshes_dir / name)
    out['_input_file'] = input_file
    out['_last'] = out[list(CASES)[-1]]['name']
    return out


@needs_mesh_tools
@pytest.mark.parametrize('case', list(CASES))
def test_mesh_is_built_and_passes_every_check(built, case):
    from vtxmpasmeshes import mesh_checks as mc
    b = built[case]
    assert b['proc'].returncode == 0, b['proc'].stdout[-3000:] + b['proc'].stderr[-3000:]
    d, name, p = b['dir'], b['name'], b['p']
    grid = d / f'{name}.grid.nc'
    graph = d / f'{name}.graph.info'
    for f in (grid, graph, d / f'{name}_for_plotting.grid.nc', d / 'mesh_input_file.txt'):
        assert f.exists(), f'missing output {f.name}'
    part = d / f'{name}.graph.info.part.{p["N"]}'
    assert part.exists() == (p['N'] > 1)
    results = mc.check_mesh(str(grid), p, graph_file=str(graph),
                            partition_file=str(part) if p['N'] > 1 else None,
                            nparts=p['N'])
    failed = [r for r in results if not r.ok]       # warnings count as failures here
    assert not failed, '\n' + mc.report(failed)
    if p['cut'] != 'global':
        assert any(r.name == 'boundary layers' for r in results)
    # mesh plot and report
    assert (d / f'{name}.png').stat().st_size > 10_000
    report = (d / f'{name}.report.html').read_text()
    for r in results:
        assert r.name in report, f'check {r.name!r} missing from the report'
    assert 'all ' in report and 'checks passed' in report
    assert report.count('src="data:image/png;base64,') >= (8 if p['cut'] != 'global' else 7)
    assert 'Boundary layers of this mesh' in report or p['cut'] == 'global'
    assert 'WARNING: the mesh plot/report could not be made' not in b['proc'].stderr


@needs_mesh_tools
def test_uniform_global_has_uniform_cells(built):
    from vtxmpasmeshes import mesh_checks as mc
    b = built['uniform-global']
    m = mc.Mesh(str(b['dir'] / f"{b['name']}.grid.nc"))
    size = mc.cell_size_km(m)
    assert abs(np.median(size) / 240 - 1) < 0.05
    assert np.std(size) / np.mean(size) < 0.05


@needs_mesh_tools
def test_refinement_is_actually_fine_at_the_center(built):
    """The finest cells are near the requested high_res at the center."""
    from vtxmpasmeshes import mesh_checks as mc
    for case in ('circle-circle', 'ellipse-ellipse', 'circle-global', 'ellipse-global'):
        b = built[case]
        m = mc.Mesh(str(b['dir'] / f"{b['name']}.grid.nc"))
        d = geo.haversine_km(m.lat, m.lon, b['p']['lat'], b['p']['lon'])
        near = mc.cell_size_km(m)[d < 150]
        assert abs(np.median(near) / b['p']['high_res'] - 1) < 0.1, case


@needs_mesh_tools
def test_no_report_skips_plot_and_report(tmp_path):
    proc = run_create(CASES['uniform-global'], tmp_path, extra=['--no_report'])
    assert proc.returncode == 0, proc.stderr[-2000:]
    name = mesh_name(resolve(CASES['uniform-global'])[0]) + '.region'
    assert (tmp_path / name / f'{name}.grid.nc').exists()
    assert not (tmp_path / name / f'{name}.report.html').exists()
    assert not (tmp_path / name / f'{name}.png').exists()


@needs_mesh_tools
def test_steep_transition_is_flagged(tmp_path):
    """A transition where neighbouring target sizes differ by ~1.3-1.4x is reported."""
    from vtxmpasmeshes import mesh_checks as mc
    params = dict(CASES['ellipse-global'], delta=1)
    proc = run_create(params, tmp_path)
    assert proc.returncode == 0          # a warning, not an error
    assert '[WARN] smooth grading' in proc.stdout
    p, _ = resolve(params)
    name = mesh_name(p) + '.region'
    results = mc.check_mesh(str(tmp_path / name / f'{name}.grid.nc'), p)
    assert [r.name for r in results if not r.ok] == ['smooth grading']


@needs_mesh_tools
def test_input_file_has_one_file_name_and_it_is_the_last_mesh(built):
    lines = built['_input_file'].read_text().splitlines()
    names = [l for l in lines if l.startswith('file_name=')]
    assert names == [f"file_name={built['_last']}"]
    assert lines[0] == '# test input file'


def test_names_are_exact_and_distinct():
    names = {mesh_name(resolve(p)[0]) for p in CASES.values()}
    assert len(names) == len(CASES)
    p, _ = resolve(dict(CENTER, refinement='circle', inner_radius=300, outer_radius=400,
                        high_res=1.5, low_res=5, cut='circle'))
    name = mesh_name(p)
    assert 'hres_1.5_' in name and 'lres_5_' in name and 'lat_-22.7_' in name
    p2, _ = resolve(dict(p, high_res=2.5))
    assert mesh_name(p2) != name


# --------------------------------------------------------------------------
# Input handling (fast, no JIGSAW)
# --------------------------------------------------------------------------
def test_missing_keys_are_listed_with_their_meaning():
    with pytest.raises(MeshConfigError) as err:
        resolve(dict(CENTER, refinement='circle', cut='circle', high_res=3))
    msg = str(err.value)
    for key in ('low_res', 'inner_radius', 'outer_radius'):
        assert key in msg
    with pytest.raises(MeshConfigError, match='cut_radius'):
        resolve(dict(CENTER, refinement='uniform', high_res=3, cut='circle'))
    with pytest.raises(MeshConfigError, match='cut_a_km'):
        resolve(dict(CENTER, **CIRCLE, cut='ellipse'))


def test_unused_keys_are_reported_not_rejected():
    p, ignored = resolve(dict(CENTER, refinement='uniform', high_res=3, cut='global',
                              inner_radius=10, a_km=5, cut_radius=9))
    assert ignored == ['a_km', 'cut_radius', 'inner_radius']
    assert 'inner_radius' not in p


def test_defaults_follow_the_refinement():
    p, _ = resolve(dict(CENTER, **CIRCLE, cut='circle'))
    assert p['cut_radius'] == CIRCLE['outer_radius'] and p['n_layers'] == 8 and p['N'] == 1
    p, _ = resolve(dict(CENTER, **ELLIPSE, cut='ellipse'))
    assert p['cut_a_km'] == pytest.approx(1.5 * 600)
    assert p['cut_b_km'] == pytest.approx(1.5 * 400)
    assert p['cut_angle'] == 30


def test_old_grid_type_keys_explain_the_replacement():
    with pytest.raises(MeshConfigError, match='refinement=circle, cut=circle'):
        resolve(dict(CENTER, grid_type='doughnut', do_regional='y'))
    with pytest.raises(MeshConfigError, match='refinement=uniform, cut=global'):
        resolve(dict(CENTER, grid_type='constant', do_regional='n'))


@pytest.mark.parametrize('bad, match', [
    (dict(n_layers=7), 'n_layers'),
    (dict(high_res=100), 'high_res must be smaller than low_res'),
    (dict(inner_radius=700), 'inner_radius must be smaller'),
    (dict(refinement='doughnut'), 'refinement must be one of'),
    (dict(cut='square'), 'cut must be one of'),
    (dict(N=0), 'N must be a positive integer'),
])
def test_invalid_inputs_are_rejected(bad, match):
    params = dict(CENTER, **CIRCLE, cut='circle')
    params.update(bad)
    with pytest.raises(MeshConfigError, match=match):
        resolve(params)


def test_cut_through_the_transition_is_rejected_with_minimum_size():
    p, _ = resolve(dict(CENTER, **CIRCLE, cut='circle', cut_radius=450))
    msg = geo.check_cut_encloses_refinement(p, 80)
    assert msg and 'cut_radius >= 600' in msg
    p, _ = resolve(dict(CENTER, **ELLIPSE, cut='ellipse', cut_a_km=700, cut_b_km=500))
    msg = geo.check_cut_encloses_refinement(p, 80)
    assert msg and 'scale the cut ellipse' in msg
    p, _ = resolve(dict(CENTER, **ELLIPSE, cut='circle', cut_radius=1000))
    assert geo.check_cut_encloses_refinement(p, 80) is None


def test_cli_stops_before_jigsaw_on_bad_input(tmp_path):
    proc = run_create(dict(CENTER, **CIRCLE, cut='circle', cut_radius=450), tmp_path)
    assert proc.returncode == 1 and 'cut passes through' in proc.stderr
    proc = run_create(dict(CENTER, refinement='circle', cut='circle', high_res=3), tmp_path)
    assert proc.returncode == 2 and 'missing inputs' in proc.stderr
    assert not list(tmp_path.glob('*.region'))


def test_ellipse_global_map_does_not_depend_on_cut_settings():
    base = dict(CENTER, **ELLIPSE, cut='global')
    p1, _ = resolve(base)
    p2, _ = resolve(dict(base, cut_radius=999, cut_a_km=1, n_layers=12))
    assert p1 == p2
    lat, lon = np.meshgrid(np.linspace(-60, 20, 50), np.linspace(-90, 0, 60))
    np.testing.assert_array_equal(geo.target_resolution(lat, lon, p1, 80, 8),
                                  geo.refinement_resolution(lat, lon, p1))


def test_ellipse_cut_boundary_matches_the_resolution_map():
    """The polygon given to MPAS-Limited-Area lies exactly on the cut ellipse."""
    for lat0, lon0 in ((-22.7, -45.0), (55.0, 178.0)):
        p, _ = resolve(dict(ELLIPSE, lat=lat0, lon=lon0, cut='ellipse'))
        blat, blon = geo.cut_boundary(p)
        d = geo.distance_beyond_cut(blat, blon, p)
        assert np.max(np.abs(d)) < 1.0
        assert np.all(np.abs(blon) <= 180)
        res = geo.refinement_resolution(blat, blon, p)
        np.testing.assert_allclose(res, p['low_res'], rtol=1e-9)


def test_ring_and_far_field_of_target_resolution():
    p, _ = resolve(dict(CENTER, **CIRCLE, cut='circle'))
    dist = np.array([0, 300, 450, 600, 900, 1240, 1300, 3000])
    lat, lon = geo.destination(p['lat'], p['lon'], dist, 45.0)
    t = geo.target_resolution(lat, lon, p, 80, 8)
    np.testing.assert_allclose(t[:4], [40, 40, 60, 80], rtol=1e-6)
    np.testing.assert_allclose(t[4:6], 80, rtol=1e-6)          # ring: 600..1240 km
    assert t[6] > 80 and t[7] <= geo.FAR_FIELD_MAX_RES        # far field


# --------------------------------------------------------------------------
# Bash scripts
# --------------------------------------------------------------------------
def bash(cmd, **kw):
    return subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, **kw)


def test_read_mesh_inputs_last_value_comments_and_crlf(tmp_path):
    f = tmp_path / 'in.txt'
    f.write_bytes(b'# comment\r\nlat=-22.5 # inline\r\n\r\n  N = 4 \r\n'
                  b'file_name=old.region\r\nfile_name=new.region\r\nPATH=/evil\r\nbad line\r\n'
                  b'lon=-45')                                  # no final newline
    proc = bash(f'source "{SCRIPTS}/utils.bash"; read_mesh_inputs "{f}"; '
                'echo "$lat|$N|$file_name|$lon|${#MESH_INPUT_KEYS[@]}"; echo "$PATH" | head -c 5')
    out = proc.stdout.splitlines()
    assert out[0] == '-22.5|4|new.region|-45|5'
    assert out[1] != '/evil'
    assert "invalid key 'PATH'" in proc.stderr and "without '='" in proc.stderr


@needs_mesh_tools
def test_create_mesh_bash_runs_from_anywhere_and_uses_the_new_name(tmp_path):
    """Second run with a changed input must not pick up the first mesh's name."""
    f = tmp_path / 'my_input.txt'
    f.write_text(f'lat=-22.7\nlon=-45\nrefinement=uniform\nhigh_res=240\ncut=global\n'
                 f'meshes_dir={tmp_path}/grids\nexp_dir=/old/path\ntypo_key=1\n')
    env = dict(os.environ, CONDA_DEFAULT_ENV='monan-grid', MESH_CONDA_ENV='monan-grid',
               PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ['PATH'])
    first = bash(f'bash "{SCRIPTS}/create_mesh.bash" "{f}"', cwd='/', env=env)
    assert first.returncode == 0, first.stdout[-2000:] + first.stderr[-2000:]
    assert "unknown key 'typo_key'" in first.stderr
    assert "'exp_dir' is no longer used" in first.stdout
    f.write_text(f.read_text().replace('high_res=240', 'high_res=300'))
    second = bash(f'bash "{SCRIPTS}/create_mesh.bash" "{f}"', cwd='/', env=env)
    assert second.returncode == 0, second.stdout[-2000:] + second.stderr[-2000:]
    names = [l for l in f.read_text().splitlines() if l.startswith('file_name=')]
    assert names == ['file_name=lat_-22.7_lon_-45_uniform_res_300_cut_global.region']
    made = sorted(p.name for p in (tmp_path / 'grids').glob('*.region'))
    assert made == ['lat_-22.7_lon_-45_uniform_res_240_cut_global.region',
                    'lat_-22.7_lon_-45_uniform_res_300_cut_global.region']
    assert not list((tmp_path / 'grids').glob('work_*'))
