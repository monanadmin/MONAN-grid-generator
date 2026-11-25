#!/usr/bin/env python3
"""
Check an existing MPAS/MONAN mesh with vtxmpasmeshes/mesh_checks.py.

The mesh does not need to come from this generator: any MPAS mesh file works
(x1.40962.grid.nc, a static.nc / init.nc, a regional mesh from MPAS-Limited-Area...).

Usage (inside the monan-grid conda env, from anywhere):
    python3 tools/check_mesh.py path/to/mesh.grid.nc
    python3 tools/check_mesh.py path/to/mesh_folder
    python3 tools/check_mesh.py mesh.grid.nc --graph_file mesh.graph.info
    python3 tools/check_mesh.py mesh.grid.nc --partition_file mesh.graph.info.part.64
    python3 tools/check_mesh.py mesh.grid.nc --input_file my_mesh.txt
    python3 tools/check_mesh.py mesh.grid.nc --strict
    python3 tools/check_mesh.py mesh.grid.nc --out_dir reports/   # where the report goes
    python3 tools/check_mesh.py mesh.grid.nc --no_report          # checks only, no plots

What is checked:
  - always: the MPAS Voronoi structure (geometry, connectivity, Voronoi property,
    areas, dvEdge/dcEdge, near-centroidal cells, smooth grading) and, for regional
    meshes (bdyMaskCell present), the 7 boundary layers;
  - only if found or given: the graph file, the partition file;
  - only with a mesh input file of this generator: whether the cell sizes follow
    the requested resolution map, the boundary layers sit at uniform resolution
    and the cut region is complete.
Anything that is not available is skipped and listed, never treated as an error.

Report (default): <name>.png (the mesh coloured by cell size) and <name>.report.html
(one self-contained page: check results with explanations, drawings of what the
checks test, distributions of cell size, quality and shapes, partition balance).
They are written next to the mesh, or to --out_dir.

Exit code: 0 = no errors, 1 = errors (or, with --strict, errors or warnings),
2 = bad arguments / not an MPAS mesh.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'sources' / 'CGFD-USP-Create-Mesh' / 'vtx-mpas-meshes'))

from vtxmpasmeshes.mesh_config import ALL_KEYS, MeshConfigError, resolve  # noqa: E402

STRING_KEYS = ('refinement', 'cut')


def die(msg):
    print(f'ERROR: {msg}', file=sys.stderr)
    sys.exit(2)


def read_input_file(path):
    """key=value lines, '#' comments, last value wins (same rules as scripts/utils.bash)."""
    raw = {}
    for line in Path(path).read_text(errors='replace').splitlines():
        line = line.split('#', 1)[0].strip()
        if '=' not in line:
            continue
        key, value = (s.strip() for s in line.split('=', 1))
        if key in ALL_KEYS and value:
            raw[key] = value if key in STRING_KEYS else float(value)
    return raw


def find_grid(target):
    target = Path(target).resolve()
    if target.is_file():
        return target
    if not target.is_dir():
        die(f'not found: {target}')
    ncs = [f for f in target.glob('*.nc') if '_for_plotting' not in f.name]
    grids = [f for f in ncs if f.name.endswith('.grid.nc')] or ncs
    if len(grids) != 1:
        die(f'expected one mesh (.nc) file in {target}, found '
            f'{[g.name for g in grids] or "none"}; give the file itself')
    return grids[0]


def companions(grid):
    """graph/partition/input files next to the grid, if it follows this generator's naming."""
    folder = grid.parent
    stem = grid.name
    for suffix in ('.grid.nc', '.static.nc', '.init.nc', '.nc'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    graph = folder / f'{stem}.graph.info'
    parts = sorted(folder.glob(f'{stem}.graph.info.part.*'))
    inputs = [f for f in folder.glob('*.txt')
              if 'refinement' in f.read_text(errors='replace')]
    return (graph if graph.exists() else None,
            parts[0] if len(parts) == 1 else None,
            inputs[0] if len(inputs) == 1 else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Check an existing MPAS/MONAN mesh.')
    ap.add_argument('mesh', help='MPAS mesh file (.nc), or a folder with one mesh file')
    ap.add_argument('--graph_file', help='graph.info file to check (default: <name>.graph.info '
                    'next to the mesh, if present)')
    ap.add_argument('--partition_file', help='partition file <...>.part.<N> to check '
                    '(default: the one next to the mesh, if present)')
    ap.add_argument('--input_file', help='mesh input file of this generator, to also check '
                    'the mesh against the requested resolution map')
    ap.add_argument('--no_auto', action='store_true',
                    help='do not look for graph/partition/input files next to the mesh')
    ap.add_argument('--strict', action='store_true',
                    help='treat warnings as failures (as tests/ does)')
    ap.add_argument('--out_dir', help='folder for the report (default: the mesh folder)')
    ap.add_argument('--no_report', action='store_true',
                    help='only print the check results; no mesh plot or HTML report')
    args = ap.parse_args(argv)

    grid = find_grid(args.mesh)
    graph, part, input_file = (None, None, None) if args.no_auto else companions(grid)
    graph = Path(args.graph_file) if args.graph_file else graph
    part = Path(args.partition_file) if args.partition_file else part
    input_file = Path(args.input_file) if args.input_file else input_file
    for f in (graph, part, input_file):
        if f is not None and not f.is_file():
            die(f'not found: {f}')

    p = None
    if input_file:
        try:
            p, _ = resolve(read_input_file(input_file))
        except MeshConfigError as err:
            die(f'{input_file}: {err}')
    nparts = None
    if part:
        try:
            nparts = int(part.name.rsplit('.', 1)[1])
        except ValueError:
            die(f'partition file name must end in .part.<N>: {part.name}')

    from vtxmpasmeshes import mesh_checks as mc
    try:
        mesh = mc.Mesh(str(grid))
    except ValueError as err:
        die(str(err))
    except OSError as err:
        die(f'cannot read {grid} as netCDF ({err})')

    kind = 'regional (bdyMaskCell present)' if mesh.regional else 'global (no bdyMaskCell)'
    print(f'Mesh:        {grid}')
    print(f'             {mesh.nCells} cells, {kind}')
    print(f'Graph file:  {graph or "- not checked"}')
    print(f'Partition:   {part or "- not checked"}')
    print(f'Input file:  {input_file or "- not given: resolution-map checks skipped"}')
    print()

    results = mc.check_mesh(str(grid), p, graph_file=str(graph) if graph else None,
                            partition_file=str(part) if part else None, nparts=nparts)
    print(mc.report(results))

    n_err = sum(not r.ok and r.severity == 'error' for r in results)
    n_warn = sum(not r.ok and r.severity != 'error' for r in results)
    print(f'\n{len(results)} checks: {len(results) - n_err - n_warn} OK, '
          f'{n_warn} warnings, {n_err} errors')
    if not mesh.regional and any(r.name in ('connectivity', 'topology (V - E + F)') and not r.ok
                                 for r in results):
        print('Note: the mesh has no bdyMaskCell, so it was checked as a GLOBAL mesh. If it is '
              'a regional mesh, it lacks the boundary-layer masks MPAS needs '
              '(create them with MPAS-Limited-Area).')
    if not args.no_report:
        write_report(grid, p, results, graph, part, nparts, args.out_dir)
    return 1 if n_err or (args.strict and n_warn) else 0


def write_report(grid, p, results, graph, part, nparts, out_dir):
    """Mesh plot + HTML report; a failure here only warns (the checks above still stand)."""
    out = Path(out_dir) if out_dir else grid.parent
    name = grid.name[:-3] if grid.name.endswith('.nc') else grid.name
    if name.endswith('.grid'):
        name = name[:-5]
    print('\nPlotting the mesh and writing the report...')
    try:
        out.mkdir(parents=True, exist_ok=True)
        from vtxmpasmeshes.mesh_report import write_report as _write
        png, html = _write(str(grid), p, results, str(out), name,
                           graph_file=str(graph) if graph else None,
                           part_file=str(part) if part else None, nparts=nparts or 1,
                           made_by='tools/check_mesh.py')
    except Exception as err:
        print(f'WARNING: the report could not be made ({type(err).__name__}: {err}). '
              'The check results above are still valid.', file=sys.stderr)
        return
    print(f'Mesh plot: {png}\nReport:    {html}')


if __name__ == '__main__':
    sys.exit(main())
