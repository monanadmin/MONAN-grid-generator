"""
Create an MPAS/MONAN mesh: global, or cut to a regional domain.

A mesh is defined by a refinement (uniform | circle | ellipse) and a cut
(global | circle | ellipse); see vtxmpasmeshes/mesh_config.py and
scripts/mesh_input_file.txt for all parameters. Usually called by
scripts/create_mesh.bash, which passes only the keys present in the input file.

Outputs go to <meshes_dir>/<name>.region/:
  <name>.region.grid.nc               mesh for the MPAS workflow
  <name>.region.graph.info            graph file
  <name>.region.graph.info.part.<N>   partition file (when N > 1)
  <name>.region_for_plotting.grid.nc  mesh with the generation parameters as attributes
  mesh input file copy
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from vtxmpasmeshes.mesh_config import (ALL_KEYS, DESCRIPTIONS, MPAS_N_LAYERS,
                                       MeshConfigError, mesh_name, resolve)

DEFAULT_MESHES_DIR = Path(__file__).resolve().parents[3] / 'grids'
STRING_KEYS = ('refinement', 'cut')
AUTO_HEADER = '# Automatic additions (written by create_regional_mesh.py)'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Create a global or regional MPAS/MONAN mesh.')
    parser.add_argument('--meshes_dir', default=str(DEFAULT_MESHES_DIR),
                        help='folder for the generated meshes (default: <repo>/grids)')
    parser.add_argument('--input_file', default=None,
                        help='mesh input file: file_name=<mesh name> is written '
                             'to it and a copy is stored with the mesh')
    parser.add_argument('--overwrite', action='store_true',
                        help='regenerate the mesh even if it already exists')
    parser.add_argument('--do_plots', action='store_true',
                        help='also plot the resolution map and the mesh')
    parser.add_argument('--keep_work_dir', action='store_true',
                        help='keep the temporary work folder (JIGSAW files, global mesh)')
    for key in ALL_KEYS:
        parser.add_argument(f'--{key}', type=str if key in STRING_KEYS else float,
                            default=None, help=DESCRIPTIONS[key])
    # Old options: accepted only to explain how to replace them.
    for key in ('grid_type', 'do_regional'):
        parser.add_argument(f'--{key}', type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def update_input_file(input_file, name):
    """Replace the file_name line of the input file (instead of appending)."""
    lines = Path(input_file).read_text().splitlines()
    lines = [l for l in lines
             if not l.strip().startswith('file_name=') and l.strip() != AUTO_HEADER
             and l.strip() != '# Automatic additions']
    while lines and not lines[-1].strip():
        lines.pop()
    lines += [AUTO_HEADER, f'file_name={name}']
    Path(input_file).write_text('\n'.join(lines) + '\n')


def partition(graph_file, nparts):
    """Partition with gpmetis; returns the partition file."""
    if shutil.which('gpmetis') is None:
        raise RuntimeError('gpmetis not found: activate the monan-grid conda env')
    subprocess.run(['gpmetis', '-minconn', '-contig', '-niter=200',
                    graph_file, str(nparts)], check=True)
    part = f'{graph_file}.part.{nparts}'
    if not os.path.isfile(part):
        raise RuntimeError(f'gpmetis did not write {part}')
    return part


def main(argv=None):
    args = parse_args(argv)
    raw = {k: getattr(args, k) for k in ALL_KEYS}
    raw.update(grid_type=args.grid_type, do_regional=args.do_regional)
    try:
        p, ignored = resolve(raw)
    except MeshConfigError as err:
        print(f'ERROR: {err}', file=sys.stderr)
        return 2
    if ignored:
        print(f"Note: not used for refinement={p['refinement']}, cut={p['cut']}: "
              f"{', '.join(ignored)}")
    if p['cut'] != 'global' and p['n_layers'] > MPAS_N_LAYERS:
        print(f"WARNING: n_layers={p['n_layers']} gives {p['n_layers'] - 1} boundary "
              f'layers; MPAS uses 7 (n_layers={MPAS_N_LAYERS}).')

    # Imported here so input errors are reported without loading the mesh stack.
    from vtxmpasmeshes.mesh_generator import generate_mesh
    from vtxmpasmeshes import mesh_checks as mc

    name = mesh_name(p) + '.region'
    meshes_dir = os.path.abspath(args.meshes_dir)
    out_dir = os.path.join(meshes_dir, name)
    final_grid = os.path.join(out_dir, name + '.grid.nc')
    print(f'Mesh name: {name}')

    if os.path.exists(final_grid) and not args.overwrite:
        print(f'Mesh already exists in {out_dir} -- nothing to be done '
              '(use --overwrite to regenerate).')
    else:
        os.makedirs(meshes_dir, exist_ok=True)
        work_dir = tempfile.mkdtemp(prefix='work_', dir=meshes_dir)
        print(f'Work folder: {work_dir}')
        try:
            paths = generate_mesh(p, os.path.join(work_dir, name + '.grid.nc'),
                                  work_dir, do_plots=args.do_plots)
            print('\n>> Checking the mesh')
            results = mc.check_mesh(paths['grid'], p, graph_file=paths['graph_info'])
            print(mc.report(results))
            if mc.has_errors(results):
                raise RuntimeError('the mesh failed the checks above; it was not '
                                   'partitioned or moved to the output folder')
            outputs = [paths['grid'], paths['graph_info'], paths['plotting']]
            if p['N'] > 1:
                part = partition(paths['graph_info'], p['N'])
                res = mc.check_partition_file(mc.Mesh(paths['grid']), part, p['N'])
                print(mc.report([res]))
                if not res.ok:
                    raise RuntimeError('the partition file is invalid')
                outputs.append(part)
            for plot in ('resolution.pdf', 'resolution_mesh.png'):
                if os.path.exists(os.path.join(work_dir, plot)):
                    outputs.append(os.path.join(work_dir, plot))
        except Exception as err:
            print(f'\nERROR: {err}', file=sys.stderr)
            print(f'Intermediate files kept in {work_dir}', file=sys.stderr)
            return 1
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        for f in outputs:
            shutil.move(f, os.path.join(out_dir, os.path.basename(f)))
        if args.keep_work_dir:
            print(f'Work folder kept: {work_dir}')
        else:
            shutil.rmtree(work_dir)

    if args.input_file:
        update_input_file(args.input_file, name)
        shutil.copy(args.input_file, out_dir)
        print(f'file_name={name} written to {args.input_file}')
    print(f'\nDone. Mesh files in {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
