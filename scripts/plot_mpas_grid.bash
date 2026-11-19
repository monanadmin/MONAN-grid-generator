#!/bin/bash
#-----------------------------------------------------------------------------#
# Plot an MPAS grid: cells coloured by their resolution (km).
#
# Usage:  bash plot_mpas_grid.bash GRID_FILE [OUTPUT_PNG] [extra options]
#   GRID_FILE   MPAS grid (e.g. grids/<name>.region/<name>.region.grid.nc)
#   OUTPUT_PNG  default: GRID_FILE with .png instead of .nc
#   extra options are passed to mpas_plot_grid.py, e.g.
#     -lat_min -60 -lat_max 25 -lon_min -110 -lon_max -10   (zoom)
#     -vmin 5 -vmax 15                                       (colour range)
#     -nc 8                                                  (parallel chunks)
#
# Requires the conda env monan-grid (set MESH_CONDA_ENV to use another env).
#-----------------------------------------------------------------------------#
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$SCRIPT_DIR")

if [[ $# -lt 1 ]]; then
    sed -n '3,13p' "${BASH_SOURCE[0]}"
    exit 1
fi
GFILEPATH=$1; shift
if [[ $# -gt 0 && "$1" != -* ]]; then
    POSTFILEPATH=$1; shift
else
    POSTFILEPATH=${GFILEPATH%.nc}.png
fi

CONDA_ENV=${MESH_CONDA_ENV:-monan-grid}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

python3 "$REPO/sources/CGFD-USP-Post-Proc/mpas_plot_grid.py" \
    -g "$GFILEPATH" -o "$POSTFILEPATH" "$@"
echo "Plot written to $POSTFILEPATH"
