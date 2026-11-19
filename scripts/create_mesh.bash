#!/bin/bash
#-----------------------------------------------------------------------------#
# Create an MPAS/MONAN mesh from a mesh input file.
#
# Usage:  bash create_mesh.bash [mesh_input_file]
#         (default: mesh_input_file.txt next to this script)
#
# The input file only needs the keys of the chosen refinement and cut; see the
# reference scripts/mesh_input_file.txt for every key and its meaning.
# Meshes are written to <repo>/grids/<name>.region/ unless meshes_dir is set.
# Requires the conda env monan-grid (sources/CGFD-USP-Create-Mesh/envs/
# monan_grid_env.yml); set MESH_CONDA_ENV to use another env.
#-----------------------------------------------------------------------------#
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$SCRIPT_DIR")
source "$SCRIPT_DIR/utils.bash"

INPUT_FILE=${1:-$SCRIPT_DIR/mesh_input_file.txt}
INPUT_FILE="$(cd "$(dirname "$INPUT_FILE")" && pwd)/$(basename "$INPUT_FILE")"

# Defaults (the input file may override them)
vtx_mpas_meshes_dir=$REPO/sources/CGFD-USP-Create-Mesh/vtx-mpas-meshes
meshes_dir=$REPO/grids

echo "Reading $INPUT_FILE"
read_mesh_inputs "$INPUT_FILE"

# Keys passed on to Python when present in the input file
MESH_KEYS=(lat lon refinement cut high_res low_res inner_radius outer_radius
           a_km b_km angle delta cut_radius cut_a_km cut_b_km cut_angle
           n_layers N grid_type do_regional)
OTHER_KEYS=(vtx_mpas_meshes_dir meshes_dir file_name)
OLD_KEYS=(exp_dir)

args=()
for key in "${MESH_KEYS[@]}"; do
    if [[ -n "${!key:-}" ]]; then
        args+=("--$key" "${!key}")
    fi
done
for key in "${MESH_INPUT_KEYS[@]}"; do
    if [[ " ${OLD_KEYS[*]} " == *" $key "* ]]; then
        echo "Note: '$key' is no longer used and is ignored."
    elif [[ " ${MESH_KEYS[*]} ${OTHER_KEYS[*]} " != *" $key "* ]]; then
        echo "WARNING: unknown key '$key' in $INPUT_FILE (typo?) - ignored." >&2
    fi
done

mkdir -p "$meshes_dir"

# Activate the conda environment (provides python, jigsaw and gpmetis)
CONDA_ENV=${MESH_CONDA_ENV:-monan-grid}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

python3 "$vtx_mpas_meshes_dir/create_regional_mesh.py" \
    --input_file "$INPUT_FILE" --meshes_dir "$meshes_dir" "${args[@]}"
