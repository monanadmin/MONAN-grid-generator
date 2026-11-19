# MONAN-grid-generator

Variable-resolution horizontal meshes for MONAN (and MPAS-Atmosphere): fine cells where you need detail, coarser cells elsewhere, global or cut to a regional domain with the boundary layers MPAS expects. Every mesh is checked against the assumptions of the MPAS formulation and comes with a plot and a report.

**Start here: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)** — setup, a step-by-step tutorial, how to read the report, every input key, troubleshooting.

Quick start:

```bash
conda env create -f sources/CGFD-USP-Create-Mesh/envs/monan_grid_env.yml   # once
open tools/mesh_input_designer.html                                        # design the mesh on a map (optional)
bash scripts/create_mesh.bash my_mesh.txt                                  # build, check, plot and report
```
