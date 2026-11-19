import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VTX = REPO / 'sources' / 'CGFD-USP-Create-Mesh' / 'vtx-mpas-meshes'
SCRIPTS = REPO / 'scripts'
sys.path.insert(0, str(VTX))

needs_mesh_tools = pytest.mark.skipif(
    shutil.which('jigsaw') is None or shutil.which('gpmetis') is None,
    reason='jigsaw/gpmetis not found: run inside the monan-grid conda env')
