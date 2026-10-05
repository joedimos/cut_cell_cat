"""Compare a reference produced by oceananigans_reference.jl to the Python core."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cutcell import CutCellGrid, DiffusionModel
from cutcell.reference import OCEANANIGANS_COMMIT


def compare(path):
    path = Path(path)
    if path.read_text().splitlines()[0] != '# Oceananigans commit=' + OCEANANIGANS_COMMIT:
        raise ValueError('reference must identify the pinned upstream commit')
    data = np.genfromtxt(path, delimiter=',', names=True, skip_header=1)
    if len(data) != 32:
        raise ValueError('expected the 32-cell reference problem')
    grid = CutCellGrid.uniform(32)
    initial = .5 + .2 * np.cos(np.pi * grid.centers) * np.sinc(1/64)
    model = DiffusionModel(grid, initial, .1)
    for _ in range(20):
        model.step(.001)
    np.testing.assert_allclose(data['z'], grid.centers, atol=1e-14, rtol=0)
    np.testing.assert_allclose(data['time'], model.time, atol=1e-14, rtol=0)
    np.testing.assert_allclose(data['c'], model.state, atol=1e-11, rtol=1e-10)
    # For this linear autonomous operator, both 3-stage order-3 methods have
    # the same degree-three stability polynomial, despite different stages.
    print(f'Pinned reference comparison passed; max error={max(abs(data["c"]-model.state)):.3e}')


if __name__ == '__main__':
    compare(sys.argv[1])
