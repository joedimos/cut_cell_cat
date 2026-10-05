"""Reproducible analytic refinement study; no Oceananigans runtime required."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cutcell import CutCellGrid, DiffusionModel
from cutcell.reference import OCEANANIGANS_COMMIT


def study():
    rows = []
    previous = None
    for n in (16, 32, 64, 128):
        grid = CutCellGrid.uniform(n)
        initial = .5 + .2 * np.cos(np.pi * grid.centers) * np.sinc(1/(2*n))
        model = DiffusionModel(grid, initial, .1)
        model.run_until(.03, max_dt=1)
        exact = .5 + (initial - .5) * np.exp(-.1 * np.pi**2 * model.time)
        error = float(np.sqrt(np.dot(grid.volumes, (model.state-exact)**2)))
        order = float(np.log2(previous / error)) if previous is not None else None
        rows.append({'cells':n, 'steps':model.iteration, 'l2_error':error, 'order':order,
                     'mass_drift': model.mass()-model.mass(initial),
                     'maximum_step_residual':max(abs(b.residual) for b in model.history)})
        previous = error
    if not all(row['order'] > 1.9 for row in rows[1:]):
        raise AssertionError(f'spatial convergence failed: {rows}')
    return {'reference_commit':OCEANANIGANS_COMMIT, 'comparison':'analytic_cell_averages', 'rows':rows}


if __name__ == '__main__':
    print(json.dumps(study(), indent=2, allow_nan=False))
