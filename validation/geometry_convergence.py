"""Analytic PDE refinement on smooth nonuniform and physical partial-bottom grids."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cutcell import CutCellGrid, DiffusionModel


def study():
    studies = {}
    for kind in ('nonuniform', 'partial_bottom'):
        rows = []
        for n in (16, 32, 64):
            s = np.linspace(0, 1, n+1)
            if kind == 'nonuniform':
                grid = CutCellGrid(s + .15*np.sin(2*np.pi*s)/(2*np.pi))
            else:
                grid = CutCellGrid.partial_bottom(s, .31, minimum_fraction=.01)
            wet = grid.active
            lower = grid.faces[1:] - grid.volumes
            bottom = lower[wet][0]
            length = grid.faces[-1] - bottom
            k = np.pi / length
            # True cell averages over the active intervals, including the cut.
            initial = np.zeros(n)
            initial[wet] = .5 + .2*(np.sin(k*(grid.faces[1:][wet]-bottom))
                              - np.sin(k*(lower[wet]-bottom)))/(k*grid.volumes[wet])
            model = DiffusionModel(grid, initial)
            model.run_until(.03, max_dt=1)
            exact = .5 + (initial[wet]-.5)*np.exp(-.1*k*k*model.time)
            error = float(np.sqrt(np.dot(grid.volumes[wet], (model.state[wet]-exact)**2)/length))
            order = float(np.log2(rows[-1]['l2_error']/error)) if rows else None
            rows.append({'cells':n, 'l2_error':error, 'order':order,
                         'steps':model.iteration, 'numerical_bottom':float(bottom),
                         'cumulative_residual':model.history[-1].cumulative_residual})
        if not all(row['order'] > 1.8 for row in rows[1:]):
            raise AssertionError(f'{kind} convergence below required order: {rows}')
        studies[kind] = rows
    return studies


if __name__ == '__main__':
    print(json.dumps(study(), indent=2, allow_nan=False))
