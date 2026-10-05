"""Conservative one-dimensional scalar transport on bounded cut-cell grids."""

from .grid import CutCellGrid
from .operators import BoundaryCondition, DiffusionOperator
from .model import DiffusionModel

__all__ = ["CutCellGrid", "BoundaryCondition", "DiffusionOperator", "DiffusionModel"]
