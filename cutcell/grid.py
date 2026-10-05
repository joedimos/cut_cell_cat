"""Cell/face geometry; see docs/OCEANANIGANS_REFERENCE.md for provenance."""

from dataclasses import dataclass
import numpy as np


def vector(value, size, name):
    array = np.array(value, dtype=float, copy=True)
    if array.ndim == 0:
        array = np.full(size, float(array))
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array


@dataclass(frozen=True, init=False)
class CutCellGrid:
    """Unit-cross-section control volumes with independent face apertures.

    Zero-volume cells are solid and must have closed adjacent faces. Fractions
    alone do not locate a physical cut; provide centroids or use partial_bottom.
    Generic fractional geometry is a conservative network, not a 3-D mesh.
    """

    faces: np.ndarray
    centers: np.ndarray
    volume_fractions: np.ndarray
    apertures: np.ndarray
    volumes: np.ndarray

    def __init__(self, faces, volume_fractions=1.0, apertures=None, centers=None):
        faces = np.array(faces, dtype=float, copy=True)
        if (faces.ndim != 1 or len(faces) < 2 or not np.all(np.isfinite(faces))
                or np.any(np.diff(faces) <= 0)):
            raise ValueError("faces must be finite and strictly increasing")
        if not np.all(np.isfinite(np.diff(faces))):
            raise ValueError("cell widths must be finite")
        n = len(faces) - 1
        fractions = vector(volume_fractions, n, "volume_fractions")
        if np.any((fractions < 0) | (fractions > 1)) or not np.any(fractions > 0):
            raise ValueError("fractions must be in [0, 1] with at least one wet cell")
        active = fractions > 0
        allowed = np.r_[active[0], active[:-1] & active[1:], active[-1]]
        aperture = allowed.astype(float) if apertures is None else vector(apertures, n + 1, "apertures")
        if np.any((aperture < 0) | (aperture > 1)) or np.any(aperture[~allowed] != 0):
            raise ValueError("apertures must be in [0, 1] and closed next to solid cells")
        centers = (faces[:-1] / 2 + faces[1:] / 2 if centers is None
                   else vector(centers, n, "centers"))
        if np.any(centers <= faces[:-1]) or np.any(centers >= faces[1:]):
            raise ValueError("centers must lie strictly inside their underlying cells")
        volumes = fractions * np.diff(faces)
        if not np.all(np.isfinite(volumes)) or np.any(volumes[active] <= 0):
            raise ValueError("active volumes must be positive and representable")
        for name, data in (("faces", faces), ("centers", centers),
                           ("volume_fractions", fractions), ("apertures", aperture),
                           ("volumes", volumes)):
            data.setflags(write=False)
            object.__setattr__(self, name, data)

    @property
    def size(self):
        return len(self.volumes)

    @property
    def active(self):
        return self.volumes > 0

    @classmethod
    def uniform(cls, size, length=1.0, **kwargs):
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("size must be a positive integer")
        if not np.isfinite(length) or length <= 0:
            raise ValueError("length must be positive and finite")
        return cls(np.linspace(0, length, size + 1), **kwargs)

    @classmethod
    def partial_bottom(cls, faces, bottom, minimum_fraction=0.2):
        """Fit a bottom cut, enlarging tiny wet cells as PartialCellBottom does.

        The numerical bottom can be lower than the requested bottom. Its volume
        and centroid are adjusted together. Cells below it are impermeable.
        """
        base = cls(faces)
        if not np.isfinite(bottom) or not 0 < minimum_fraction <= 1:
            raise ValueError("finite bottom and minimum_fraction in (0, 1] required")
        if bottom < base.faces[0] or bottom >= base.faces[-1]:
            raise ValueError("bottom must lie inside the domain, below its top")
        widths = np.diff(base.faces)
        heights = np.maximum(0, base.faces[1:] - np.maximum(base.faces[:-1], bottom))
        active = heights > 0
        heights[active] = np.maximum(heights[active], minimum_fraction * widths[active])
        centers = base.centers.copy()
        centers[active] = base.faces[1:][active] - heights[active] / 2
        apertures = np.r_[False, active[:-1] & active[1:], active[-1]].astype(float)
        return cls(base.faces, heights / widths, apertures=apertures, centers=centers)
