import matplotlib.transforms
import numpy as np

from .projections import PlateCarree

__all__ = ["SkyTransform"]


class SkyTransform(matplotlib.transforms.Transform):
    input_dims = 2
    output_dims = 2
    is_separable = False
    has_inverse = True

    def __init__(self, proj, inverse=False):
        self._inverse = inverse
        if not inverse:
            self.source_proj = PlateCarree()
            self.target_proj = proj
        else:
            self.source_proj = proj
            self.target_proj = PlateCarree()

        # Number of geodesic sub-samples for paths.
        self._nsamp = 10

        super().__init__()

    def inverted(self):
        if not self._inverse:
            # Return the inverse
            return SkyTransform(self.target_proj, inverse=True)
        else:
            # Re-invert it
            return SkyTransform(self.source_proj, inverse=False)

    def transform_non_affine(self, xy):
        res = self.target_proj.transform_points(self.source_proj,
                                                xy[:, 0], xy[:, 1])

        return res

    def transform_path_non_affine(self, path):
        # For each point in the path, we take 10 geodesic sub-samples
        # Only support MOVETO, LINETO
        # codes is None it's MOVETO then LINETO ...

        # FIXME
        return super().transform_path_non_affine(path)
