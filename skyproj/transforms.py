import matplotlib.transforms
from matplotlib.path import Path
import numpy as np
from pyproj import Geod

from .projections import PlateCarree, RADIUS
from .utils import wrap_values

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
        self._geod = Geod(a=RADIUS)

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

        if self._inverse:
            # Just send this upstream if we're not computing geodesics.
            return super().transform_path_non_affine(path)

        lonlats = []
        codes = []

        last_vertex = None
        first_vertex = None
        is_polygon = False
        for vertex, code in path.iter_segments():
            if last_vertex is None or code == Path.MOVETO:
                lonlats.extend([(vertex[0], vertex[1])])
                codes.append(Path.MOVETO)
                last_vertex = vertex
                first_vertex = vertex
            elif code in (Path.LINETO, Path.CLOSEPOLY, None):
                # Connect the last vertex
                lonlats_step = self._geod.npts(last_vertex[0], last_vertex[1],
                                               vertex[0], vertex[1], self._nsamp + 1,
                                               initial_idx=1, terminus_idx=0)
                lonlats.extend(lonlats_step)
                if code == Path.CLOSEPOLY:
                    is_polygon = True
                codes.extend([Path.LINETO]*len(lonlats_step))
                last_vertex = vertex
            else:
                raise ValueError("Unsupported code type %d" % (code))

        lonlats = np.array(lonlats)
        codes = np.array(codes)

        # Normalize range
        lonlats[:, 0] = wrap_values(lonlats[:, 0])

        # Cut into segments that wrap around
        delta = lonlats[: -1, 0] - lonlats[1:, 0]
        cut, = np.where(np.abs(delta) > 180.)

        if cut.size > 0:
            # Modify path to hit the edge and jump to the other side.
            insertions = []
            insertions_locs = []
            insertions_codes = []
            for c in cut:
                if lonlats[c, 0] > 0:
                    insertions.extend([(180. - 1e-5, lonlats[c, 1])])
                    insertions.extend([(-180., lonlats[c + 1, 1])])
                else:
                    insertions.extend([(-180., lonlats[c, 1])])
                    insertions.extend([(180. - 1e-5, lonlats[c + 1, 1])])
                insertions_locs.extend([c + 1, c + 1])
                insertions_codes.extend([Path.LINETO, Path.MOVETO])

            insertions = np.array(insertions)
            insertions[:, 0] = wrap_values(insertions[:, 0])

            lonlats = np.insert(lonlats, insertions_locs, insertions, axis=0)
            codes = np.insert(codes, insertions_locs, insertions_codes)

            # Roll to the first cut at edge.  Add 2 because of the point insertion.
            lonlats = np.roll(lonlats, -(cut[0] + 2), axis=0)
            codes = np.roll(codes, -(cut[0] + 2))

            # recompute cut locs after insertions
            delta = lonlats[: -1, 0] - lonlats[1:, 0]
            cut, = np.where(np.abs(delta) > 180.)

            if is_polygon:
                # At each cut, we need to follow along the edge to close the polygon
                poly_vertex_start = lonlats[0, :]
                insertions = []
                insertions_locs = []
                insertions_codes = []
                for c in cut:
                    lonlats_step = self._geod.npts(lonlats[c - 1, 0], lonlats[c - 1, 1],
                                                   poly_vertex_start[0], poly_vertex_start[1],
                                                   self._nsamp + 1,
                                                   initial_idx=1, terminus_idx=0)
                    insertions.extend(lonlats_step)
                    insertions_locs.extend([c + 1]*len(lonlats_step))
                    insertions_codes.extend([Path.LINETO]*len(lonlats_step))
                    # Mark the next point as the start of the next polygon
                    poly_vertex_start = lonlats[c + 1, :]

                insertions = np.array(insertions)
                insertions[:, 0] = wrap_values(insertions[:, 0])

                lonlats = np.insert(lonlats, insertions_locs, insertions, axis=0)
                codes = np.insert(codes, insertions_locs, insertions_codes)

                # And the final line
                lonlats_step = self._geod.npts(lonlats[-1, 0], lonlats[-1, 1],
                                               poly_vertex_start[0], poly_vertex_start[1],
                                               self._nsamp + 1,
                                               initial_idx=1, terminus_idx=0)
                appendages = np.array(lonlats_step)
                appendages[:, 0] = wrap_values(appendages[:, 0])
                appendages_codes = [Path.LINETO]*len(lonlats_step)
                lonlats = np.append(lonlats, appendages, axis=0)
                codes = np.append(codes, appendages_codes)

                # And another recompute!
                delta = lonlats[: -1, 0] - lonlats[1:, 0]
                cut, = np.where(np.abs(delta) > 180.)

        new_vertices = self.target_proj.transform_points(self.source_proj,
                                                         lonlats[:, 0], lonlats[:, 1])

        # Need to clean up code into functions.
        # And need to test with other wraps, and somehow get that information in here...
        # but lon_0 is available, isn't it!  In the target_proj.
        # And then we're in business with wrapping filled polygons!

        new_path = Path(new_vertices, codes)

        return new_path
