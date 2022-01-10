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

        self._lon_0 = self.target_proj.proj4_params['lon_0']
        self._wrap = (self._lon_0 + 180.) % 360.

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
        if self._inverse:
            # Just send this upstream if we're not computing geodesics.
            return super().transform_path_non_affine(path)

        lonlats = []
        codes = []

        last_vertex = None
        is_polygon = False
        # Make sure we don't simplify the path segments, which gets coordinate
        # systems all messed up.
        for vertex, code in path.iter_segments(simplify=False):
            if last_vertex is None or code == Path.MOVETO:
                lonlats.extend([(vertex[0], vertex[1])])
                codes.append(Path.MOVETO)
                last_vertex = vertex
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
        lonlats[:, 0] = wrap_values(lonlats[:, 0], wrap=self._wrap)

        # Cut into segments that wrap around
        cuts = self._compute_cuts(lonlats[:, 0])

        if cuts.size > 0:
            # Modify path to hit the edge and jump to the other side.
            lonlats, codes = self._insert_jumps(lonlats, codes, cuts)

            if is_polygon:
                # If this is a polygon we need to add extra edges.
                lonlats, codes = self._complete_cut_polygons(lonlats, codes)

        vertices_xform = self.target_proj.transform_points(self.source_proj,
                                                           lonlats[:, 0], lonlats[:, 1])

        new_path = Path(vertices_xform, codes)

        return new_path

    def _compute_cuts(self, lons):
        """Compute cut locations from a list of longitudes.

        Parameters
        ----------
        lons : `np.ndarray`
            Array of longitude values.

        Returns
        -------
        cuts : `np.ndarray`
            Array of jump/cut locations.
        """
        cuts, = np.where(np.abs(lons[: -1] - lons[1:]) > 180.0)
        return cuts

    def _insert_jumps(self, lonlats, codes, cuts):
        """Insert jump points at edges.

        Parameters
        ----------
        lonlats : `np.ndarray`
            [N, 2] array of longitude/latitudes
        codes : `np.ndarray`
            Array of path codes.
        cuts : `np.ndarray`
            Array of cut locations

        Returns
        -------
        lonlats : `np.ndarray`
            New [M, 2] array of longitude/latitudes
        codes : `np.ndarray`
            New array of path codes.
        """
        lonlats_insert = []
        locs_insert = []
        codes_insert = []
        for c in cuts:
            jump_vals = [self._wrap - 1e-10, self._wrap - 360. + 1e-10]
            if lonlats[c, 0] < self._lon_0:
                # Switch the order if it's at the low end.
                jump_vals = jump_vals[::-1]

            lonlats_insert.extend([(jump_vals[0], lonlats[c, 1])])
            lonlats_insert.extend([(jump_vals[1], lonlats[c + 1, 1])])
            locs_insert.extend([c + 1]*2)
            codes_insert.extend([Path.LINETO, Path.MOVETO])

        lonlats = np.insert(lonlats, locs_insert, lonlats_insert, axis=0)
        codes = np.insert(codes, locs_insert, codes_insert)

        return lonlats, codes

    def _complete_cut_polygons(self, lonlats, codes):
        """Complete cut polygon shapes.

        Parameters
        ----------
        lonlats : `np.ndarray`
            [N, 2] array of lonlats.
        codes : `np.ndarray`
            Array of path codes.

        Returns
        -------
        lonlats : `np.ndarray`
            New [M, 2] array of longitude/latitudes
        codes : `np.ndarray`
            New array of path codes.
        """
        # First, we need to roll to the first cut at the edge.
        cuts = self._compute_cuts(lonlats[:, 0])
        lonlats = np.roll(lonlats, -(cuts[0] + 1), axis=0)
        codes = np.roll(codes, -(cuts[0] + 1))

        # Recompute cuts after roll
        cuts = self._compute_cuts(lonlats[:, 0])

        # We need to handle inserting and appending separately, apparently.
        poly_vertex_start = lonlats[0, :]

        if cuts.size > 0:
            lonlats_insert = []
            locs_insert = []
            codes_insert = []
            for c in cuts:
                lonlats_step = self._geod.npts(lonlats[c - 1, 0], lonlats[c - 1, 1],
                                               poly_vertex_start[0], poly_vertex_start[1],
                                               self._nsamp + 1,
                                               initial_idx=1, terminus_idx=0)
                lonlats_insert.extend(lonlats_step)
                locs_insert.extend([c + 1]*len(lonlats_step))
                codes_insert.extend([Path.LINETO]*len(lonlats_step))
                poly_vertex_start = lonlats[c + 1, :]

            lonlats_insert = np.array(lonlats_insert)
            lonlats_insert[:, 0] = wrap_values(lonlats_insert[:, 0], wrap=self._wrap)

            lonlats = np.insert(lonlats, locs_insert, lonlats_insert, axis=0)
            codes = np.insert(codes, locs_insert, codes_insert)

        # And the final connection
        lonlats_step = self._geod.npts(lonlats[-1, 0], lonlats[-1, 1],
                                       poly_vertex_start[0], poly_vertex_start[1],
                                       self._nsamp + 1,
                                       initial_idx=1, terminus_idx=0)
        lonlats_append = np.array(lonlats_step)
        lonlats_append[:, 0] = wrap_values(lonlats_append[:, 0], wrap=self._wrap)
        codes_append = [Path.LINETO]*len(lonlats_append)

        lonlats = np.append(lonlats, lonlats_append, axis=0)
        codes = np.append(codes, codes_append)

        return lonlats, codes
