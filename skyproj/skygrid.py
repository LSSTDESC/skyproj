import numpy as np
import matplotlib.collections

__all__ = ["SkyGridlines"]


class SkyGridlines(matplotlib.collections.LineCollection):
    def __init__(self, segments=[], grid_helper=None, **kwargs):
        super().__init__(segments, **kwargs)

        self._grid_helper = grid_helper

    def draw(self, renderer):
        # Code here to set segments.

        # Ah an update on the draw!
        # if self._grid_helper is not None:
        #     self._grid_helper.update_lim(self.axes)
        #     gl = self._grid_helper.get_gridlines(self._which, self._axis)
        #     self.set_segments([np.transpose(l) for l in gl])

        # self.set_segments(self._grid_helper.get_gridlines())

        import IPython
        IPython.embed()

        gridlines = self._grid_helper.get_gridlines()
        self.set_segments([np.transpose(line) for line in gridlines])

        super().draw(renderer)
