import numpy as np
import matplotlib.collections

__all__ = ["SkyGridlines"]


class SkyGridlines(matplotlib.collections.LineCollection):
    def __init__(self, segments=[], grid_helper=None, **kwargs):
        super().__init__(segments, **kwargs)

        self.set_clip_on(True)
        # Note that this will not work unless you call
        # set_clip_box(axes.bbox) before drawing.

        self._grid_helper = grid_helper

    def draw(self, renderer):
        gridlines = self._grid_helper.get_gridlines()

        self.set_segments([np.transpose(line) for line in gridlines])

        super().draw(renderer)
