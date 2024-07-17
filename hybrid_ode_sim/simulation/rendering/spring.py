from typing import Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from hybrid_ode_sim.simulation.base import BaseModel
from hybrid_ode_sim.simulation.rendering.base import (PlotElement,
                                                      PlotEnvironment)
from hybrid_ode_sim.utils.logging_tools import Logger, LogLevel


class BlockOnSpring(PlotElement):
    def __init__(
        self, env: PlotEnvironment, system: Optional[BaseModel], wall_x: float = 0.0
    ):
        super().__init__(env, logging_level=LogLevel.ERROR)

        self.ts, self.ys, self.history = system.history()
        self.wall_x = wall_x

        self.block_width, self.block_height = 0.1, 0.1

        (self.spring_plot,) = self.env.ax.plot([], [], lw=1, color="gray", zorder=1)
        self.block_plot = patches.Rectangle(
            (self.ys[0, 0], 0.0),
            self.block_width,
            self.block_height,
            linewidth=1,
            color="red",
            zorder=2,
        )
        self.env.ax.add_patch(self.block_plot)

        # Plot the wall and ground
        max_x = max(self.ys[:, 0]) * 1.1
        self.env.ax.plot(
            [self.wall_x, self.wall_x],
            [0, 2 * self.block_height],
            color="black",
            lw=2,
            zorder=0,
        )
        self.env.ax.plot([self.wall_x, max_x], [0, 0], color="black", lw=2, zorder=0)

    def update(self, t):
        x, v = self.history(t)

        if x > self.wall_x:
            s = np.linspace(self.wall_x, x, 100)
        else:
            s = np.linspace(x, self.wall_x, 100)

        spring_vals = 0.5 * self.block_height * np.sin(s * 6 * np.pi / x) + (
            self.block_height / 2
        )

        self.spring_plot.set_data(s, spring_vals)
        self.block_plot.set_x(x)

        return self.spring_plot, self.block_plot
