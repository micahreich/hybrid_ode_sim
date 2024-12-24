from typing import Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from hybrid_ode_sim.simulation.base import BaseModel
from hybrid_ode_sim.simulation.rendering.base import PlotElement, PlotEnvironment
from hybrid_ode_sim.utils.logging_tools import Logger, LogLevel

from spatialmath.base import q2r

class Pose3D(PlotElement):
    def __init__(
        self,
        system: Optional[BaseModel],
        env: Optional[PlotEnvironment] = None,
    ):
        super().__init__(env, logging_level=LogLevel.ERROR)

        self.system = system
        _, _, self.history = system.history()

    def init_environment(self, env):
        super().init_environment(env)

        self.frame_x_vec = self.env.ax.quiver(0, 0, 0, 0, 0, 0, color="r", length=1.0, normalize=True)
        self.frame_y_vec = self.env.ax.quiver(0, 0, 0, 0, 0, 0, color="g", length=1.0, normalize=True)
        self.frame_z_vec = self.env.ax.quiver(0, 0, 0, 0, 0, 0, color="b", length=1.0, normalize=True)

    def update(self, t):
        if self.env is None:
            raise RuntimeError("The plot environment has not been initialized.")

        x = self.history(t)
        r = x[:3]
        q = x[3:]
        
        R = q2r(q)
        
        x_vec_endpt = r + R[:, 0]
        y_vec_endpt = r + R[:, 1]
        z_vec_endpt = r + R[:, 2]

        self.frame_x_vec.set_segments([[r, x_vec_endpt]])
        self.frame_y_vec.set_segments([[r, y_vec_endpt]])
        self.frame_z_vec.set_segments([[r, z_vec_endpt]])
