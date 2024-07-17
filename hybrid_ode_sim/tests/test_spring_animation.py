from dataclasses import dataclass
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np

from hybrid_ode_sim.simulation.base import (ContinuousTimeModel,
                                            DiscreteTimeModel)
from hybrid_ode_sim.simulation.ode_solvers.adaptive_step_solver import RK23
from hybrid_ode_sim.simulation.ode_solvers.fixed_step_solver import RK4
from hybrid_ode_sim.simulation.rendering.base import PlotEnvironment
from hybrid_ode_sim.simulation.rendering.spring import BlockOnSpring
from hybrid_ode_sim.simulation.simulator import ModelGraph, Simulator


@dataclass
class SpringParams:
    k: float
    m: float
    b: float


@dataclass
class PIDControllerParams:
    kP: float
    kD: float
    kI: float


class Spring(ContinuousTimeModel):
    def __init__(self, y0, params: SpringParams):
        super().__init__(y0, name="spring_mass_damper", params=params)

    def continuous_dynamics(self, t, y) -> np.ndarray:
        inputs = self.collect_inputs()
        r, v = y

        f_applied = 0.0  # inputs['pd_controller']
        return np.array(
            [v, (f_applied - self.params.b * v - self.params.k * r) / self.params.m]
        )


if __name__ == "__main__":
    spring = Spring(y0=np.array([1.0, 0.0]), params=SpringParams(k=100, m=1, b=5))

    model_graph = ModelGraph(models=[spring])

    t_range = [0, 3]
    simulator = Simulator(model_graph, RK23(atol=1e-6, rtol=1e-3))
    simulator.simulate(t_range=t_range)

    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    ax.set_aspect("equal", adjustable="box")

    env = PlotEnvironment(fig, ax, t_range, frame_rate=20, t_start=0.0)
    block_on_spring = BlockOnSpring(env, spring, wall_x=0.0)

    env.render(
        plot_elements=[block_on_spring],
        show_time=True,
        save=True,
        save_path="spring_animation.gif",
    )
