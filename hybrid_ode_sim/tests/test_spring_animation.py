from dataclasses import dataclass
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np

from hybrid_ode_sim.simulation.base import ContinuousTimeModel, DiscreteTimeModel
from hybrid_ode_sim.simulation.ode_solvers.adaptive_step_solver import RK23
from hybrid_ode_sim.simulation.ode_solvers.fixed_step_solver import RK4
from hybrid_ode_sim.simulation.rendering.base import PlotEnvironment
from hybrid_ode_sim.simulation.rendering.spring import BlockOnSpring
from hybrid_ode_sim.simulation.simulator import ModelGraph, Simulator, SimulationEnvironment


@dataclass
class SpringParams:
    k: float
    m: float
    b: float


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
    spring = Spring(y0=np.array([1.0, 0.0]), params=SpringParams(k=20, m=1, b=20))

    model_graph = ModelGraph(models=[spring])

    t_range = [0, 3]
    simulator = Simulator(model_graph, RK23(atol=1e-6, rtol=1e-3))

    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    ax.set_aspect("equal", adjustable="box")
    plot_env = PlotEnvironment(fig, ax, t_range, frame_rate=20, t_start=0.0).attach_element(
        BlockOnSpring(system=spring, wall_x=0.0)
    )

    env = SimulationEnvironment(simulator=simulator, plot_env=plot_env).run_simulation(
        t_range=t_range,
        realtime=False,
        show_time=True,
    )
