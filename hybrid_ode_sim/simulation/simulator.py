import heapq
import itertools
import time
from multiprocessing import Process, Pipe, Event, Manager, Value

from enum import Enum
from fractions import Fraction
from functools import reduce
from typing import Any, List, Optional, Tuple, Type, Union

import numpy as np

from hybrid_ode_sim.simulation.base import (
    BaseModel,
    ContinuousTimeModel,
    DiscreteTimeModel,
)
from hybrid_ode_sim.simulation.ode_solvers.solver_base import Integrator
from hybrid_ode_sim.utils.logging_tools import Logger, LogLevel


class ModelGraph:
    def __init__(self, models: List[BaseModel]):
        model_names = set()

        # Sanity check the models to ensure no duplicates are found
        for model in model_names:
            if model.name in model_names:
                raise ValueError(
                    f"Model name '{model.name}' is not unique in the model graph!"
                )
            model_names.add(model.name)

        self.enumerated_models = {model: i for (i, model) in enumerate(models)}
        self.adj = {model: [] for model in models}

        for model in models:
            self.adj[model] += model.children_models
            del (
                model.children_models
            )  # Remove children_models attribute since we don't need it anymore
        self.eval_order = self._topo_sort()

        self.continuous_time_models = list(
            filter(
                lambda model: isinstance(model, ContinuousTimeModel),
                self.enumerated_models.keys(),
            )
        )
        self.discrete_time_models = list(
            filter(
                lambda model: isinstance(model, DiscreteTimeModel),
                self.enumerated_models.keys(),
            )
        )

    def initialize_history_buffers(self, proc_manager=None):
        for model in self.enumerated_models.keys():
            model.initialize_history_buffers(proc_manager=proc_manager)

    def _topo_sort(self):
        def _topo_sort_recurse(v, visited, stack):
            visited[self.enumerated_models[v]] = True
            for m in self.adj[v]:
                if not visited[self.enumerated_models[m]]:
                    _topo_sort_recurse(m, visited, stack)
            stack.insert(0, v)

        visited = [False] * len(self.adj)
        stack = []

        # Return a queue of model instances to run, topologically sorted
        for model, i in self.enumerated_models.items():
            if not visited[i]:
                _topo_sort_recurse(model, visited, stack)

        return stack


class SimulationClock:
    def __init__(self, start_time=None) -> None:
        self.start_time = start_time

    def __call__(self) -> float:
        if self.start_time is not None:
            return time.perf_counter() - self.start_time
        else:
            return 0.0

    def reset(self):
        self.start_time = time.perf_counter()


class Simulator:
    def __init__(
        self,
        model_graph: ModelGraph,
        integrator: Integrator,
        logging_level=LogLevel.INFO,
    ):
        self.model_graph = model_graph
        self.logger = Logger(logging_level, f"{self.__class__.__name__}")

        self.integrator = integrator
        self.integrator.register_models(self.model_graph.continuous_time_models)

        self._t = None
        self._terminated_flag = False
        self._begin_flag = False

    def _round_up_to_multiple(self, A, B):
        if B == 0:
            raise ValueError("B cannot be zero.")
        # Compute the ceiling of A / B, ensuring exact integer arithmetic
        multiplier = -(-A // B)  # This is the ceiling of A / B
        # Return the result as the exact multiple of B
        return multiplier * B

    def _collect_sim_timesteps(
        self, t_range: Tuple[Fraction, Fraction]
    ) -> List[Fraction]:
        """Get simulation timesteps at which DiscreteTimeModel dynamics functions must be called

        Args:
            t_range (Tuple[Fraction, Fraction]): _description_

        Returns:
            List[Fraction]: _description_
        """

        simulation_timesteps = iter([])

        for model in self.model_graph.discrete_time_models:
            n_steps = int(
                model.sample_rate
                * self._round_up_to_multiple(
                    t_range[1] - t_range[0], Fraction(1, model.sample_rate)
                )
            )
            model_timesteps = [Fraction(n, model.sample_rate) for n in range(n_steps)]

            simulation_timesteps = heapq.merge(simulation_timesteps, model_timesteps)

        return [
            key for key, _ in itertools.groupby(list(simulation_timesteps))
        ]  # Remove duplicate timesteps

    def _step(self, t, t_next):
        t_float = np.float64(t)
        t_float_next = np.float64(t_next)

        for model in self.model_graph.eval_order:
            # Step the discrete-time models in evaluation order
            if (
                isinstance(model, DiscreteTimeModel)
                and t % Fraction(1, model.sample_rate) == 0
            ):
                model.y = model.discrete_dynamics(t_float, model.y)
                model.record_state(t_float_next, model.y)

        # Step the continuous-time models simultaneously
        self.integrator.solve(t_range=[t, t_next])

    def is_terminated(self):
        return self._t is not None and self._terminated_flag

    def _wait_on_paused_flag(self, sim_progress_value):
        SLEEP_TIME_MS = 10

        while True:
            if (
                sim_progress_value.value
                == SimulationEnvironment.ProgressType.PAUSED.value
            ):
                time.sleep(SLEEP_TIME_MS / 1e3)
            else:
                break

    def simulate(
        self,
        t_range: Tuple[Fraction, Fraction],
        realtime: bool = False,  # Required for realtime simulation
        sim_progress_value=None,  # Required for realtime simulation
        sim_latest_timestep_value=None,  # Required for realtime simulation
    ):
        if realtime:
            assert (sim_progress_value is not None) and (
                sim_latest_timestep_value is not None
            ), "Real-time simulation requires progress and latest timestep values to be passed in!"

        self._t = None
        self._terminated_flag = False

        if not (simulation_timesteps := self._collect_sim_timesteps(t_range)):
            simulation_timesteps = [t_range[0]]

        if len(simulation_timesteps) == 1 and realtime:
            n_hz = 30
            self.logger.warning(
                f"Real-time simulation with no discrete-time models will step at {n_hz} Hz"
            )

            n_steps = int(
                n_hz
                * self._round_up_to_multiple(t_range[1] - t_range[0], Fraction(1, n_hz))
            )
            simulation_timesteps = [Fraction(n, n_hz) for n in range(n_steps)]

        # Write all initial states to the history
        for model in self.model_graph.eval_order:
            if model.t_history is None or model.y_history is None:
                raise ValueError(
                    f"Model {model.name} does not have initialized history buffers!"
                )

            model.record_state(t_range[0], model.y)

        # Step through simulation timesteps, represented as rationals for numerical stability in running discrete-time models
        clock = SimulationClock()
        clock.reset()
        start_time = clock()

        accumulated_pause_time = 0.0

        for i in range(len(simulation_timesteps)):
            if sim_progress_value:
                with sim_progress_value.get_lock():
                    if (
                        sim_progress_value.value
                        == SimulationEnvironment.ProgressType.TERMINATED.value
                    ):
                        break

                pause_wait_start_time = clock()

                self._wait_on_paused_flag(sim_progress_value)
                accumulated_pause_time += clock() - pause_wait_start_time

            t = simulation_timesteps[i]
            self._t = t

            t_next = (
                simulation_timesteps[i + 1]
                if i + 1 < len(simulation_timesteps)
                else t_range[1]
            )

            try:
                self._step(t, t_next)
            except Exception as e:
                self.logger.error(
                    f"Simulation encountered exception at t={t}... Stopping simulation!"
                )
                raise e

            self._t = t_next

            if realtime:
                with sim_latest_timestep_value.get_lock():
                    sim_latest_timestep_value.value = float(self._t)

                next_real_time_deadline = (
                    start_time + accumulated_pause_time + float(t_next - t_range[0])
                )
                time_to_sleep = next_real_time_deadline - clock()

                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                else:
                    self.logger.warning(
                        f"Simulation is running behind real-time! ({-time_to_sleep:.2f} s behind)"
                    )

        for model in self.model_graph.adj.keys():
            model.record_state(self._t, model.y)

        elapsed_time = clock() - start_time - accumulated_pause_time

        if sim_progress_value:
            with sim_progress_value.get_lock():
                if (
                    sim_progress_value.value
                    != SimulationEnvironment.ProgressType.TERMINATED.value
                ):
                    self.logger.info(
                        f"Simulation Complete! Elapsed Time: {elapsed_time:.2f} s ({(t_range[1] - t_range[0]) / elapsed_time:.2f}x real-time)"
                    )
                else:
                    self.logger.info(
                        "Simulation terminated early by user after {:.2f} s".format(
                            elapsed_time
                        )
                    )

                sim_progress_value.value = (
                    SimulationEnvironment.ProgressType.TERMINATED.value
                )
        else:
            self.logger.info(
                f"Simulation Complete! Elapsed Time: {elapsed_time:.2f} s ({(t_range[1] - t_range[0]) / elapsed_time:.2f}x real-time)"
            )

        self._terminated_flag = True


class SimulationEnvironment:
    class ProgressType(Enum):
        IN_PROGRESS = 0
        PAUSED = 1
        TERMINATED = 2

    def __init__(self, simulator: Simulator, plot_env=None):
        self.simulator = simulator
        self.plot_env = plot_env

    def _plotting_process(
        self,
        sim_progress_value,
        sim_latest_timestep_value,
        show_time: bool,
    ) -> None:
        self.plot_env.render_realtime(
            sim_progress_value=sim_progress_value,
            sim_latest_timestep_value=sim_latest_timestep_value,
            show_time=show_time,
        )

    def run_simulation(
        self,
        t_range: Tuple[Fraction, Fraction],
        realtime: bool = False,
        show_time: bool = False,
    ) -> None:
        if realtime:
            with Manager() as manager:
                self.simulator.model_graph.initialize_history_buffers(
                    proc_manager=manager
                )

                sim_progress_value = Value(
                    "i", SimulationEnvironment.ProgressType.IN_PROGRESS.value
                )
                sim_latest_timestep_value = Value("d", float(t_range[0]))

                if self.plot_env is not None:
                    p = Process(
                        target=self._plotting_process,
                        args=(
                            sim_progress_value,
                            sim_latest_timestep_value,
                            show_time,
                        ),
                    )

                    p.start()

                    self.simulator.simulate(
                        t_range=t_range,
                        realtime=realtime,
                        sim_progress_value=sim_progress_value,
                        sim_latest_timestep_value=sim_latest_timestep_value,
                    )

                    p.join()
                else:
                    self.simulator.simulate(
                        t_range=t_range,
                        realtime=realtime,
                        sim_progress_value=sim_progress_value,
                        sim_latest_timestep_value=sim_latest_timestep_value,
                    )
        else:
            self.simulator.model_graph.initialize_history_buffers()
            self.simulator.simulate(t_range=t_range, realtime=realtime)

            if self.plot_env is not None:
                self.plot_env.render(show_time=show_time)


# if __name__ == "__main__":
#     environment = BaseModel(None, name="Environment")
#     motor_allocator = BaseModel(None, name="MotorAllocator")
#     rigid_body_dynamics = BaseModel(None, name="RigidBodyDynamics")
#     planner = BaseModel(None, name="Planner")
#     sensors = BaseModel(None, name="Sensors")
#     state_estimator = BaseModel(None, name="StateEstimator")
#     controller = BaseModel(None, name="Controller")

#     sensors.inputs_to(state_estimator)
#     state_estimator.inputs_to(controller)
#     controller.inputs_to(motor_allocator)
#     motor_allocator.inputs_to(rigid_body_dynamics)
#     planner.inputs_to(controller)
#     environment.inputs_to(rigid_body_dynamics)
#     sensors.feedback_from(rigid_body_dynamics)

#     model_graph = ModelGraph([
#         environment,
#         motor_allocator,
#         rigid_body_dynamics,
#         planner,
#         sensors,
#         state_estimator,
#         controller
#     ])

#     for m in model_graph.eval_order:
#         print(f"    {m.name}")
