import numpy as np
import heapq
import itertools
import time

from fractions import Fraction
from typing import List, Tuple, Any, Optional, Union, Type
from functools import reduce
from enum import Enum
from hybrid_ode_sim.simulation.ode_solvers.solver_base import Integrator
from hybrid_ode_sim.simulation.base import BaseModel, DiscreteTimeModel, ContinuousTimeModel
from hybrid_ode_sim.utils.logging_tools import Logger, LogLevel


class ModelGraph:
    def __init__(self, models: List[BaseModel]):
        self.enumerated_models = { model: i for (i, model) in enumerate(models) }
        self.adj = { model: [] for model in models}
        
        for model in models:    
            self.adj[model] += model.children_models
            del model.children_models  # Remove children_models attribute since we don't need it anymore
        self.eval_order = self._topo_sort()
    
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
        for (model, i) in self.enumerated_models.items():
            if not visited[i]:
                _topo_sort_recurse(model, visited, stack)
        
        return stack

class Simulator:
    def __init__(self, model_graph: ModelGraph, integrator: Integrator,
                 logging_level=LogLevel.INFO):
        self.model_graph = model_graph
                
        self.continuous_time_models = list(filter(lambda model: isinstance(model, ContinuousTimeModel), self.model_graph.adj.keys()))
        self.discrete_time_models = list(filter(lambda model: isinstance(model, DiscreteTimeModel), self.model_graph.adj.keys()))
        
        self.logger = Logger(logging_level, f"{self.__class__.__name__}")
        
        self.integrator = integrator
        self.integrator.register_models(self.continuous_time_models)
        
    def _collect_sim_timesteps(self, t_range: Tuple[Fraction, Fraction]) -> List[Fraction]:
        """Get simulation timesteps at which DiscreteTimeModel dynamics functions must be called

        Args:
            t_range (Tuple[Fraction, Fraction]): _description_

        Returns:
            List[Fraction]: _description_
        """
        simulation_timesteps = iter([])
        
        for model in self.discrete_time_models:
            n_steps = int(model.sample_rate * (t_range[1] - t_range[0]))
            model_timesteps = [Fraction(n, model.sample_rate) for n in range(n_steps)]

            simulation_timesteps = heapq.merge(simulation_timesteps, model_timesteps)

        return [key for key, _ in itertools.groupby(list(simulation_timesteps))] # Remove duplicate timesteps
    
    def _step(self, t, t_next):
        t_float = np.float64(t)
        
        for model in self.model_graph.eval_order:
            # Step the discrete-time models in evaluation order        
            if isinstance(model, DiscreteTimeModel) and t % Fraction(1, model.sample_rate) == 0:
                model.record_state(t_float, model.y)
                model.y = model.discrete_dynamics(t_float, model.y)

        # Step the continuous-time models simultaneously
        self.integrator.solve(t_range=[t, t_next])
    
    def simulate(self, t_range: Tuple[Fraction, Fraction]):        
        if not (simulation_timesteps := self._collect_sim_timesteps(t_range)):
            simulation_timesteps = [t_range[0]]
        
        start_time = time.time()
        
        # Step through simulation timesteps, represented as rationals for numerical stability in running discrete-time models
        for i in range(len(simulation_timesteps)):
            t = simulation_timesteps[i]
            t_next = simulation_timesteps[i + 1] if i + 1 < len(simulation_timesteps) else t_range[1]
            
            try:
                self._step(t, t_next)
            except Exception as e:
                self.logger.error(f"Simulation encountered exception at t={t}... Stopping simulation!")
                raise e
            

        elapsed_time = time.time() - start_time
        self.logger.info(f"Simulation Complete! Elapsed Time: {elapsed_time:.2f} s ({(t_range[1] - t_range[0]) / elapsed_time:.2f}x real-time)")
        
        for model in self.model_graph.adj.keys():
            model.record_state(t_range[1], model.y)
        
        
        
        
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