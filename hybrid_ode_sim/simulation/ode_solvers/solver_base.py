from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

import numpy as np

from hybrid_ode_sim.simulation.base import ContinuousTimeModel
from hybrid_ode_sim.utils.logging_tools import Logger, LogLevel

# Useful ~typedefs for combining the individual model functions into single functions
CombinedContinuousDynamics = Callable[[float, np.ndarray], np.ndarray]
CombinedYValidate = Callable[[np.ndarray], np.ndarray]
CombinedRecordState = Callable[[float, np.ndarray], None]
WriteBackModelY = Callable[[np.ndarray], None]


class Integrator:
    def __init__(self, logging_level: LogLevel):
        self.models = None
        self.logger = Logger(logging_level, f"{self.__class__.__name__}")
        self.models_registered = False

    def register_models(self, models: List[ContinuousTimeModel]):
        self.models = models

        model_state_dimensions = np.array([0] + [model.y.size for model in models])
        if not Integrator._model_list_validate(model_state_dimensions):
            raise AttributeError(
                "All ContinuousTimeModel models must have a non-empty initial state"
            )

        self.model_state_dimensions = np.cumsum(model_state_dimensions, dtype=int)

        # (
        #     self.f_combined,
        #     self.y_validate_combined,
        #     self.record_state_combined,
        #     self.write_back_model_y,
        # ) = Integrator._get_combined_fns(self.models)
        self.models_registered = True

    @staticmethod
    def _model_list_validate(model_state_dimensions: np.ndarray):
        return np.all(model_state_dimensions[1:] > 0)

    def f_combined(self, t, y_stacked):
        stacked_derivative = np.empty_like(y_stacked)
        self.write_back_model_y(y_stacked)

        for i, model in enumerate(self.models):
            u, v = self.model_state_dimensions[i], self.model_state_dimensions[i + 1]
            y = y_stacked[u:v]

            model_derivative = model.continuous_dynamics(t, y)
            stacked_derivative[u:v] = model_derivative

        return stacked_derivative

    def y_validate_combined(self, y_stacked):
        validated_y_stacked = np.empty_like(y_stacked)

        for i, model in enumerate(self.models):
            model_output = model.output_validate(
                y_stacked[
                    self.model_state_dimensions[i] : self.model_state_dimensions[i + 1]
                ]
            )
            validated_y_stacked[
                self.model_state_dimensions[i] : self.model_state_dimensions[i + 1]
            ] = model_output

        return validated_y_stacked

    def record_state_combined(self, t, combined_state):
        for i, model in enumerate(self.models):
            model.record_state(
                t,
                combined_state[
                    self.model_state_dimensions[i] : self.model_state_dimensions[i + 1]
                ],
            )

    def write_back_model_y(self, combined_state):
        for i, model in enumerate(self.models):
            model.y = combined_state[
                self.model_state_dimensions[i] : self.model_state_dimensions[i + 1]
            ]

    # @staticmethod
    # def _get_combined_fns(
    #     models: List[ContinuousTimeModel],
    # ) -> Tuple[
    #     CombinedContinuousDynamics,
    #     CombinedYValidate,
    #     CombinedRecordState,
    #     WriteBackModelY,
    # ]:
    #     model_state_dimensions = np.array([0] + [model.y.size for model in models])
    #     if not Integrator._model_list_validate(model_state_dimensions):
    #         raise AttributeError(
    #             "All ContinuousTimeModel models must have a non-empty initial state"
    #         )

    #     model_state_dimensions = np.cumsum(model_state_dimensions, dtype=int)

    #     def _combined_continuous_dynamics(t, y_stacked):
    #         stacked_derivative = np.empty_like(y_stacked)
    #         _write_back_model_y(y_stacked)

    #         for i, model in enumerate(models):
    #             u, v = model_state_dimensions[i], model_state_dimensions[i + 1]
    #             y = y_stacked[u:v]

    #             model_derivative = model.continuous_dynamics(t, y)
    #             stacked_derivative[u:v] = model_derivative

    #         return stacked_derivative

    #     def _combined_y_validate(y_stacked):
    #         validated_y_stacked = np.empty_like(y_stacked)

    #         for i, model in enumerate(models):
    #             model_output = model.output_validate(
    #                 y_stacked[model_state_dimensions[i] : model_state_dimensions[i + 1]]
    #             )
    #             validated_y_stacked[
    #                 model_state_dimensions[i] : model_state_dimensions[i + 1]
    #             ] = model_output

    #         return validated_y_stacked

    #     def _combined_record_state(t, combined_state):
    #         for i, model in enumerate(models):
    #             model.record_state(
    #                 t,
    #                 combined_state[
    #                     model_state_dimensions[i] : model_state_dimensions[i + 1]
    #                 ],
    #             )

    #     def _write_back_model_y(combined_state):
    #         for i, model in enumerate(models):
    #             model.y = combined_state[
    #                 model_state_dimensions[i] : model_state_dimensions[i + 1]
    #             ]

    #     return (
    #         _combined_continuous_dynamics,
    #         _combined_y_validate,
    #         _combined_record_state,
    #         _write_back_model_y,
    #     )

    def solve(self, t_range: Tuple[float, float]):
        raise NotImplementedError


class RKIntegrator(Integrator):
    def __init__(self, logging_level: LogLevel):
        super().__init__(logging_level)
        self.h_min = 10.0 * np.finfo(np.float64).eps

        assert self.butcher_tableau.n_stages == self.butcher_tableau.c.size

    def _compute_K(
        self, f: Callable, t: float, y: np.ndarray, h: float, k0_fsal=None
    ) -> np.ndarray:
        K = np.empty((self.butcher_tableau.n_stages, y.size))

        K[0] = f(t, y) if k0_fsal is None else k0_fsal

        for i in range(1, self.butcher_tableau.n_stages):
            K[i] = f(
                t + self.butcher_tableau.c[i] * h,
                y + h * np.dot(self.butcher_tableau.A[i, :i], K[:i]),
            )

        return K
