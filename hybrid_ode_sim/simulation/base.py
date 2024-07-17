from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

from hybrid_ode_sim.utils.logging_tools import Logger, LogLevel


def ensure_iterable(x: Any) -> List[Any]:
    if not isinstance(x, list):
        return [x]
    return x


class InputDirectionType(Enum):
    FORWARD = 1
    REVERSE = 2


class BaseModel:
    def __init__(
        self,
        y0,
        name: str = "UNTITLED_MODEL",
        params=None,
        logging_level=LogLevel.ERROR,
    ):
        self.name = name
        self.params = params

        self.input_models = {}  # Map name -> instance of input model
        self.children_models = (
            []
        )  # List of models which receive input from this model, will get deleted

        self.y = y0
        self.t_history = []
        self.y_history = []

        self.logger = Logger(logging_level, f"{self.__class__.__name__}")

    def history(
        self, interpolator_kind=None
    ) -> Tuple[np.ndarray, np.ndarray, interp1d]:
        ts = np.array(self.t_history)
        ys = np.array(self.y_history)

        # Linear interpolation on states for continuous-time models, zero-order hold for discrete-time models
        if interpolator_kind is None:
            interpolator_kind = (
                "linear" if isinstance(self, ContinuousTimeModel) else "previous"
            )

        return ts, ys, interp1d(ts, ys, axis=0, kind=interpolator_kind)

    def record_state(self, t, y):
        self.t_history.append(t)
        self.y_history.append(y)

    def collect_inputs(self) -> Dict[str, Any]:
        return {name: model.y for (name, model) in self.input_models.items()}

    def _register_model(self, model_B, direction: InputDirectionType) -> None:
        def individual(model, direction: InputDirectionType) -> None:
            if direction == InputDirectionType.FORWARD:
                if self.name in model.input_models:
                    raise ValueError(
                        f"Model {self.name} is already registered as an input to {model.name}!"
                    )

                model.input_models[self.name] = self
                self.children_models.append(model)
            elif direction == InputDirectionType.REVERSE:
                self.input_models[model.name] = model

        model_B = ensure_iterable(model_B)

        for model in model_B:
            individual(model, direction)

    def inputs_to(self, model_B) -> None:
        """Imposes model evaluation order such that `self` is evaluated before `model_B`.

        `self -[input]-> model_B` in block diagram is equivalent to `self.inputs_to(model_B)`

        Args:
            model_B (BaseModel): The model which is fed input.
        """
        self._register_model(model_B, InputDirectionType.FORWARD)

    def feedback_from(self, model_B) -> None:
        """Does not necesarrily impose evaluation order, but indicates that `self` is fed input by `model_B`.

        `model_B -[feedback]-> self` in block diagram is equivalent to `self.feedback_from(model_B)`

        Args:
            model_B (BaseModel): The model which is fed as input.
        """
        self._register_model(model_B, InputDirectionType.REVERSE)


class DiscreteTimeModel(BaseModel):
    def __init__(
        self,
        y0: Any,
        sample_rate: int,
        name: str = "UNTITLED_DISCRETE_MODEL",
        params=None,
        logging_level=LogLevel.ERROR,
    ):
        super().__init__(y0, name, params, logging_level)
        self.sample_rate = sample_rate

    def discrete_dynamics(self, t: float, y: Any) -> Any:
        raise NotImplementedError  # y[k+1] = f(t, y[k], ...)


class ContinuousTimeModel(BaseModel):
    def __init__(
        self,
        y0: np.ndarray,
        name: str = "UNTITLED_CONTINUOUS_MODEL",
        params=None,
        logging_level=LogLevel.ERROR,
    ):
        super().__init__(y0, name, params, logging_level)

    def continuous_dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # dy/dt = f(t, y, x)

    def output_validate(self, y: np.ndarray) -> np.ndarray:
        return y

    def inputs_to(self, model_B) -> None:
        model_B = ensure_iterable(model_B)

        if any([isinstance(c, DiscreteTimeModel) for c in model_B]):
            raise ValueError(
                f"{self.name} (ContinuousTimeModel) cannot have DiscreteTimeModel descendants."
            )

        return super().inputs_to(model_B)
