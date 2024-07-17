from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from hybrid_ode_sim.simulation.ode_solvers.solver_base import RKIntegrator
from hybrid_ode_sim.utils.logging_tools import Logger, LogLevel


@dataclass
class AdaptiveStepButcherTableau:
    c: np.ndarray  # Time step coefficients
    A: np.ndarray  # Coefficients for the Runge-Kutta stages
    b_lower_order: np.ndarray  # Coefficients of k_i's for the lower order estimate
    b_higher_order: np.ndarray  # Coefficients of k_i's for the higher order estimate
    order: int  # Order of the RK method
    n_stages: int  # Number of stages in the RK method
    error_estimation_order: int  # Order of the error estimation
    use_higher_order_estimate: bool = field(
        default=True
    )  # Whether to use the higher order estimate or not
    name: Optional[str] = field(default=None)  # Name of the Butcher Tableau


def _rms_norm(x: np.ndarray) -> float:
    """Computes the root mean squared norm of a vector `x`.
    RMS norm is defined as `sqrt(1 / n * sum x_i^2)` where `n` is the length of the vector `x`.

    Args:
        x (np.ndarray): Vector

    Returns:
        float: Root-mean-squared norm of the vector `x`
    """
    return np.linalg.norm(x) / x.size**0.5


class AdaptiveStepRK(RKIntegrator):
    def __init__(
        self,
        logging_level: LogLevel = LogLevel.DEBUG,
        h_max: float = np.inf,
        max_h_factor: float = 10,
        min_h_factor=0.2,
        h0: Optional[float] = None,
        safety_factor: float = 0.9,
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ):
        super().__init__(logging_level)

        self.h_max = h_max
        self.max_h_factor = max_h_factor
        self.min_h_factor = min_h_factor
        self.h0 = h0

        self.safety_factor = safety_factor
        self.atol = atol  # Absolute error tolerance
        self.rtol = rtol  # Relative error tolerance

        if self.butcher_tableau.use_higher_order_estimate and np.allclose(
            self.butcher_tableau.A[-1], self.butcher_tableau.b_higher_order, atol=1e-9
        ):
            self.fsal = True  # Automatically use FSAL property if available
        else:
            self.fsal = False

        self.k0_fsal = None
        self.error_exponent = -1 / (self.butcher_tableau.error_estimation_order + 1)

        # Perform some additional checks on the Butcher Tableau
        assert (
            self.butcher_tableau.b_lower_order.size
            == self.butcher_tableau.b_higher_order.size
        )

    def _compute_ys_new(
        self, y: np.ndarray, h: float, K: np.ndarray, y_validate: Callable
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_new_higher_order = y_validate(
            y + h * np.dot(self.butcher_tableau.b_higher_order, K)
        )
        y_new_lower_order = y_validate(
            y + h * np.dot(self.butcher_tableau.b_lower_order, K)
        )

        return y_new_higher_order, y_new_lower_order

    def _validate_h(self, t_range: Tuple[float, float], t: float, h: float):
        return np.clip(min(h, t_range[1] - t), self.h_min, self.h_max)

    def _update_step_size(
        self,
        t: float,
        h: float,
        y: np.ndarray,
        y_new_higher_order: np.ndarray,
        y_new_lower_order: np.ndarray,
    ) -> Tuple[bool, float]:
        y_new = (
            y_new_higher_order
            if self.butcher_tableau.use_higher_order_estimate
            else y_new_lower_order
        )

        scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_new))
        error_norm = _rms_norm((y_new_lower_order - y_new_higher_order) / scale)

        step_accepted = error_norm < 1.0

        if not step_accepted and h <= self.h_min:
            self.logger.critical(
                f"Step size (h={h}) is too small near t={t} with continued step rejection!"
            )

        if error_norm == 0:
            factor = self.max_h_factor
        else:
            factor = self.safety_factor * error_norm**self.error_exponent

            if not step_accepted:
                factor = min(1.0, factor)

        factor = np.clip(factor, self.min_h_factor, self.max_h_factor)

        return step_accepted, h * factor

    def _initial_step_size(
        self, f: Callable, f0: np.ndarray, t0: float, y0: np.ndarray
    ) -> float:
        # Taken from scipy.integrate._ivp.common.select_initial_step (https://github.com/scipy/scipy/blob/v1.12.0/scipy/integrate/_ivp/common.py#L68)
        scale = self.atol + np.abs(y0) * self.rtol
        d0 = _rms_norm(y0 / scale)
        d1 = _rms_norm(f0 / scale)

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 1e-2 * d0 / d1

        direction = 1.0  # Integrating forwards

        y1 = y0 + h0 * direction * f0
        f1 = f(t0 + h0 * direction, y1)
        d2 = _rms_norm((f1 - f0) / scale) / h0

        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (1e-2 / max(d1, d2)) ** (1 / (self.butcher_tableau.order + 1))

        return min(1e2 * h0, h1)

    def _step(
        self,
        t_range: Tuple[float, float],
        t: float,
        h: float,
        y: np.ndarray,
        f: Callable,
        y_validate: Callable,
    ) -> Tuple[np.ndarray, float, float]:
        step_accepted = False

        # Perform one adaptive RK step, updating the timestep as needed until the step is accepted
        while not step_accepted:
            K = self._compute_K(f, t, y, h, k0_fsal=self.k0_fsal)

            y_new_higher_order, y_new_lower_order = self._compute_ys_new(
                y, h, K, y_validate
            )

            step_accepted, new_h_whole = self._update_step_size(
                t, h, y, y_new_higher_order, y_new_lower_order
            )

            if not step_accepted:
                h = self._validate_h(t_range, t, new_h_whole)

        if self.fsal:
            self.k0_fsal = K[-1]

        y_new = (
            y_new_higher_order
            if self.butcher_tableau.use_higher_order_estimate
            else y_new_lower_order
        )

        return y_new, new_h_whole, h

    def solve(self, t_range: Tuple[float, float]):
        if not self.models_registered:
            raise RuntimeError("No models registered with the integrator! Exiting...")

        t, t_f = np.float64(t_range[0]), np.float64(t_range[1])

        # Perform some checks on the time range to ensure validity
        if not t_f > np.nextafter(t, np.inf):
            self.logger.error(
                f"Invalid time range {t_range} (t_f must be strictly greater than t)"
            )

        y0_combined = np.concatenate([model.y for model in self.models])
        f0_combined = self.f_combined(t, y0_combined)
        if self.fsal:
            self.k0_fsal = f0_combined

        # Choose the initial step size
        h = self._validate_h(
            t_range,
            t,
            (
                self._initial_step_size(self.f_combined, f0_combined, t, y0_combined)
                if self.h0 is None
                else self.h0
            ),
        )

        y_combined = y0_combined

        while t < t_f:
            self.record_state_combined(t, y_combined)
            y_new_combined, next_h_whole, h_used = self._step(
                t_range, t, h, y_combined, self.f_combined, self.y_validate_combined
            )

            # Put the states back into `y` for each model
            self.write_back_model_y(y_new_combined)

            t += h_used  # Increment timestep by the h used
            y_combined = y_new_combined  # Update the state to the new state
            h = self._validate_h(
                t_range, t, next_h_whole
            )  # Update the timestep for the next iteration after validation

        self.h0 = (
            next_h_whole  # Update the initial step size for the next .solve() call
        )


class RK45(AdaptiveStepRK):
    butcher_tableau = AdaptiveStepButcherTableau(
        c=np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]),
        A=np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [1 / 5, 0, 0, 0, 0, 0, 0],
                [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
                [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
                [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
                [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
                [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
            ]
        ),
        b_higher_order=np.array(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
        ),
        b_lower_order=np.array(
            [
                5179 / 57600,
                0,
                7571 / 16695,
                393 / 640,
                -92097 / 339200,
                187 / 2100,
                1 / 40,
            ]
        ),
        order=5,
        n_stages=7,
        error_estimation_order=4,
        use_higher_order_estimate=True,
        name="Dormand-Prince",
    )


class RK23(AdaptiveStepRK):
    butcher_tableau = AdaptiveStepButcherTableau(
        c=np.array([0, 1 / 2, 3 / 4, 1]),
        A=np.array(
            [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 3 / 4, 0, 0], [2 / 9, 1 / 3, 4 / 9, 0]]
        ),
        b_higher_order=np.array([2 / 9, 1 / 3, 4 / 9, 0]),
        b_lower_order=np.array([7 / 24, 1 / 4, 1 / 3, 1 / 8]),
        order=3,
        n_stages=4,
        error_estimation_order=2,
        use_higher_order_estimate=True,
        name="Bogacki-Shampine",
    )
