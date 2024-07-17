import numpy as np

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Any, Callable
from hybrid_ode_sim.simulation.ode_solvers.solver_base import RKIntegrator
from hybrid_ode_sim.simulation.base import ContinuousTimeModel
from hybrid_ode_sim.utils.logging_tools import LogLevel, Logger


@dataclass
class FixedStepButcherTableau:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    order: int
    n_stages: int
    name: Optional[str] = field(default=None)


class FixedStepRK(RKIntegrator):
    def __init__(self, h: float, logging_level: LogLevel=LogLevel.DEBUG) -> None:
        super().__init__(logging_level)
        self.h = h
        
        # Perform some additional checks on the Butcher Tableau
        assert self.butcher_tableau.b.size == self.butcher_tableau.n_stages
    
    def _compute_y_new(self, y: np.ndarray, h: float, K: np.ndarray, y_validate: Callable) -> np.ndarray:
        return y_validate( y + h * np.dot(self.butcher_tableau.b, K) )
    
    def _validate_h(self, t_range: Tuple[float, float], t: float, h: float) -> float:
        return max(self.h_min, min(h, t_range[1] - t))

    def solve(self, t_range: Tuple[float, float]) -> None:
        if not self.models_registered:
            raise RuntimeError("No models registered with the integrator! Exiting...")
        
        t, t_f = np.float64(t_range[0]), np.float64(t_range[1])
        
        # Perform some checks on the time range to ensure validity
        if not t_f > np.nextafter(t, np.inf):
            self.logger.error(f"Invalid time range {t_range} (t_f must be strictly greater than t)")
            
        y0_combined = np.concatenate([model.y for model in self.models])
        y_combined = y0_combined
        
        while t < t_f:
            self.record_state_combined(t, y_combined)
            
            h = self._validate_h(t_range, t, self.h)
            K = self._compute_K(self.f_combined, t, y_combined, h)
            
            y_new_combined = self._compute_y_new(y_combined, h, K, self.y_validate_combined) 
            
            # Put the states back into `y` for each
            self.write_back_model_y(y_new_combined)
            
            t += h
            y_combined = y_new_combined


class RK4(FixedStepRK):
    butcher_tableau = FixedStepButcherTableau(
        c = np.array([0, 1/2, 1/2, 1]),
        A = np.array([
            [0,   0,   0, 0],
            [1/2, 0,   0, 0],
            [0,   1/2, 0, 0],
            [0,   0,   1, 0]
        ]),
        b = np.array([1/6, 1/3, 1/3, 1/6]),
        order = 4,
        n_stages = 4,
        name = "RK4"
    )
