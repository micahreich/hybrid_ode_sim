# hybrid_ode_sim

Contains a hybrid ODE solver for hybrid discrete-continuous time systems. This framework is used to simulate software systems which run at a constant rate (e.g. controllers, planners) along with rigid body dynamics.
- Continuous time systems inherit from the `ContinuousTimeModel` class
- Discrete time systems inherit from the `DiscreteTimeModel` class.
- Models are compiled into a `ModelGraph` which encodes any runtime dependancies between models
- All `ContinuousTimeModel`s in the `ModelGraph` are integrated together

Contains simulation rendering tools for creating visualizations of simulations after they occur, can also save video/GIF

## ODE Solvers
### Adaptive RK
An instance of an adaptive Runge-Kutta implementation can be provided as the integrator to a simulator. Currently implemented are RK23 and RK54, but adding other methods is as easy as providing the Butcher Tableau.

### Fixed-Step RK
An instance of a fixed-step Runge-Kutta implementation can be provided as the integrator to a simulator. Currently implemented is RK4, but adding other methods is as easy as providing the Butcher Tableau.

## Rendering

Matplotlib animations can be created by creating and rendering PlotElements which grab data from simulation systems. Multiple plot elements can be combined into one PlotEnvironment, rendered, and also saved as a video or GIF 