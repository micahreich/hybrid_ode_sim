import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from hybrid_ode_sim.utils.logging_tools import Logger, LogLevel
from hybrid_ode_sim.simulation.simulator import SimulationClock
from types import SimpleNamespace


class PlotElement:
    def __init__(self, env=None, logging_level=LogLevel.ERROR) -> None:
        if env is not None:
            self.init_environment(env)
        else:
            self.env = None
            
        self.logger = Logger(logging_level, f"{self.__class__.__name__}")

    def update(self, t):
        return

    def reset(self):
        return

    def init_environment(self, env):
        self.env = env


class PlotEnvironment:
    def __init__(
        self,
        fig,
        ax,
        sim_t_range: Tuple[float, float],
        frame_rate: int,
        t_start=None,
        t_end=None,
    ) -> None:
        self.fig = fig
        self.ax = ax
        self.frame_rate = frame_rate
        self.t_range = sim_t_range
        self.ani_paused = False
        self.logger = Logger(LogLevel.INFO, f"{self.__class__.__name__}")

        self.plot_elements = []

        if t_start is None:
            self.t_start = sim_t_range[0]
        else:
            assert (
                sim_t_range[0] <= t_start < sim_t_range[1]
            ), "The start time must be within the simulation time range."
            self.t_start = t_start

        if t_end is None:
            self.t_end = sim_t_range[1]
        else:
            assert (
                self.t_start < t_end <= sim_t_range[1]
            ), "The end time must be within the simulation time range."
            self.t_end = t_end

    def attach_element(self, element: PlotElement):
        element.init_environment(self)
        self.plot_elements.append(element)

        return self

    def render_realtime(
        self, sim_termination_event, sim_latest_timestep, t_range, show_time=False
    ):
        frame_reset_data = SimpleNamespace(t=t_range[0], frame_idx=0)

        interval_s = 1 / self.frame_rate
        interval_ms = interval_s * 1e3

        def env_update(frame_idx, frame_reset_data):
            is_realtime_terminated = sim_termination_event.is_set()

            # with sim_latest_timestep.get_lock():  # Lock to ensure safe read
            if not np.allclose(sim_latest_timestep.value, frame_reset_data.t):
                frame_reset_data.t = sim_latest_timestep.value
                frame_reset_data.frame_idx = frame_idx
                t = frame_reset_data.t
            else:
                delta_frame_cnt = frame_idx - frame_reset_data.frame_idx
                t = frame_reset_data.t + delta_frame_cnt * interval_s

            if show_time and not is_realtime_terminated:
                self.ax.set_title(f"t={t : .1f}s")

            for element in self.plot_elements:
                element.update(t)

        def on_key_press(event):
            if event.key == "q":
                sim_termination_event.set()
                plt.close(self.fig)  # Close the figure window
            elif event.key == " ":
                if self.ani_paused:
                    ani.event_source.start()
                    self.ani_paused = False
                else:
                    ani.event_source.stop()
                    self.ani_paused = True

        ani = FuncAnimation(
            self.fig,
            env_update,
            interval=interval_ms,
            repeat=False,
            fargs=(frame_reset_data,),
        )

        self.fig.canvas.mpl_connect("key_press_event", on_key_press)
        self.fig.tight_layout()

        plt.show()

    def render(
        self,
        show_time=True,
        save=False,
        save_path=None,
    ):
        frame_times = np.arange(self.t_start, self.t_end, 1 / self.frame_rate)

        def env_update(t):
            if show_time:
                self.ax.set_title(f"t={t : .1f}s")
            for element in self.plot_elements:
                element.update(t)

            if t == frame_times[-1]:
                for element in self.plot_elements:
                    element.reset()

        def on_key_press(event):
            if event.key == "q":
                plt.close(self.fig)  # Close the figure window
            elif event.key == " ":
                if self.ani_paused:
                    ani.event_source.start()
                    self.ani_paused = False
                else:
                    ani.event_source.stop()
                    self.ani_paused = True

        ani = FuncAnimation(
            self.fig,
            env_update,
            frames=frame_times,
            interval=1 / self.frame_rate * 1e3,
            repeat=not save,
        )

        self.fig.canvas.mpl_connect("key_press_event", on_key_press)
        self.fig.tight_layout()

        if save:
            if not save_path:
                raise ValueError("Please provide a save path for the animation.")

            self.logger.info(f"Saving animation to {save_path}")

            base_path, _ = os.path.splitext(save_path)
            ani.save(f"{base_path}.gif", writer=FFMpegWriter(fps=int(self.frame_rate)))
        else:
            plt.show()
