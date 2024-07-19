import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from hybrid_ode_sim.utils.logging_tools import Logger, LogLevel


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

    def render(self, plot_elements: List, show_time=False, save=False, save_path=None):
        frame_times = np.arange(self.t_start, self.t_end, 1 / self.frame_rate)
        
        def env_update(t):
            if show_time:
                self.ax.set_title(f"t={t : .1f}s")

            for element in plot_elements:
                element.update(t)
            
            if t == frame_times[-1]:
                for element in plot_elements:
                    element.reset()

        ani = FuncAnimation(
            self.fig,
            env_update,
            frames=frame_times,
            interval=1 / self.frame_rate * 1e3,
            repeat=not save,
        )

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


class PlotElement:
    def __init__(self, env, logging_level=LogLevel.ERROR) -> None:
        self.env = env
        self.logger = Logger(logging_level, f"{self.__class__.__name__}")

    def update(self, t): return
    
    def reset(self): return
