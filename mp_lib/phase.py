from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import make_interp_spline as spi_make_interp_spline


class PhaseGenerator(ABC):

    def __init__(self):
        return

    @abstractmethod
    def phase(self, time):

        # Base class...
        pass


class LinearPhaseGenerator(PhaseGenerator):

    def __init__(self, duration=1.0):

        PhaseGenerator.__init__(self)
        self.duration = duration

    def phase(self, time):

        phase = np.array(time) / self.duration

        return phase


class SmoothPhaseGenerator(PhaseGenerator):

    def __init__(self, duration=1):

        PhaseGenerator.__init__(self)
        self.duration = duration

        left = [(1, 0.0), (2, 0.0)]
        right = [(1, 0.0), (2, 0.0)]

        self.b_spline_f = spi_make_interp_spline([0, self.duration], [0, 1], bc_type=(left, right), k=5)

    def phase(self, time):
        phase = self.b_spline_f(time)

        return phase


class RhythmicPhaseGenerator(PhaseGenerator):

    def __init__(self, phase_period=1.0, use_modulo=False):
        PhaseGenerator.__init__(self)
        self.phase_period = phase_period
        self.use_modulo = use_modulo

    def phase(self, time):
        phase = time / self.phase_period
        if self.use_modulo:
            phase = np.mod(phase, 1.0)

        return phase


class ExpDecayPhaseGenerator(PhaseGenerator):

    def __init__(self, duration=1, alpha_phase=2):

        PhaseGenerator.__init__(self)

        self.tau = 1.0 / duration
        self.alpha_phase = alpha_phase

    def phase(self, time):

        time = time * self.tau

        phase = np.exp(- time * self.alpha_phase)

        return phase
