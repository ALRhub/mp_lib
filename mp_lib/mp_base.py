import numpy as np
import mp_lib.phase as mpl_phase
import mp_lib.basis as mpl_basis
import abc


class BaseMP(abc.ABC):
    def __init__(self,
                 num_dof: int,
                 duration: float,
                 dt: float,
                 basis_generator: mpl_basis.BasisGenerator,
                 phase_generator: mpl_phase.PhaseGenerator,
                 ):

        self.basis_generator = basis_generator
        self.phase_generator = phase_generator  # actually only used in the basis_generator
        self.n_dof = num_dof

        self.num_time_steps = int(duration / dt)
        self.dt = dt
        self.duration = duration

    @property
    def n_basis(self):
        return self.basis_generator.num_basis

    @property
    def weights(self):
        raise NotImplemented

    def set_weights(self, w):
        raise NotImplemented

    def reference_trajectory(self, t):
        raise NotImplemented

