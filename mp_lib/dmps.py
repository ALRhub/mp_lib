import mp_lib.phase as mpl_phase
import mp_lib.basis as mpl_basis
from mp_lib.base_mp import BaseMP
import numpy as np


class DMP(BaseMP):
    def __init__(self,
                 num_dof: int,
                 num_basis: int,
                 duration: float,
                 dt: float,
                 basis_generator: mpl_basis.BasisGenerator = None,
                 phase_generator: mpl_phase.PhaseGenerator = None,
                 ):

        # if phase_generator is None:
        #     phase_generator = mpl_phase.ExpDecayPhaseGenerator(duration=duration, alpha_phase=4)
        #
        # if basis_generator is None:
        #     basis_generator = mpl_basis.DMPBasisGenerator(phase_generator=phase_generator, num_basis=num_basis,
        #                                                   duration=duration)

        # assert basis_generator.num_basis == num_basis

        super().__init__(num_dof=num_dof, dt=dt, basis_generator=basis_generator, phase_generator=phase_generator)

        self.dmp_alpha_x = 25
        self.dmp_beta_x = 25 / 4
        self.il_regularization = 10 ** -12

        self._dmp_start_pos = None
        self._dmp_start_vel = None
        # self._dmp_start_pos = np.zeros(num_dof)
        # self._dmp_start_vel = np.zeros(num_dof)

        self._dmp_goal_pos = np.zeros(num_dof)
        self._dmp_goal_vel = np.zeros(num_dof)

        self.dmp_amplitude_modifier = np.ones(num_dof)

        self.n_weights = self.n_basis * self.n_dof
        # self._weights = np.zeros(shape=(num_basis * num_dof, 1))  # initial dmp weights
        self._weights = None

        # TODO: should these be input arguments, or set externally in application?
        self.use_tau = True
        self.use_dmp_goal_pos = True
        self.use_dmp_start_vel = False
        self.use_dmp_goal_vel = False
        self.use_dmp_amplitude_modifier = False

    @property
    def dmp_start_pos(self):
        return self._dmp_start_pos

    @dmp_start_pos.setter
    def dmp_start_pos(self, x):
        self._dmp_start_pos = x

    @property
    def dmp_start_vel(self):
        return self._dmp_start_vel

    @dmp_start_vel.setter
    def dmp_start_vel(self, x):
        self._dmp_start_vel = x

    @property
    def dmp_goal_vel(self):
        return self._dmp_goal_vel

    @property
    def dmp_goal_pos(self):
        return self._dmp_goal_pos

    @property
    def weights(self):
        return self._weights

    def set_weights(self, weights, goal=None):
        weights = np.atleast_2d(weights)
        if weights.shape[0] != self.n_weights:
            weights = weights.T
        # assert weights.shape == self._weights.shape
        self._weights = weights

        if goal is not None:
            assert goal.shape == self._dmp_goal_pos.shape
            self._dmp_goal_pos = goal

    def reference_trajectory(self, time):
        num_time_steps = len(time)
        tau = 1 / time[-1]

        reference_pos = np.zeros((num_time_steps, self.n_dof))
        reference_vel = np.zeros((num_time_steps, self.n_dof))

        reference_pos[0, :] = self.dmp_start_pos
        reference_vel[0, :] = self.dmp_start_vel

        basis_multi_dof = self.basis_generator.basis_multi_dof(time, self.n_dof)
        forcing_function = np.reshape(basis_multi_dof @ self.weights, (num_time_steps, self.n_dof), order='F')

        for i in range(num_time_steps - 1):
            goal_vel = self.dmp_goal_vel * tau / (self.dt * num_time_steps)
            moving_goal = self.dmp_goal_pos - goal_vel * self.dt * (num_time_steps - i)

            acc = self.dmp_alpha_x * (self.dmp_beta_x * (moving_goal - reference_pos[i, :]) * tau ** 2
                                      + (goal_vel - reference_vel[i, :]) * tau) \
                  + self.dmp_amplitude_modifier * forcing_function[i, :] * tau ** 2

            reference_vel[i + 1, :] = reference_vel[i, :] + self.dt * acc

            reference_pos[i + 1, :] = reference_pos[i, :] + self.dt * reference_vel[i + 1, :]

        return reference_pos, reference_vel

    def learn_from_imitation(self, time, reference_pos):

        num_time_steps = len(time)
        basis = self.basis_generator.basis(time)

        reference_vel = np.diff(reference_pos, axis=0) / self.dt
        reference_vel = np.vstack((reference_vel, reference_vel[-1, :]))
        reference_acc = np.diff(reference_pos, axis=0, n=2) / self.dt ** 2
        reference_acc = np.vstack((reference_acc, reference_acc[-2:, :]))

        self.dmp_start_pos = reference_pos[0, :]
        self.tau = 1.0 / (time[-1] + self.dt)

        if self.use_dmp_start_vel:
            self.dmp_start_vel = reference_vel[0, :]
        else:
            self.dmp_start_vel = np.zeros((1, self.n_dof))

        self._dmp_goal_pos = reference_pos[-1:, :]
        if self.use_dmp_goal_vel:
            self.dmp_goal_vel = reference_vel[0:1, :]
        else:
            self.dmp_goal_vel = np.zeros((1, self.n_dof))

        if self.use_dmp_amplitude_modifier:
            self.dmp_amplitude_modifier = np.max(reference_pos, axis=0) - np.min(reference_pos, axis=0)
            self.dmp_amplitude_modifier[self.dmp_amplitude_modifier < 0.01] = 1
        else:
            self.dmp_amplitude_modifier = np.ones((1, self.n_dof))

        # FIXME: make nicer and check for dimensionalities
        moving_goal = self.dmp_goal_pos - self.tau / (num_time_steps * self.dt) \
                      * (time[-1] - time)[:, None] * self.dmp_goal_vel

        forcing_function = reference_acc / self.tau ** 2 - self.dmp_alpha_x * (
                self.dmp_beta_x * (moving_goal - reference_pos) + (self.dmp_goal_vel - reference_vel) / self.tau)
        forcing_function = forcing_function / self.dmp_amplitude_modifier

        weight_matrix = np.linalg.solve(basis.T @ basis + np.eye(basis.shape[1]) * self.il_regularization,
                                        basis.T @ forcing_function)
        self._weights = weight_matrix
