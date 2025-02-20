import mp_lib.phase as mpl_phase
import mp_lib.basis as mpl_basis
import numpy as np

# TODO: phase instead of time


class DMP:

    def __init__(self,
                 basis_generator: mpl_basis.BasisGenerator,
                 phase_generator: mpl_phase.PhaseGenerator,
                 num_dof: int,
                 duration: float = 1.,
                 dt: float = 0.01):
        self.basis_generator = basis_generator
        self.phase_generator = phase_generator
        self.n_dof = num_dof

        self.num_time_steps = int(duration / dt)
        self.dt = dt
        self.duration = duration

        self.tau = 1.0 / (self.dt * self.num_time_steps)

        self.dmp_alpha_x = 25
        self.dmp_beta_x = 25 / 4
        self.il_regularization = 10 ** -12

        self.dmp_start_pos = np.zeros((1, num_dof))
        self.dmp_start_vel = np.zeros((1, num_dof))

        self._dmp_goal_pos = np.zeros((1, num_dof))
        self.dmp_goal_vel = np.zeros((1, num_dof))

        self.dmp_amplitude_modifier = np.ones((1, num_dof))

        self._dmp_weights = np.zeros((basis_generator.num_basis, num_dof))  # initial dmp weights

        # TODO: should these be input arguments, or set externally in application?
        self.use_tau = True
        self.use_dmp_goal_pos = True
        self.use_dmp_start_vel = False
        self.use_dmp_goal_vel = False
        self.use_dmp_amplitude_modifier = False

    @property
    def n_basis(self):
        return self.basis_generator.num_basis

    @property
    def weights(self):
        return self._dmp_weights

    @property
    def dmp_goal_pos(self):
        return self._dmp_goal_pos

    def set_weights(self, w, goal=None):
        assert w.shape == self._dmp_weights.shape
        self._dmp_weights = w
        if goal is not None:
            assert goal.shape == self._dmp_goal_pos[0].shape
            self._dmp_goal_pos[0] = goal

    def reference_trajectory(self, time):

        basis = self.basis_generator.basis(time)

        reference_pos = np.zeros((self.num_time_steps, self.n_dof))
        reference_vel = np.zeros((self.num_time_steps, self.n_dof))

        reference_pos[0, :] = self.dmp_start_pos
        reference_vel[0, :] = self.dmp_start_vel

        forcing_function = basis @ self.weights

        for i in range(self.num_time_steps - 1):
            goal_vel = self.dmp_goal_vel * self.tau / (self.dt * self.num_time_steps)
            moving_goal = self.dmp_goal_pos - goal_vel * self.dt * (self.num_time_steps - i)

            acc = self.dmp_alpha_x * (self.dmp_beta_x * (moving_goal - reference_pos[i, :]) * self.tau ** 2
                                      + (goal_vel - reference_vel[i, :]) * self.tau) \
                  + self.dmp_amplitude_modifier * forcing_function[i, :] * self.tau ** 2

            reference_vel[i + 1, :] = reference_vel[i, :] + self.dt * acc

            reference_pos[i + 1, :] = reference_pos[i, :] + self.dt * reference_vel[i + 1, :]

        return reference_pos, reference_vel

    def learn_from_imitation(self, time, reference_pos):

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
        moving_goal = self.dmp_goal_pos - self.tau / (self.num_time_steps * self.dt) \
                      * (time[-1] - time)[:, None] * self.dmp_goal_vel

        forcing_function = reference_acc / self.tau ** 2 - self.dmp_alpha_x * (
                    self.dmp_beta_x * (moving_goal - reference_pos) + (self.dmp_goal_vel - reference_vel) / self.tau)
        forcing_function = forcing_function / self.dmp_amplitude_modifier

        weight_matrix = np.linalg.solve(basis.T @ basis + np.eye(basis.shape[1]) * self.il_regularization,
                                        basis.T @ forcing_function)
        self._dmp_weights = weight_matrix
