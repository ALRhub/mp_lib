from abc import ABC, abstractmethod
import numpy as np
import mp_lib.phase as mpl_phase


class BasisGenerator(ABC):

    def __init__(self, phase_generator: mpl_phase.PhaseGenerator, num_basis: int = 10):

        self.num_basis = num_basis
        self.phase_generator = phase_generator

    @abstractmethod
    def basis(self, time):
        pass

    def basis_multi_dof(self, time, num_dof):
        basis_single_dof = self.basis(time)

        basis_multi_dof = np.zeros((basis_single_dof.shape[0] * num_dof, basis_single_dof.shape[1] * num_dof))

        for i in range(num_dof):
            row_indices = slice(i * basis_single_dof.shape[0], (i + 1) * basis_single_dof.shape[0])
            column_indices = slice(i * basis_single_dof.shape[1], (i + 1) * basis_single_dof.shape[1])

            basis_multi_dof[row_indices, column_indices] = basis_single_dof

        return basis_multi_dof


class DMPBasisGenerator(BasisGenerator):

    def __init__(self, phase_generator, num_basis=10,
                 duration: float = 1, basis_bandwidth_factor: int = 3):
        BasisGenerator.__init__(self, phase_generator, num_basis)

        self.basis_bandwidth_factor = basis_bandwidth_factor

        time_points = np.linspace(0, duration, self.num_basis)
        self.centers = self.phase_generator.phase(time_points)

        tmp_bandwidth = np.hstack((self.centers[1:]-self.centers[0:-1], self.centers[-1] - self.centers[- 2]))

        # The centers should not overlap too much (makes w almost random due to aliasing effect). Empirically chosen
        self.bandwidth = self.basis_bandwidth_factor / (tmp_bandwidth ** 2)

    def basis(self, time):
        phase = self.phase_generator.phase(time)

        diff_sqr = (phase[:, None] - self.centers[None, :]) ** 2 * self.bandwidth[None, :]
        basis = np.exp(- diff_sqr / 2)

        sum_b = np.sum(basis, axis=1)
        basis = basis * phase[:, None] / sum_b[:, None]

        return basis


class NormalizedRBFBasisGenerator(BasisGenerator):

    def __init__(self, phase_generator, num_basis=10,
                 duration: float = 1, basis_bandwidth_factor: int = 3,
                 zero_start=False, zero_goal=False,
                 n_zero_basis: int = 2, num_basis_outside: int = 2,
                 off_set=0):
        BasisGenerator.__init__(self, phase_generator, num_basis)

        self.basis_bandwidth_factor = basis_bandwidth_factor
        self.n_basis_outside = num_basis_outside
        self.n_zero_basis = n_zero_basis

        n_add_basis = 0
        if zero_start:
            n_add_basis += n_zero_basis
        if zero_goal:
            n_add_basis += n_zero_basis

        if not zero_start and not zero_goal:
            basis_dist = duration / (self.num_basis - 2 * self.n_basis_outside - 1)

            time_points = np.linspace(-self.n_basis_outside * basis_dist,
                                      duration + self.n_basis_outside * basis_dist,
                                      self.num_basis)
        else:
            time_points = np.linspace(off_set,
                                      duration + off_set,
                                      self.num_basis + n_add_basis)

        self.centers = self.phase_generator.phase(time_points)

        tmp_bandwidth = np.hstack((self.centers[1:] - self.centers[0:-1],
                                   self.centers[-1] - self.centers[- 2]))

        # The centers should not overlap too much (makes w almost random due to aliasing effect). Empirically chosen
        self.bandwidth = self.basis_bandwidth_factor / (tmp_bandwidth ** 2)

        self.zero_start = zero_start
        self.zero_goal = zero_goal

    def basis(self, time):

        if isinstance(time, (float, int)):
            time = np.array([time])

        phase = self.phase_generator.phase(time)

        diff_sqr = (phase[:, None] - self.centers[None, :]) ** 2 * self.bandwidth[None, :]
        basis = np.exp(- diff_sqr / 2)

        sum_b = np.sum(basis, axis=1)
        basis = basis / sum_b[:, None]
        return basis
        # return np.array(basis).transpose()

    def basis_and_der(self, time):
        phase = self.phase_generator.phase(time)

        diffs = phase[:, None] - self.centers[None, :]

        basis = np.exp(- diffs ** 2 * self.bandwidth[None, :] / 2)
        db_dz = - diffs * self.bandwidth[None, :] * basis

        sum_b = np.sum(basis, axis=1)[:, None]
        sum_db_dz = np.sum(db_dz, axis=1)[:, None]

        basis_der = (db_dz * sum_b - basis * sum_db_dz) / sum_b ** 2
        basis = basis / sum_b

        return basis, basis_der

    def basis_and_der_multi_dof(self, time, num_dof):
        basis_single_dof, basis_der_single_dof = self.basis_and_der(time)

        basis_multi_dof = np.zeros((basis_single_dof.shape[0] * num_dof, basis_single_dof.shape[1] * num_dof))
        basis_der_multi_dof = np.zeros((basis_der_single_dof.shape[0] * num_dof, basis_der_single_dof.shape[1] * num_dof))

        for i in range(num_dof):
            row_indices = slice(i * basis_single_dof.shape[0], (i + 1) * basis_single_dof.shape[0])
            column_indices = slice(i * basis_single_dof.shape[1], (i + 1) * basis_single_dof.shape[1])

            basis_multi_dof[row_indices, column_indices] = basis_single_dof
            basis_der_multi_dof[row_indices, column_indices] = basis_der_single_dof

        return basis_multi_dof, basis_der_multi_dof


# TODO: some things still missing
class NormalizedRhythmicBasisGenerator(BasisGenerator):

    def __init__(self, phase_generator, num_basis=10,
                 duration=1, basis_bandwidth_factor=3):

        BasisGenerator.__init__(self, phase_generator, num_basis)

        self.num_bandwidth_factor = basis_bandwidth_factor
        self.centers = np.linspace(0, 1, self.num_basis)

        tmp_bandwidth = np.hstack((self.centers[1:] - self.centers[0:-1],
                                   self.centers[-1] - self.centers[- 2]))

        # The Centers should not overlap too much (makes w almost random due to aliasing effect).Empirically chosen
        self.bandwidth = self.num_bandwidth_factor / (tmp_bandwidth ** 2)

    def basis(self):

        phase = self.getInputTensorIndex(0)

        diff = np.array([np.cos((phase - self.centers) * self.bandwidth * 2 * np.pi)])
        basis = np.exp(diff)

        sum_b = np.sum(basis, axis=1)
        basis = [column / sum_b for column in basis.transpose()]
        return np.array(basis).transpose()
