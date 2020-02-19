import numpy as np
import mp_lib.phase as mpl_phase
import mp_lib.basis as mpl_basis
import scipy.stats as stats
from mp_lib.utils import plot_mean_and_std
# from typing import Type


class ProMP:

    def __init__(self,
                 basis_generator: mpl_basis.BasisGenerator,
                 phase_generator: mpl_phase.PhaseGenerator,
                 num_dof: int):
        self.basis = basis_generator
        self.phase = phase_generator
        self.num_dof = num_dof
        self.num_weights = basis_generator.num_basis * self.num_dof
        self.mu = np.zeros(self.num_weights)
        self.cov_mat = np.eye(self.num_weights)
        self.observation_sigma = np.ones(self.num_dof)

    def get_trajectory_samples(self, time, n_samples=1):
        basis_multi_dof = self.basis.basis_multi_dof(time=time, num_dof=self.num_dof)
        weights = np.random.multivariate_normal(self.mu, self.cov_mat, n_samples)
        weights = weights.transpose()
        trajectory_flat = basis_multi_dof.dot(weights)
        # a = trajectory_flat
        trajectory_flat = trajectory_flat.reshape((self.num_dof,
                                                   int(trajectory_flat.shape[0] / self.num_dof),
                                                   n_samples))
        trajectory_flat = np.transpose(trajectory_flat, (1, 0, 2))
        # trajectory_flat = trajectory_flat.reshape((a.shape[0] / self.numDoF, self.numDoF, n_samples))

        return trajectory_flat

    def get_mean_and_covariance_trajectory(self, time):
        basis_multi_dof = self.basis.basis_multi_dof(time=time, num_dof=self.num_dof)
        trajectory_flat = basis_multi_dof.dot(self.mu.transpose())
        trajectory_mean = trajectory_flat.reshape((self.num_dof, int(trajectory_flat.shape[0] / self.num_dof)))
        trajectory_mean = np.transpose(trajectory_mean, (1, 0))
        covariance_trajectory = np.zeros((self.num_dof, self.num_dof, len(time)))

        for i in range(len(time)):

            basis_single_t = basis_multi_dof[slice(i, (self.num_dof - 1) * len(time) + i + 1, len(time)), :]
            covariance_time_step = basis_single_t.dot(self.cov_mat).dot(basis_single_t.transpose())
            covariance_trajectory[:, :, i] = covariance_time_step

        return trajectory_mean, covariance_trajectory

    def get_mean_and_std_trajectory(self, time):
        basis_multi_dof = self.basis.basis_multi_dof(time=time, num_dof=self.num_dof)
        trajectory_flat = basis_multi_dof.dot(self.mu.transpose())
        trajectory_mean = trajectory_flat.reshape((self.num_dof, trajectory_flat.shape[0] // self.num_dof))
        trajectory_mean = np.transpose(trajectory_mean, (1, 0))
        std_trajectory = np.zeros((len(time), self.num_dof))

        for i in range(len(time)):

            basis_single_t = basis_multi_dof[slice(i, (self.num_dof - 1) * len(time) + i + 1, len(time)), :]
            covariance_time_step = basis_single_t.dot(self.cov_mat).dot(basis_single_t.transpose())
            std_trajectory[i, :] = np.sqrt(np.diag(covariance_time_step))

        return trajectory_mean, std_trajectory

    def get_mean_and_covariance_trajectory_full(self, time):
        basis_multi_dof = self.basis.basis_multi_dof(time=time, num_dof=self.num_dof)

        mean_flat = basis_multi_dof.dot(self.mu.transpose())
        covariance_trajectory = basis_multi_dof.dot(self.cov_mat).dot(basis_multi_dof.transpose())

        return mean_flat, covariance_trajectory

    def joint_space_conditioning(self, time, desired_theta, desired_var):
        new_promp = ProMP(self.basis, self.phase, self.num_dof)
        basis_matrix = self.basis.basis_multi_dof(time, self.num_dof)
        temp = self.cov_mat.dot(basis_matrix.transpose())
        L = np.linalg.solve(desired_var + basis_matrix.dot(temp), temp.transpose())
        L = L.transpose()
        new_promp.mu = self.mu + L.dot(desired_theta - basis_matrix.dot(self.mu))
        new_promp.cov_mat = self.cov_mat - L.dot(basis_matrix).dot(self.cov_mat)
        return new_promp

    def get_trajectory_log_likelihood(self, time, trajectory):

        trajectory_flat = trajectory.transpose().reshape(trajectory.shape[0] * trajectory.shape[1])
        mean_flat, covariance_trajectory = self.get_mean_and_covariance_trajectory_full(time)

        return stats.multivariate_normal.logpdf(trajectory_flat, mean=mean_flat, cov=covariance_trajectory)

    def get_weights_log_likelihood(self, weights):

        return stats.multivariate_normal.logpdf(weights, mean=self.mu, cov=self.cov_mat)

    def plot_promp(self, time, indices=None):
        trajectory_mean, std_trajectory = self.get_mean_and_std_trajectory(time)

        plot_mean_and_std(time, trajectory_mean, std_trajectory, indices)


class MAPWeightLearner:

    def __init__(self, promp: ProMP, regularization_coeff=10**-9, prior_covariance=10**-4, prior_weight=1):
        self.promp = promp
        self.prior_covariance = prior_covariance
        self.prior_weight = prior_weight
        self.regularization_coeff = regularization_coeff

    def learn_from_data(self, trajectory_list, time_list):

        num_traj = len(trajectory_list)
        weight_matrix = np.zeros((num_traj, self.promp.num_weights))
        for i in range(num_traj):

            trajectory = trajectory_list[i]
            time = time_list[i]
            trajectory_flat = trajectory.transpose().reshape(trajectory.shape[0] * trajectory.shape[1])
            basis_matrix = self.promp.basis.basis_multi_dof(time, self.promp.num_dof)
            temp = basis_matrix.transpose().dot(basis_matrix) \
                   + np.eye(self.promp.num_weights) * self.regularization_coeff
            weight_vector = np.linalg.solve(temp, basis_matrix.transpose().dot(trajectory_flat))
            weight_matrix[i, :] = weight_vector

        self.promp.mu = np.mean(weight_matrix, axis=0)

        sample_cov = np.cov(weight_matrix.transpose())
        self.promp.covMat = (num_traj * sample_cov + self.prior_covariance * np.eye(self.promp.num_weights)) \
                            / (num_traj + self.prior_covariance)
