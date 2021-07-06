import numpy as np
import matplotlib.pyplot as plt


class DeterministicProMP:

    def __init__(self, n_basis, n_dof, width=None, off=0.2, zero_start=False, zero_goal=False, n_zero_bases=2):
        self.n_basis = n_basis
        self.n_dof = n_dof
        self._weights = np.zeros(shape=(self.n_basis, self.n_dof))
        self.n_zero_bases = n_zero_bases
        add_basis = 0
        if zero_start:
            add_basis += n_zero_bases
        if zero_goal:
            add_basis += n_zero_bases
        self.centers = np.linspace(-off, 1. + off, self.n_basis + add_basis)
        if width is None:
            self.widths = np.ones(self.n_basis + add_basis) * ((1. + off) / (2. * (self.n_basis + add_basis)))
        else:
            self.widths = np.ones(self.n_basis + add_basis) * width
        self.scale = None
        self.zero_start = zero_start
        self.zero_goal = zero_goal

    @property
    def weights(self):
        return self._weights

    def _exponential_kernel(self, z):
        z_ext = z[:, None]
        diffs = z_ext - self.centers[None, :]
        w = np.exp(-(np.square(diffs) / (2 * self.widths[None, :])))
        w_der = -(diffs / self.widths[None, :]) * w
        w_der2 = -(1 / self.widths[None, :]) * w + np.square(diffs / self.widths[None, :]) * w
        sum_w = np.sum(w, axis=1)[:, None]
        sum_w_der = np.sum(w_der, axis=1)[:, None]
        sum_w_der2 = np.sum(w_der2, axis=1)[:, None]

        tmp = w_der * sum_w - w * sum_w_der
        return w / sum_w, tmp / np.square(sum_w), \
               ((w_der2 * sum_w - sum_w_der2 * w) * sum_w - 2 * sum_w_der * tmp) / np.power(sum_w, 3)

    def learn(self, t, pos, lmbd=1e-6):
        scale = np.max(t)
        # We normalize the timesteps to be in the interval [0, 1]
        phi = self._exponential_kernel(t / scale)[0]
        weights = np.linalg.solve(np.dot(phi.T, phi) + lmbd * np.eye(phi.shape[1]), np.dot(phi.T, pos))
        self.set_weights(scale, weights)

    def compute_trajectory(self, frequency, scale=1):
        corrected_scale = self.scale / scale
        N = int(corrected_scale * frequency)
        t = np.linspace(0, 1, N)
        pos_features, vel_features, acc_features = self._exponential_kernel(t)
        return t * corrected_scale, np.dot(pos_features, self.weights), \
               np.dot(vel_features, self.weights) / corrected_scale, \
               np.dot(acc_features, self.weights) / np.square(corrected_scale)

    # def get_weights(self):
    #     return np.copy(self.weights)

    def set_weights(self, scale, weights):
        self.scale = scale
        if self.zero_start:
            weights = np.concatenate((np.zeros((self.n_zero_bases, self.n_dof)), weights), axis=0)
        if self.zero_goal:
            weights = np.concatenate((weights, np.zeros((self.n_zero_bases, self.n_dof))), axis=0)
        self._weights = weights

    def visualize(self, frequency, scale=1):
        corrected_scale = self.scale / scale
        N = int(corrected_scale * frequency)
        t = np.linspace(0, 1, N)
        pos_features, __, __ = self._exponential_kernel(t)
        plt.plot(t, pos_features)
        plt.show()
