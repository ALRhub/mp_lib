import numpy as np
import matplotlib.pyplot as plt
import mp_lib.phase as mpl_phase
import mp_lib.basis as mpl_basis
import mp_lib.promps as mpl_promps

phase_generator = mpl_phase.LinearPhaseGenerator()
basis_generator = mpl_basis.NormalizedRBFBasisGenerator(phase_generator,
                                                        num_basis=5,
                                                        duration=1,
                                                        basis_bandwidth_factor=3,
                                                        num_basis_outside=1)
time = np.linspace(0, 1, 100)
n_dof = 7
promp = mpl_promps.ProMP(basis_generator, phase_generator, n_dof)   # 3 argument = nDOF
trajectories = promp.get_trajectory_samples(time, n_samples=4)   # 2nd argument is numSamples/Demonstrations/trajectories
mean_traj, cov_traj = promp.get_mean_and_covariance_trajectory(time)
plotDof = 2
plt.figure()
plt.plot(time, trajectories[:, plotDof, :])
#
# plt.figure()
# plt.plot(time, meanTraj[:, 0])

learnedProMP = mpl_promps.ProMP(basis_generator, phase_generator, n_dof)
learner = mpl_promps.MAPWeightLearner(learnedProMP)
trajectoriesList = []
timeList = []

for i in range(trajectories.shape[2]):
    trajectoriesList.append(trajectories[:, :, i])
    timeList.append(time)

learner.learn_from_data(trajectoriesList, timeList)
trajectories = learnedProMP.get_trajectory_samples(time, 10)
plt.figure()
plt.plot(time, trajectories[:, plotDof, :])
plt.xlabel('time')
plt.title('MAP sampling')

phaseGeneratorSmooth = mpl_phase.SmoothPhaseGenerator(duration = 1)
proMPSmooth = mpl_promps.ProMP(basis_generator, phaseGeneratorSmooth, n_dof)
proMPSmooth.mu = learnedProMP.mu
proMPSmooth.cov_mat = learnedProMP.cov_mat

trajectories = proMPSmooth.get_trajectory_samples(time, 10)
plt.plot(time, trajectories[:, plotDof, :], '--')

################################################################

# Conditioning in JointSpace
desiredTheta = np.array([0.5, 0.7, 0.9, 0.2, 0.6, 0.8, 0.1])
desiredVar = np.eye(len(desiredTheta)) * 0.0001
newProMP = promp.joint_space_conditioning(0.5, desired_theta=desiredTheta, desired_var=desiredVar)
trajectories = newProMP.get_trajectory_samples(time, 4)
plt.figure()
plt.plot(time, trajectories[:, plotDof, :])
plt.xlabel('time')
plt.title('Joint-Space conditioning')
newProMP.plot_promp(time, [3, 4])

plt.show()
