from mp_lib.det_promp import DeterministicProMP
import time
import numpy as np
import matplotlib.pyplot as plt


num_basis = 8
num_dof = 3

# phaseGenerator = PhaseGenerator(dataManager)
# phase_generator = ExpDecayPhaseGenerator()
# basis_generator = DMPBasisGenerator(phase_generator, num_basis=num_basis)
trajectory_generator = DeterministicProMP(num_basis, width=0.0035, off=0.01)

# numSamples = 50  # number of trajectories

# trajectory_generator.dmp_goal_pos = np.random.normal(loc=0.0, scale=1.0, size=(1, num_dof))
# trajectory_generator.dmp_weights = np.random.normal(0.0, 10.0, (num_basis, num_dof))

trajectory_generator.set_weights(3.5, np.random.normal(0.0, 10.0, (num_basis, num_dof)))

trajectory_generator.visualize(500)

start = time.process_time()

t = np.linspace(0, 1, 1750)
something1, ref_pos, ref_vel, something2 = trajectory_generator.compute_trajectory(500, 1)

plt.figure()
plt.plot(t, ref_pos)
plt.show()

# ref_pos = np.zeros(shape=(200, 5))
# ref_pos[:, 0] = np.pi
#
# timeNeeded = (time.process_time() - start)
# print('Trajectory computation: %f' % timeNeeded)
#
# plt.figure()
# plt.plot(t, ref_pos)
# plt.xlabel('time')
# plt.title('MAP sampling')
#
# plt.figure()
# # plt.plot(basis_generator.basis(t))
#
# trajectory_generator.learn(t, ref_pos)
# something1, ref_pos_learned, ref_vel_learned, something2 = trajectory_generator.compute_trajectory(100, 1)
#
# plt.figure()
# plt.plot(ref_pos)
# plt.plot(ref_pos_learned, '--')
#
# plt.show()
#
# print('PhaseGeneration Done')
