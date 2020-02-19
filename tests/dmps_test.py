from mp_lib.phase import ExpDecayPhaseGenerator
from mp_lib.basis import DMPBasisGenerator
from mp_lib.dmps import DMP
import time
import numpy as np
import matplotlib.pyplot as plt


num_basis = 10
num_dof = 4

# phaseGenerator = PhaseGenerator(dataManager)
phase_generator = ExpDecayPhaseGenerator()
basis_generator = DMPBasisGenerator(phase_generator, num_basis=num_basis)
trajectory_generator = DMP(basis_generator, phase_generator, num_dof=num_dof)

# numSamples = 50  # number of trajectories

trajectory_generator.dmp_goal_pos = np.random.normal(loc=0.0, scale=1.0, size=(1, num_dof))
trajectory_generator.dmp_weights = np.random.normal(0.0, 10.0, (num_basis, num_dof))

start = time.process_time()

t = np.linspace(0, 1, 100)
ref_pos, ref_vel = trajectory_generator.reference_trajectory(t)

timeNeeded = (time.process_time() - start)
print('Trajectory computation: %f' % timeNeeded)

plt.figure()
plt.plot(t, ref_pos)
plt.xlabel('time')
plt.title('MAP sampling')

plt.figure()
plt.plot(basis_generator.basis(t))

trajectory_generator.learn_from_imitation(t, ref_pos)
ref_pos_learned, ref_vel_learned = trajectory_generator.reference_trajectory(t)

plt.figure()
plt.plot(ref_pos)
plt.plot(ref_pos_learned, '--')

plt.show()

print('PhaseGeneration Done')
