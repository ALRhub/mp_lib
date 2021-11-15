from mp_lib.phase import ExpDecayPhaseGenerator
from mp_lib.basis import DMPBasisGenerator
from mp_lib.dmps import DMP
import time
import numpy as np
import matplotlib.pyplot as plt


num_basis = 5
num_dof = 3
duration = 5
dt = 0.002
num_time_steps = int(duration/dt)

# phaseGenerator = PhaseGenerator(dataManager)
phase_generator = ExpDecayPhaseGenerator(duration=duration, alpha_phase=3)
basis_generator = DMPBasisGenerator(phase_generator, num_basis=num_basis, duration=duration)
trajectory_generator = DMP(num_dof=num_dof, num_basis=num_basis, duration=duration, dt=dt, basis_generator=basis_generator, phase_generator=phase_generator)

# numSamples = 50  # number of trajectories

t = np.linspace(0, duration, num_time_steps)
phase = phase_generator.phase(t)
basis = basis_generator.basis(t)
plt.figure()
plt.plot(t, phase)
plt.plot(t, basis)
plt.show()

goal_pos = np.random.normal(loc=0.0, scale=1.0, size=(num_dof))
weights = np.random.normal(0.0, 100.0, (num_basis * num_dof, 1))
trajectory_generator.set_weights(weights, goal_pos)

start = time.process_time()

t = np.linspace(0, duration, num_time_steps)
ref_pos, ref_vel = trajectory_generator.reference_trajectory(t)

plt.figure()
plt.plot(t, ref_pos)
plt.plot(duration * np.ones(num_dof), goal_pos.flatten(), "x")
plt.show()

# timeNeeded = (time.process_time() - start)
# print('Trajectory computation: %f' % timeNeeded)
#
# plt.figure()
# plt.plot(t, ref_pos)
# plt.xlabel('time')
# plt.title('MAP sampling')
#
# plt.figure()
# plt.plot(basis_generator.basis(t))
#
# trajectory_generator.learn_from_imitation(t, ref_pos)
# ref_pos_learned, ref_vel_learned = trajectory_generator.reference_trajectory(t)
#
# plt.figure()
# plt.plot(ref_pos)
# plt.plot(ref_pos_learned, '--')
#
# plt.show()
#
# print('PhaseGeneration Done')
