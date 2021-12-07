import numpy as np
import matplotlib.pyplot as plt
import mp_lib.phase as mpl_phase
import mp_lib.basis as mpl_basis
import mp_lib.promps as mpl_promps


n_basis = 5
n_dof = 3
duration = 2
dt = 0.02
num_time_steps = int(duration/dt)

weights = np.arange(n_basis * n_dof)

phase_generator = mpl_phase.LinearPhaseGenerator(duration=duration)
basis_generator = mpl_basis.NormalizedRBFBasisGenerator(phase_generator,
                                                        zero_start=True,
                                                        zero_goal=True,
                                                        num_basis=n_basis,
                                                        duration=duration)
t = np.linspace(0, duration, num_time_steps)

promp = mpl_promps.ProMP(num_dof=n_dof, num_basis=n_basis, duration=duration, dt=dt,
                         basis_generator=basis_generator, phase_generator=phase_generator)   # 3 argument = nDOF

promp.set_weights(weights)

des_pos, des_vel = promp.reference_trajectory(t)

plt.figure()
plt.plot(des_pos)
# plt.show()
plt.figure()
plt.plot(des_vel)
# plt.show()

# shorter time, should see increase in velocity
duration = 1
dt = 0.02
num_time_steps = int(duration/dt)

phase_generator = mpl_phase.LinearPhaseGenerator(duration=duration)
basis_generator = mpl_basis.NormalizedRBFBasisGenerator(phase_generator,
                                                        zero_start=True,
                                                        zero_goal=True,
                                                        num_basis=n_basis,
                                                        duration=duration)
t = np.linspace(0, duration, num_time_steps)

promp = mpl_promps.ProMP(num_dof=n_dof, num_basis=n_basis, duration=duration, dt=dt,
                         basis_generator=basis_generator, phase_generator=phase_generator)   # 3 argument = nDOF

promp.set_weights(weights)

des_pos, des_vel = promp.reference_trajectory(t)

plt.figure()
plt.plot(des_pos)
# plt.show()
plt.figure()
plt.plot(des_vel)
plt.show()

input()
