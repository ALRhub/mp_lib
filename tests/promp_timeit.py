import time
import numpy as np
import mp_lib.phase as mpl_phase
import mp_lib.basis as mpl_basis
import mp_lib.promps as mpl_promps


n_basis = 5
n_dof = 3
duration = 2
dt = 0.02
num_time_steps = int(duration/dt)

phase_generator = mpl_phase.LinearPhaseGenerator(duration=duration)
basis_generator = mpl_basis.NormalizedRBFBasisGenerator(phase_generator,
                                                        zero_start=False,
                                                        zero_goal=False,
                                                        num_basis=n_basis,
                                                        duration=duration)
t = np.linspace(0, duration, num_time_steps)

promp = mpl_promps.ProMP(num_dof=n_dof, num_basis=n_basis, duration=duration, dt=dt,
                         basis_generator=basis_generator, phase_generator=phase_generator)   # 3 argument = nDOF

# weights = np.random.normal(0.0, 100.0, (n_basis * n_dof, 1))
weights = np.arange(n_basis * n_dof)
promp.set_weights(weights)

t_start = time.time()
for i in range(10000):
    des_pos, des_vel = promp.reference_trajectory(t)
t_end = time.time()
print((t_end - t_start) / 10000)

t_start = time.time()
for i in range(10000):
    des_pos, des_vel = promp.reference_trajectory_weight_matrix(t)
t_end = time.time()
print((t_end - t_start) / 10000)
