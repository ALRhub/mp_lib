import matplotlib.pyplot as plt
import numpy as np
import mp_lib

phaseGenerator = mp_lib.phase.LinearPhaseGenerator()
basisGenerator = mp_lib.basis.NormalizedRBFBasisGenerator(phaseGenerator,
                                                          num_basis=10,
                                                          duration=1,
                                                          basis_bandwidth_factor=3,
                                                          num_basis_outside=1)

time = np.linspace(0, 1, 100)
basis = basisGenerator.basis(time)
basis_multi_dof = basisGenerator.basis_multi_dof(time, 3)

plt.figure()
plt.plot(time, basis)

plt.show()

print('done')
