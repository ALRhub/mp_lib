import matplotlib.pyplot as plt
import numpy as np

from mp_lib.phase import ExpDecayPhaseGenerator

phaseGenerator = ExpDecayPhaseGenerator()
time = np.linspace(0, 1, 100)
phase = phaseGenerator.phase(time)

plt.figure()
plt.plot(time, phase)
# plt.hold(True)

phaseGenerator.tau = 2
phase = phaseGenerator.phase(time)
plt.plot(time, phase)

phaseGenerator.tau = 0.5
phase = phaseGenerator.phase(time)
plt.plot(time, phase)

plt.show()

print('PhaseGeneration Done')
