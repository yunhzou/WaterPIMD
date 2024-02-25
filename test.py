from __future__ import print_function
from analysis import *
import time
import os
start_time = time.time()
save_dir = r"Result"
# Analysis
sim = simulation_state(save_dir)
sim.compute_PE()
sim.compute_K_virial()
sim.compute_H2O_structure()
sim.compute_E_virial()
print (f"E_virial_avg: {sim.E_virial_avg}")
#save the result as result.npz
np.savez(os.path.join(save_dir, "result.npz"), geometry=sim.geometry, E_virial_avg= np.array(sim.E_virial_avg))
end_time = time.time()
print (end_time - start_time)