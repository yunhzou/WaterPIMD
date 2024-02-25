from __future__ import print_function
from water_pimd_module import *
from analysis import *
import time
import uuid
import os

start_time = time.time()
unique_id = uuid.uuid4()
#### Main code 
params = initialize_parameters()
params = params | {'uuid': unique_id}
simulation = setup_system(params) 
setup_reporters(simulation, params)
minimize_and_equilibrate(simulation, params)
PE, POS, FORCES = step_simulation(simulation, params)
#save the result as pickle
save_dir = r"Result"
np.save(os.path.join(save_dir, "pe.npy"), PE)
np.save(os.path.join(save_dir, "pos.npy"), POS)
np.save(os.path.join(save_dir, "forces.npy"), FORCES)
print("Simulation completed successfully")

# Analysis
sim = simulation_state(save_dir)
sim.compute_PE()
sim.compute_K_virial()
sim.compute_H2O_structure()
sim.compute_E_virial
#save the result as result.npz
np.savez(os.path.join(save_dir, "result.npz"), geometry=sim.geometry, E_virial= np.array(sim.E_virial))
end_time = time.time()
print (end_time - start_time)