from __future__ import print_function
from water_pimd_module import *
import time
import uuid
import os

start_time = time.time()
unique_id = uuid.uuid4()
#### Main code 
params = initialize_parameters()
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

end_time = time.time()
print (end_time - start_time)