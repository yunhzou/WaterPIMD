from __future__ import print_function
from water_pimd_module import *
from analysis import *
import time
import uuid
import os
import argparse

start_time = time.time()
unique_id = uuid.uuid4()
#### Main code 
#steps,  equilibration_steps, skip_steps, gamma0, dt, temperature, P, pdb_file, forcefield_file, platform_name
parser = argparse.ArgumentParser(description='Water PIMD simulation')
parser.add_argument('--steps', type=int, default=1000, help='Total number of simulation steps.')
parser.add_argument('--equilibration_steps', type=int, default=100, help='Number of steps for equilibration.')
parser.add_argument('--skip_steps', type=int, default=1, help='Number of steps to skip for data collection.')
parser.add_argument('--gamma0', type=float, default=(1.0 / 0.17) , help='Friction coefficient in 1/ps.')
parser.add_argument('--dt', type=float, default=0.12, help='Time step for the simulation in femtoseconds.')
parser.add_argument('--temperature', type=float, default=50.0, help='Simulation temperature in Kelvin.')
parser.add_argument('--P', type=int, default=8192, help='Number of beads in the Path Integral formulation.')
args = parser.parse_args()

params = initialize_parameters(steps=args.steps, 
                               equilibration_steps=args.equilibration_steps, 
                               skip_steps=args.skip_steps , 
                               gamma0=args.gamma0/ unit.picoseconds, 
                               dt=args.dt* unit.femtoseconds, 
                               temperature=args.temperature, 
                               P=args.P)
params = {**params, 'uuid': unique_id}
params_to_json(params)
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
sim.compute_E_virial()
print (f"E_virial: {sim.E_virial_avg}")
#save the result as result.npz
np.savez(os.path.join(save_dir, "result.npz"), geometry=sim.geometry, E_virial_average= np.array(sim.E_virial_avg))
end_time = time.time()
print (end_time - start_time)