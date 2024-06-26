from water_pimd_module import *
from analysis import *
import time
import uuid
import argparse
from utils import params_to_json

start_time = time.time()
unique_id = uuid.uuid4()
#### Main code 
#steps,  equilibration_steps, skip_steps, gamma0, dt, temperature, P, pdb_file, forcefield_file, platform_name
parser = argparse.ArgumentParser(description='Water PIMD simulation')
parser.add_argument('--steps', type=int, default=10000, help='Total number of simulation steps.')
parser.add_argument('--equilibration_steps', type=int, default=1000, help='Number of steps for equilibration.')
parser.add_argument('--skip_steps', type=int, default=1, help='Number of steps to skip for data collection.')
parser.add_argument('--gamma0', type=float, default=(1.0 / 0.17) , help='Friction coefficient in 1/ps.')
parser.add_argument('--dt', type=float, default=0.12, help='Time step for the simulation in femtoseconds.')
parser.add_argument('--temperature', type=float, default=20.0, help='Simulation temperature in Kelvin.')
parser.add_argument('--P', type=int, default=100, help='Number of beads in the Path Integral formulation.')
args = parser.parse_args()

error_tolerance = 1
# run the simulation, save the results, and analyze the results, 
# if the error is greater than 1% of the mean value, estimate the additional steps required to reduce the error to 1% of the mean value
# and run the simulation again
# repeat the process until the error is less than 1% of the mean value
# save the results and the metadata of the simulation based on uuid
#save_dir = os.path.join("Result", str(unique_id))


simulation_run = simulation_run_time(uuid=unique_id,
                                     steps=args.steps, 
                                     equilibration_steps=args.equilibration_steps, 
                                     skip_steps=args.skip_steps , 
                                     gamma0=args.gamma0/ unit.picoseconds, 
                                     dt=args.dt* unit.femtoseconds, 
                                     temperature=args.temperature, 
                                     P=args.P,)
                                     #save_dir=save_dir)
simulation_run.run()
sim_analysis = simulation_state(params = simulation_run.metadata,
                                PE = simulation_run.PE,
                                POS = simulation_run.POS,
                                FORCES = simulation_run.FORCES,
                                result_dir = simulation_run.save_dir)
sim_analysis.analyze()
error_percentage = sim_analysis.error_percent

while error_percentage > error_tolerance:
    additional_steps = sim_analysis.estimate_additional_steps()
    print(f"Additional steps required: {additional_steps}")
    simulation_run.continue_simulations(additional_steps = additional_steps)
    simulation_run.save()
    sim_analysis = simulation_state(params = simulation_run.metadata,
                                    PE = simulation_run.PE,
                                    POS = simulation_run.POS,
                                    FORCES = simulation_run.FORCES,
                                    result_dir = simulation_run.save_dir)
    sim_analysis.analyze()
    error_percentage = sim_analysis.error_percent

print(f"Total steps taken: {simulation_run.steps}")
sim_analysis.save()    

end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds")
