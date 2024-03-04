from water_pimd_module import *
from analysis import *
import time
import uuid
import os
import argparse
from utils import params_to_json

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

simulation_run = simulation_run_time(uuid=unique_id,
                                     steps=args.steps, 
                                     equilibration_steps=args.equilibration_steps, 
                                     skip_steps=args.skip_steps , 
                                     gamma0=args.gamma0/ unit.picoseconds, 
                                     dt=args.dt* unit.femtoseconds, 
                                     temperature=args.temperature, 
                                     P=args.P)

simulation_run.run()
simulation_run.save()
sim_analysis = simulation_state()
sim_analysis.analyze()


end_time = time.time()
print (end_time - start_time)