from openmm import app, unit
from openmm.unit import Quantity
import openmm as mm
import sys
from tqdm import tqdm
import numpy as np
import mdtraj
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

hbar = 1.054571817e-34*6.0221367e23/(1000*1e-12)

def initialize_parameters(steps=1000,
                          equilibration_steps=100,
                          skip_steps=1,
                          gamma0=(1.0 / 0.17) / unit.picoseconds,
                          dt=0.12 * unit.femtoseconds,
                          temperature=50.0,
                          P=8192,
                          pdb_file="qtip4p.pdb",
                          forcefield_file='qtip4pf.xml',
                          platform_name='Reference'):
    """
    Initializes and returns simulation parameters.

    Returns:
        dict: A dictionary containing key simulation parameters, including:
              - steps (int): Total number of simulation steps.
              - equilibration_steps (int): Number of steps for equilibration.
              - skip_steps (int): Number of steps to skip for data collection.
              - gamma0 (Quantity): Friction coefficient in 1/ps.
              - dt (Quantity): Time step for the simulation in femtoseconds.
              - temperature (float): Simulation temperature in Kelvin.
              - P (int): Number of beads in the Path Integral formulation.
              - pdb_file (str): Path to the PDB file for system setup.
              - forcefield_file (str): Path to the forcefield XML file.
              - platform_name (str): Name of the computation platform ('Reference' or 'CPU').
              - beta (Quantity): Inverse temperature beta = 1/(kT).
              - tau (Quantity): Imaginary time step tau = beta/P.
    """
    params = {
        'steps': steps,
        'equilibration_steps': equilibration_steps,
        'skip_steps': skip_steps,
        'gamma0': gamma0,
        'dt': dt,
        'temperature': temperature,  # in Kelvin
        'P': P,
        'pdb_file': pdb_file,
        'forcefield_file': forcefield_file,
        'platform_name': platform_name,  # or 'CPU' for faster, multi-core computations
    }
    params['beta'] = (1000.0 / (params['temperature'] * 8.31415))
    params['tau'] = params['beta'] / params['P']
    return params


def setup_system(params):
    """
    Sets up the simulation system based on the specified parameters.

    Parameters:
        params (dict): Dictionary of simulation parameters returned by `initialize_parameters`.

    Returns:
        Simulation: An OpenMM Simulation object ready for execution.
    """
    pdb = app.PDBFile(params['pdb_file'])
    forcefield = app.ForceField(params['forcefield_file'])
    system = forcefield.createSystem(pdb.topology, rigidWater=False)
    integrator = mm.RPMDIntegrator(params['P'], params['temperature'] * unit.kelvin, params['gamma0'], params['dt'])
    platform = mm.Platform.getPlatformByName(params['platform_name'])
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.context.computeVirtualSites()
    state = simulation.context.getState(getForces=True, getEnergy=True, getPositions=True)
    potential_energy = state.getPotentialEnergy()
    potential_energy.in_units_of(unit.kilocalorie_per_mole)
    kilocalorie_per_mole_per_angstrom = unit.kilocalorie_per_mole/unit.angstrom 
    mm.LocalEnergyMinimizer.minimize(simulation.context, 1e-1)
    simulation.context.setVelocitiesToTemperature(params["temperature"]*unit.kelvin)
    return simulation

def setup_reporters(simulation, params):
    """
    Sets up reporters for the simulation to output state data and trajectories.

    Parameters:
        simulation (Simulation): The OpenMM Simulation object.
        params (dict): Dictionary of simulation parameters including 'steps' and 'skip_steps'.
    """
    simulation.reporters.append(app.StateDataReporter(sys.stdout, params['steps'] / 10, step=True, 
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, progress=True, remainingTime=True, 
        speed=True, totalSteps=params['steps'], separator='\t'))
    simulation.reporters.append(mdtraj.reporters.HDF5Reporter('water.h5', params['skip_steps']))

def minimize_and_equilibrate(simulation, params):
    """
    Minimizes the energy of the system and equilibrates it at the specified temperature.

    Parameters:
        simulation (Simulation): The OpenMM Simulation object.
        params (dict): Dictionary of simulation parameters including 'temperature' and 'equilibration_steps'.
    """
    mm.LocalEnergyMinimizer.minimize(simulation.context, 1e-1)
    simulation.context.setVelocitiesToTemperature(params['temperature'] * unit.kelvin)
    simulation.step(params['equilibration_steps'])

def step_simulation(simulation, params):
    """
    Advances the simulation by the specified number of steps, skipping according to 'skip_steps'.

    Parameters:
        simulation (Simulation): The OpenMM Simulation object.
        params (dict): Dictionary of simulation parameters including 'steps' and 'skip_steps'.
    """
    simulation_steps = int(params['steps'] / params['skip_steps'])
    PE = np.zeros((simulation_steps, params['P']))
    POS = np.zeros((simulation_steps, params['P'], 4, 3))
    FORCES = np.zeros((simulation_steps, params['P'], 4, 3))
    for step in tqdm(range(simulation_steps)):
        simulation.step(params['skip_steps'])
        pe = np.zeros(params['P'])
        pos = np.zeros((params['P'], 4, 3))
        forces = np.zeros((params['P'], 4, 3))
        for beadi in range(params['P']):
            result = process_bead(simulation, beadi)
            pe_i, posi, forces_i = result
            # posi, force_i shape is (4,3)
            pe[beadi] = pe_i
            pos[beadi] = posi
            forces[beadi] = forces_i
        PE[step] = pe
        POS[step] = pos
        FORCES[step] = forces
    return PE, POS, FORCES

def process_bead(simulation, beadi):
    current_state = simulation.integrator.getState(beadi, getPositions=True, getEnergy=True, getForces=True)
    pe_i = current_state.getPotentialEnergy() / unit.kilojoules_per_mole
    posi = np.array(current_state.getPositions(asNumpy=True))
    forces_i = np.array(current_state.getForces() / (unit.kilojoules_per_mole / unit.nanometer))
    return pe_i, posi, forces_i

def params_to_json(params,dir="Result"):
    """
    Saves the simulation parameters to a JSON file.

    Parameters:
        params (dict): Dictionary of simulation parameters.
    """
    with open(dir+"/params.json", 'w') as f:
        json.dump(params, f, indent=4)