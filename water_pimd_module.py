from openmm import app, unit
from openmm.unit import Quantity
import openmm as mm
import sys
from tqdm import tqdm
import numpy as np
from utils import params_to_json
#import mdtraj
import os


hbar = 1.054571817e-34*6.0221367e23/(1000*1e-12)

class simulation_run_time():
    # initialize the simulation run time similar to initialize parameters function
    def __init__(self,
                 uuid: str,
                 steps=1000,
                 equilibration_steps=100,
                 skip_steps=1,
                 gamma0=(1.0 / 0.17) / unit.picoseconds,
                 dt=0.12 * unit.femtoseconds,
                 temperature=50.0,
                 P=8192,
                 pdb_file="qtip4p.pdb",
                 forcefield_file='qtip4pf.xml',
                 platform_name='Reference',
                 save_dir:str="Result"):
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
        self.save_dir = "Result"
        self.steps = steps
        self.equilibration_steps = equilibration_steps
        self.skip_steps = skip_steps
        self.gamma0 = gamma0
        self.dt = dt
        self.temperature = temperature
        self.P = P
        self.pdb_file = pdb_file
        self.forcefield_file = forcefield_file
        self.platform_name = platform_name
        self.beta = (1000.0 / (self.temperature * 8.31415))
        self.tau = self.beta / self.P
        self.uuid = uuid
        
    
    def run(self):
        self._setup_system()
        self._setup_reporters()
        self._minimize_and_equilibrate()
        PE, POS, FORCES = self._step_simulation(self.steps)
        self.PE = PE
        self.POS = POS
        self.FORCES = FORCES

    def continue_simulations(self, additional_steps):
        self.steps = self.steps + additional_steps
        PE, POS, FORCES = self._step_simulation(self.simulation)
        self.PE = np.concatenate((self.PE, PE), axis=0)
        self.POS = np.concatenate((self.POS, POS), axis=0)
        self.FORCES = np.concatenate((self.FORCES, FORCES), axis=0)

    def save(self):
        self._save_metadata()
        self._save_results()

    def _setup_system(self):
        """
        Sets up the simulation system based on the specified parameters.

        Returns:
            Simulation: An OpenMM Simulation object ready for execution.
        """
        pdb = app.PDBFile(self.pdb_file)
        forcefield = app.ForceField(self.forcefield_file)
        system = forcefield.createSystem(pdb.topology, rigidWater=False)
        integrator = mm.RPMDIntegrator(self.P, self.temperature * unit.kelvin, self.gamma0, self.dt)
        platform = mm.Platform.getPlatformByName(self.platform_name)
        simulation = app.Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)
        simulation.context.computeVirtualSites()
        state = simulation.context.getState(getForces=True, getEnergy=True, getPositions=True)
        potential_energy = state.getPotentialEnergy()
        potential_energy.in_units_of(unit.kilocalorie_per_mole)
        kilocalorie_per_mole_per_angstrom = unit.kilocalorie_per_mole/unit.angstrom
        mm.LocalEnergyMinimizer.minimize(simulation.context, 1e-1)
        simulation.context.setVelocitiesToTemperature(self.temperature*unit.kelvin)
        self.simulation = simulation
    
    def _setup_reporters(self):
        """
        Sets up reporters for the simulation to output state data and trajectories.
        """
        self.simulation.reporters.append(app.StateDataReporter(sys.stdout, self.steps / 10, step=True, 
            potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, progress=True, remainingTime=True, 
            speed=True, totalSteps=self.steps, separator='\t'))
        #simulation.reporters.append(mdtraj.reporters.HDF5Reporter('water.h5', params['skip_steps']))

    def _minimize_and_equilibrate(self):
        """
        Minimizes the energy of the system and equilibrates it at the specified temperature.

        """
        mm.LocalEnergyMinimizer.minimize(self.simulation.context, 1e-1)
        self.simulation.context.setVelocitiesToTemperature(self.temperature * unit.kelvin)
        self.simulation.step(self.equilibration_steps)

    def _step_simulation(self,steps):
        """
        Advances the simulation by the specified number of steps, skipping according to 'skip_steps'.
        """

        simulation_steps = int(steps / self.skip_steps)
        PE = np.zeros((simulation_steps, self.P))
        POS = np.zeros((simulation_steps, self.P, 4, 3))
        FORCES = np.zeros((simulation_steps, self.P, 4, 3))
        for step in tqdm(range(simulation_steps)):
            self.simulation.step(self.skip_steps)
            pe = np.zeros(self.P)
            pos = np.zeros((self.P, 4, 3))
            forces = np.zeros((self.P, 4, 3))
            for beadi in range(self.P):
                result = self._process_bead(self.simulation, beadi)
                pe_i, posi, forces_i = result
                # posi, force_i shape is (4,3)
                pe[beadi] = pe_i
                pos[beadi] = posi
                forces[beadi] = forces_i
            PE[step] = pe
            POS[step] = pos
            FORCES[step] = forces
        return PE, POS, FORCES
    
    def _process_bead(self, beadi):
        current_state = self.simulation.integrator.getState(beadi, getPositions=True, getEnergy=True, getForces=True)
        pe_i = current_state.getPotentialEnergy() / unit.kilojoules_per_mole
        posi = np.array(current_state.getPositions(asNumpy=True))
        forces_i = np.array(current_state.getForces() / (unit.kilojoules_per_mole / unit.nanometer))
        return pe_i, posi, forces_i

    def _save_metadata(self):
        """
        Saves the metadata of the simulation to a JSON file.
        """
        metadata = {
            "uuid": self.uuid,
            "steps": self.steps,
            "equilibration_steps": self.equilibration_steps,
            "skip_steps": self.skip_steps,
            "gamma0": self.gamma0,
            "dt": self.dt,
            "temperature": self.temperature,
            "P": self.P,
            "pdb_file": self.pdb_file,
            "forcefield_file": self.forcefield_file,
            "platform_name": self.platform_name,
            "beta": self.beta,
            "tau": self.tau
        }
        params_to_json(metadata, self.uuid, dir=self.save_dir)

    def _save_results(self):
        """
        Saves the results of the simulation to a compressed NumPy file.
        """
        np.save(os.path.join(self.save_dir, "pe.npy"), self.PE)
        np.save(os.path.join(self.save_dir, "pos.npy"), self.POS)
        np.save(os.path.join(self.save_dir, "forces.npy"), self.FORCES)
        print("Simulation completed successfully, data saved to", self.save_dir)
