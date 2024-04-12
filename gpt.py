from openmm import *
from openmm.app import *
from openmm.unit import *
import openmm as mm
from sys import stdout
from openmm import app, unit

# Load the PDB file
pdb = PDBFile('qtip4p.pdb')

# Create a system
forcefield = ForceField('qtip4pf.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

# Convert the system to use path integrals
nBeads = 4  # Number of path integral beads; adjust based on desired accuracy and computational budget
PIsystem = mm.app.internal.pathintegrals.PathIntegralForce(system, nBeads)

# Create an integrator
temperature = 300*unit.kelvin
frictionCoeff = 1/unit.picoseconds
stepSize = 0.002*unit.picoseconds
integrator = mm.app.internal.pathintegrals.PathIntegralLangevinIntegrator(temperature, frictionCoeff, stepSize, nBeads)

# Set up the simulation
simulation = Simulation(pdb.topology, PIsystem, integrator)
simulation.context.setPositions(pdb.positions)

# Minimize energy
simulation.minimizeEnergy()

# Equilibration: Skip or perform a short equilibration here if desired

# Production run
simulation.reporters.append(PDBReporter('output.pdb', 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=10000, separator='\t'))
simulation.step(10000)

print("Simulation complete!")
