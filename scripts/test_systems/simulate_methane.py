############################### MY CONDA ENV ##################################

# $ mamba create -n openmm-ml-taut python=3.11
# $ mamba install openmm openmm-torch openmmtools pytorch-cuda=11.8 pytorch pytest -c conda-forge -c pytorch -c nvidia
# $ pip install .
# $ pip install git+https://github.com/sef43/torchani.git@patch_openmmml_issue50

###############################################################################

from openmm.app import (
    Simulation,
    DCDReporter,
    StateDataReporter,
    PDBFile,
    Topology,
)
from openmm import System
from openmmml import MLPotential
from openmm import app
from openmm import Platform
from openmm import LangevinIntegrator
from openmm import unit
from openmm import MonteCarloBarostat
from openmmtools.constants import kB
from openmmtools.forces import FlatBottomRestraintBondForce
import os
from sys import stdout
import torch 
import sys
import openmm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# to prevent decrease of performance
torch._C._jit_set_nvfuser_enabled(False)

# define units
distance_unit = unit.angstrom
time_unit = unit.femto * unit.seconds
speed_unit = distance_unit / time_unit

# constants
stepsize = 1 * time_unit
collision_rate = unit.pico * unit.second
temperature = 300 * unit.kelvin
pressure = 1 * unit.atmosphere

# set up system
sys_name = "methane_hybrid_solv"
# xtract the topology from a PDB file
solv_system = app.PDBFile(f'test_systems/{sys_name}.pdb')
ligand_topology = solv_system.getTopology()
atoms = ligand_topology.atoms()

# get indices of tautomer 1 and tautomer 2
def get_indices(tautomer: str, ligand_topology,device) :
    indices = torch.zeros(ligand_topology.getNumAtoms(), device=device)
    # mask the hydrogen defining the other tautomer topology with a -1
    indices = torch.tensor([1 if atom.name == {"t1": "D2", "t2": "D1"}.get(tautomer) else 0 for atom in ligand_topology.atoms()])
    indices = indices.bool()
    return indices

t1_idx_mask = get_indices(tautomer="t1", ligand_topology=ligand_topology, device=device)
t2_idx_mask = get_indices(tautomer="t2", ligand_topology=ligand_topology, device=device)

# set up simulation
integrator = LangevinIntegrator(temperature, 1 / collision_rate, stepsize)
platform = Platform.getPlatformByName("CUDA")
potential = MLPotential('tautani2x', lambda_val = 1, t1_idx_mask=t1_idx_mask, t2_idx_mask=t2_idx_mask)

system = potential.createSystem(
    solv_system.getTopology(),
    implementation = "torchani"
)

#barostate = MonteCarloBarostat(unit.Quantity(1*unit.atmosphere), temperature)
barostate = MonteCarloBarostat(pressure, temperature)
system.addForce(barostate)

# get indices of heavy atom-H for tautomer 1 and tautomer 2
acceptor_t1 = next((idx for idx, atom in enumerate(ligand_topology.atoms()) if atom.name == "HET2"), None) # in this test case (methane) acceptor = donor
acceptor_t2 = next((idx for idx, atom in enumerate(ligand_topology.atoms()) if atom.name == "HET2"), None)
dummy_t1 = next((idx for idx, atom in enumerate(ligand_topology.atoms()) if atom.name == "D1"), None)
dummy_t2 = next((idx for idx, atom in enumerate(ligand_topology.atoms()) if atom.name == "D2"), None)

# add C-H dummy atom restraint
restraint_force_t1 = FlatBottomRestraintBondForce(spring_constant= 100  * unit.kilocalories_per_mole / unit.angstrom**2,
                                               well_radius= 2 * unit.angstrom,
                                               restrained_atom_index1 = acceptor_t1,  
                                               restrained_atom_index2 = dummy_t1)
restraint_force_t2 = FlatBottomRestraintBondForce(spring_constant= 100  * unit.kilocalories_per_mole / unit.angstrom**2,
                                               well_radius= 2 * unit.angstrom,
                                               restrained_atom_index1 = acceptor_t2,  
                                               restrained_atom_index2 = dummy_t2)
system.addForce(restraint_force_t1)
system.addForce(restraint_force_t2)

sim = Simulation(
    solv_system.getTopology(), 
    system, 
    integrator, 
    platform=platform,
    platformProperties={
                    "Precision": "mixed",
                    "DeviceIndex": str(1),
                },)

#base = sys.argv[0]
state_file = f"test_systems/{sys_name}_statereport.csv"
print(f"State report saved to: {state_file}")

reporter = StateDataReporter(
    state_file,
    reportInterval=10,
    step=True,
    time=True,
    potentialEnergy=True,
    totalEnergy=True,
    temperature=True,
    density=True,
    speed=True,
    elapsedTime=True
)
sim.reporters.append(reporter)

trajectory_file = f"test_systems/{sys_name}_trajectory.dcd"
print(f"Trajectory saved to: {trajectory_file}")

sim.reporters.append(
    DCDReporter(
        trajectory_file,
        reportInterval=10,
    )
)

positions = solv_system.getPositions()

# run simulation
sim.context.setPositions(positions)
sim.step(100)