# SPDX-FileCopyrightText: Copyright © duneuro-py contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-duneuro-py-exception OR LGPL-3.0-or-later
import numpy as np
import time
import os

tolerance = 1e-8

working_directory = os.getcwd()
duneuropy_path=os.path.abspath(f'{working_directory}/..')
print(f'Script working directory: {working_directory}')
if os.path.isfile(f'{duneuropy_path}/duneuropy.so'):
  print(f'Using duneuropy at {duneuropy_path}')
else:
  raise ModuleNotFoundError(f'Could not find duneuropy module')

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

# load data
print('Loading data')
volume_conductor_data = np.load(f'{working_directory}/example_data/tet_volume_conductor.npz')
coil_data = np.load(f'{working_directory}/example_data/tet_meg_sensors.npz')
dipole_data = np.load(f'{working_directory}/example_data/tet_dipole_superficial.npz')
reference_solution = np.loadtxt(f'{working_directory}/reference_solutions/meg_local_subtraction_reference_solution_cg_tet.txt')

nodes = volume_conductor_data['nodes']
elements = volume_conductor_data['elements']
labels = volume_conductor_data['labels']
conductivities = volume_conductor_data['conductivities']

coils = coil_data['coils']
projections = coil_data['projections']

dipole_position = dipole_data['position']
dipole_moment = dipole_data['moment']
print('Data loaded')

source_model = 'local_subtraction'

# create driver
mesh_cfg = {'nodes' : nodes, 'elements' : elements}
tensor_cfg = {'labels' : labels, 'conductivities' : conductivities}
volume_conductor_cfg = {'grid' : mesh_cfg, 'tensors' : tensor_cfg}
driver_cfg = {'type' : 'fitted', 'solver_type' : 'cg', 'element_type' : 'tetrahedron', 'post_process' : 'false', 'post_process_meg' : 'true', 'subtract_mean' : 'true'}
solver_cfg = {'reduction' : '1e-16', 'edge_norm_type' : 'houston', 'penalty' : '20', 'scheme' : 'sipg', 'weights' : 'tensorOnly'}
meg_cfg = {'intorderadd' : '5', 'type' : 'physical'}
driver_cfg['solver'] = solver_cfg
driver_cfg['volume_conductor'] = volume_conductor_cfg
driver_cfg['meg'] = meg_cfg

print('Creating driver')
meeg_driver = dp.MEEGDriver3d(driver_cfg)
print('Driver created')

# set coils and projections
print('Setting coils and projections')
# we first wrap the coils and projections into appropriate data structures
coils_duneuro = [dp.FieldVector3D(coil) for coil in coils]
projections_duneuro = []
for projection_list in projections:
  current_coil_projection_list = [dp.FieldVector3D(projection_list[x : x + 3]) for x in range(0, len(projection_list), 3)]
  projections_duneuro.append(current_coil_projection_list)
meeg_driver.setCoilsAndProjections(coils_duneuro, projections_duneuro)
print('Coils and projections set')

# setting source model
# create config
source_model_config_partial_integration = {'type' : 'partial_integration'}
source_model_config_venant = \
{
  'type' : 'multipolar_venant',
  'referenceLength' : 20,
  'weightingExponent' : 1,
  'relaxationFactor' : 1e-6,
  'restrict' : True,
  'initialization' : 'closest_vertex'
}
source_model_config_subtraction = \
{
  'type' : 'subtraction',
  'intorderadd' : 0,
  'intorderadd_lb' : 0
}
source_model_config_local_subtraction = \
{
  'type' : 'local_subtraction',
  'restrict' : False,
  'initialization' : 'single_element',
  'intorderadd_eeg_patch' : 0,
  'intorderadd_eeg_boundary' : 0,
  'intorderadd_eeg_transition' : 0,
  'intorder_meg_patch' : 0,
  'intorder_meg_boundary' : 6,
  'intorder_meg_transition' : 5,
  'extensions' : 'vertex vertex'
}
source_model_config_database = \
{
  'partial_integration' : source_model_config_partial_integration,
  'subtraction' : source_model_config_subtraction,
  'multipolar_venant' : source_model_config_venant,
  'local_subtraction' : source_model_config_local_subtraction
}

driver_cfg['source_model'] = source_model_config_database[source_model]

print('Computing secondary magnetic field')

# set up dipole. First position, then moment.
dipole = dp.Dipole3d(dipole_position, dipole_moment)

# first solve the EEG forward problem
print('Numerically computing electric potentials')
solution_storage = meeg_driver.makeDomainFunction()
meeg_driver.solveEEGForward(dipole, solution_storage, driver_cfg)
print('Potentials computed')

# now solve the MEG forward problem
print('Performing Biot-Savart integration')
numerical_solution, computation_information = meeg_driver.solveMEGForward(solution_storage, driver_cfg)
numerical_solution = np.array(numerical_solution)
print('Integration performed')
print('Seconary field computed')

def relative_error(test_solution, reference_solution):
  assert len(test_solution) == len(reference_solution)
  return np.linalg.norm(np.array(test_solution) - np.array(reference_solution)) / np.linalg.norm(reference_solution)

rel_error = relative_error(numerical_solution, reference_solution)
print(f'The relative error of the numerical solution is {rel_error}')

if rel_error > tolerance:
  raise Exception(f'Relative error not within tolerance')
