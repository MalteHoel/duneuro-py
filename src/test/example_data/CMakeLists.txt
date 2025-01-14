# SPDX-FileCopyrightText: Copyright © duneuro-py contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-duneuro-py-exception OR LGPL-3.0-or-later
set(EXAMPLE_PYTHON_DATA
  tet_volume_conductor.npz
  tet_electrodes.npy
  tet_meg_sensors.npz
  tet_dipole_superficial.npz)

dune_symlink_to_source_files(FILES ${EXAMPLE_PYTHON_DATA})
