# SPDX-FileCopyrightText: Copyright © duneuro-py contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-duneuro-py-exception OR LGPL-3.0-or-later
set(PYTHON_REFERENCE_SOLUTIONS
  eeg_local_subtraction_reference_solution_cg_tet.txt
  meg_local_subtraction_reference_solution_cg_tet.txt)
dune_symlink_to_source_files(FILES ${PYTHON_REFERENCE_SOLUTIONS})
