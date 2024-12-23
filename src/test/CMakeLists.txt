# SPDX-FileCopyrightText: Copyright © duneuro contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-duneuro-exception OR LGPL-3.0-or-later
add_subdirectory(example_data)
add_subdirectory(reference_solutions)

set(PYTHON_TEST_SCRIPTS
  eeg_direct_test.py
  eeg_transfer_test.py
  meg_direct_test.py
  meg_transfer_test.py)
dune_symlink_to_source_files(FILES ${PYTHON_TEST_SCRIPTS})

dune_python_add_test(NAME eeg_direct_test
                     SCRIPT eeg_direct_test.py
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

dune_python_add_test(NAME eeg_transfer_test
                     SCRIPT eeg_transfer_test.py
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

dune_python_add_test(NAME meg_direct_test
                     SCRIPT meg_direct_test.py
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

dune_python_add_test(NAME meg_transfer_test
                     SCRIPT meg_transfer_test.py
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     TIMEOUT 600)
