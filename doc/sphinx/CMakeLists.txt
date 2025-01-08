# SPDX-FileCopyrightText: Copyright © duneuro-py contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-duneuro-py-exception OR LGPL-3.0-or-later
set(SPHINX_DOC_BINARY_DIR ${CMAKE_BINARY_DIR}/src)

# replace above variables in the conf.py for sphinx
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in ${CMAKE_CURRENT_BINARY_DIR}/conf.py)

# create symlinks to the rst document
dune_symlink_to_source_files(FILES index.rst duneuropy.rst duneuropy_2d.rst duneuropy_3d.rst)

add_custom_target(sphinx
  COMMAND ${SPHINX_EXECUTABLE} -b html ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/html)
