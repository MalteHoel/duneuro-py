# SPDX-FileCopyrightText: Copyright © duneuro-py contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-duneuro-py-exception OR LGPL-3.0-or-later
add_subdirectory("doxygen")
if (${SPHINX_FOUND})
  add_subdirectory("sphinx")
endif()
