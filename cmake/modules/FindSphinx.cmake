# Find Sphinx - the python documentation tool
#
# You may set the following variables to modify the
# behaviour of this module:
#
# :ref:`SPHINX_ROOT`
#    the path to look for sphinx with the highest priority
#
# The following variables are set by this module:
#
# :code:`SPHINX_FOUND`
#    whether Sphinx was found
#
# :code:`SPHINX_EXECUTABLE`
#    the path to the sphinx-build executable
#
# TODO export version.
#
# .. cmake_variable:: SPHINX_ROOT
#
#   You may set this variable to have :ref:`FindSphinx` look
#   for the :code:`sphinx-build` executable in the given path
#   before inspecting system paths.
#
# note: this module was taken from the dune-python repository
# Copyright (c) 2015, Dominic Kempf
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
# * Neither the name of the Universität Heidelberg nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

find_program(SPHINX_EXECUTABLE
             NAMES sphinx-build
             PATHS ${SPHINX_ROOT}
             NO_DEFAULT_PATH)

find_program(SPHINX_EXECUTABLE
             NAMES sphinx-build)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  "Sphinx"
  DEFAULT_MSG
  SPHINX_EXECUTABLE
)
