# File for module specific CMake tests.
find_package(PythonLibs REQUIRED)
if (EXISTS "${CMAKE_SOURCE_DIR}/external/pybind11/include")
  include_directories(${CMAKE_SOURCE_DIR}/external/pybind11/include)
else()
  message(FATAL_ERROR "pybind11 not found in external. did you call git submodule update?")
endif()
include_directories(${PYTHON_INCLUDE_DIR})
