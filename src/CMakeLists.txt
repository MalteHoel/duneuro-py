add_library(duneuropy SHARED
  duneuro-py.cc
  ${CMAKE_SOURCE_DIR}/duneuro/py/parameter_tree.cc)
target_link_libraries(duneuropy ${PYTHON_LIBRARIES})
set_target_properties(duneuropy PROPERTIES PREFIX "")
dune_symlink_to_source_files(FILES metadict.py)
dune_symlink_to_source_files(FILES analyticalMEG.py)
