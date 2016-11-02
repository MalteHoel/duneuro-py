set(SPHINX_DOC_BINARY_DIR ${CMAKE_BINARY_DIR}/src)

# replace above variables in the conf.py for sphinx
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in ${CMAKE_CURRENT_BINARY_DIR}/conf.py)

# create symlinks to the rst document
dune_symlink_to_source_files(FILES index.rst duneuropy.rst)

add_custom_target(sphinx
  COMMAND ${SPHINX_EXECUTABLE} -b html ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/html)
