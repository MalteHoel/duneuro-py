add_library(duneuropy SHARED duneuro-py.cc)
target_link_libraries(duneuropy ${PYTHON_LIBRARIES})
set_target_properties(duneuropy PROPERTIES PREFIX "")
