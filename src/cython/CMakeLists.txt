# CAUTION:
#  - You can specify only one .pyx file for each cython module.
#  - The name of the module and the .pyx file should be identical.
cython_add_module(server server.pyx)
cython_add_module(client client.pyx)
cython_add_module(total total.pyx)

# The library (libros_cython_example.so) is created in upper CMakeLists.txt
#target_link_libraries(server)
#target_link_libraries(client)

# Set the targets to be copied in the python package directry.
# This example only place mul_two_ints and fibonacci modules.
# we can call C version of these and python version of the rest.
set_target_properties(server
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})

#install(TARGETS add_two_ints sub_two_ints
#  ARCHIVE DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
#  LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION})
