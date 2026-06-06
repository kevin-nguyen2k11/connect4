set(tensorflow_INCLUDE_DIRS /Users/kevinnguyen/miniforge3/envs/my_tf_env/lib/python3.9/site-packages/tensorflow/include/)

mark_as_advanced(tensorflow_INCLUDE_DIRS)

set(tensorflow_LIBRARIES /Users/kevinnguyen/miniforge3/envs/my_tf_env/lib/python3.9/site-packages/tensorflow/libtensorflow_cc.2.dylib)

mark_as_advanced(tensorflow_LIBRARIES)


if(NOT tensorflow_INCLUDE_DIRS)
  message(STATUS "Could NOT find tensorflow/c/c_api.h")
endif()
if(NOT tensorflow_LIBRARIES)
  message(STATUS "Could NOT find tensorflow library")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(tensorflow DEFAULT_MSG tensorflow_INCLUDE_DIRS tensorflow_LIBRARIES)
