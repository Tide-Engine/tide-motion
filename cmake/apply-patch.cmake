# Applies PATCH to the current FetchContent source once; a repeated populate
# finds it already applied (reverse check) and leaves the tree alone.
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${PATCH}" RESULT_VARIABLE applied OUTPUT_QUIET ERROR_QUIET)
if(applied EQUAL 0)
  return()
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" apply --whitespace=nowarn "${PATCH}" RESULT_VARIABLE code)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "Could not apply ${PATCH}")
endif()
