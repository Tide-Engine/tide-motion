# Applies PATCHES (a ;-separated list, in order) to the pinned FetchContent source.
# The tracked tree is first restored to the pinned revision, so a repeated or
# changed populate always ends with exactly these patches applied once.
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" checkout -- . RESULT_VARIABLE code)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "Could not restore the pinned source before patching")
endif()
foreach(patch IN LISTS PATCHES)
  execute_process(COMMAND "${GIT_EXECUTABLE}" apply --whitespace=nowarn "${patch}" RESULT_VARIABLE code)
  if(NOT code EQUAL 0)
    message(FATAL_ERROR "Could not apply ${patch}")
  endif()
endforeach()
