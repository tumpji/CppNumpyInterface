cmake_minimum_required(VERSION 3.10)
FIND_PACKAGE(PythonLibs REQUIRED)

project(CppNumpyInterface)


if (CMAKE_PROJECT_NAME STREQUAL CppNumpyInterface)
    enable_testing()
    add_subdirectory(test)
endif()


if (CMAKE_PROJECT_NAME STREQUAL CppNumpyInterface)
    add_subdirectory(example)
endif()

MESSAGE("Using python version: ${Python3_LIBRARY_DIRS}") 

set(${PROJECT_NAME}_INCLUDE_DIRS 
        ${PROJECT_SOURCE_DIR}/src 
        ${PROJECT_SOURCE_DIR}/src/utils
        ${PYTHON_INCLUDE_DIRS}
        /usr/lib64/python3.6/site-packages/numpy/core/include/numpy
        CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE
        )
set(${PROJECT_NAME}_LINK_LIBRARIES
    ${PYTHON_LIBRARY})


