cmake_minimum_required(VERSION 3.10)

project(CppNumpyInterface)


if (CMAKE_PROJECT_NAME STREQUAL CppNumpyInterface)
    enable_testing()
    add_subdirectory(test)
endif()


if (CMAKE_PROJECT_NAME STREQUAL CppNumpyInterface)
    add_subdirectory(example)
endif()


