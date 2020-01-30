FIND_PACKAGE(PythonLibs REQUIRED)
FIND_PACKAGE(PythonInterp REQUIRED)

#FIND_PACKAGE(FindPython3 PACKAGE_VERSION_UNSUITABLE REQUIRED)

#EXECUTE_PROCESS(python3-config --includes
#FIND_PACKAGE(Python3 COMPONENTS NumPy Development)
#set(CMAKE_C_COMPILER_INIT g++)

ADD_DEFINITIONS(-Wno-missing-field-initializers )
#ADD_DEFINITIONS(-O0 -Wall -Wextra -std=c++17 -fPIC)
#ADD_DEFINITIONS(-O3 -ggdb3 -Wall -Wextra -std=c++17 -fno-inline -fno-inline-small-functions -fPIC)

ADD_DEFINITIONS(-O3 -Wall -Wextra -std=c++17 -fPIC)
ADD_LIBRARY(example SHARED main.cpp)

TARGET_INCLUDE_DIRECTORIES(example PRIVATE ${${PROJECT_NAME}_INCLUDE_DIRS}) 
TARGET_LINK_LIBRARIES(example PRIVATE ${${PROJECT_NAME}_LINK_LIBRARIES}) 

#MESSAGE("b: ${${PROJECT_NAME}_INCLUDE_DIRS}")
#=
#MESSAGE("c: ${CppNumpyInterface_INCLUDE_DIRS}")

# include 
#TARGET_INCLUDE_DIRECTORIES(example PRIVATE "${CppNumpyInterface_SOURCE_DIR}/src")
#TARGET_INCLUDE_DIRECTORIES(example PRIVATE "${CppNumpyInterface_SOURCE_DIR}/src/utils")
#TARGET_INCLUDE_DIRECTORIES(example PRIVATE "/usr/lib64/python3.6/site-packages/numpy/core/include/numpy")
#/usr/lib64/python3.6/site-packages/numpy/core/include/numpy
#TARGET_INCLUDE_DIRECTORIES(example PRIVATE ${PYTHON_INCLUDE_DIRS})
# libs
#TARGET_LINK_LIBRARIES(example ${PYTHON_LIBRARY})

#TARGET_COMPILE_DEFINITIONS(example PRIVATE "$")
