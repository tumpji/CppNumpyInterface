find_package(Boost COMPONENTS unit_test_framework REQUIRED)

set(PYTHON_EXECUTABLE python3.6)

ADD_DEFINITIONS(-O0 -ggdb3 -fno-inline -fno-inline-small-functions)

# creates main for all tests
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

# TUPLE TEST
#ADD_EXECUTABLE(test_tuple test_tuple.cpp)
#ADD_TEST(test_tuple test_tuple)

#TARGET_LINK_LIBRARIES(test_tuple ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}) 

# NDARRAY
#ADD_EXECUTABLE(test_ndarray test_ndarray.cpp)
#ADD_TEST(test_ndarray test_ndarray)
#
#TARGET_LINK_LIBRARIES(test_ndarray ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}) 
#message("CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR}")
#message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
#message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")


ADD_CUSTOM_TARGET(prepare_test_libraryes
    COMMAND "${CMAKE_COMMAND}" -E copy ${CMAKE_BINARY_DIR}/example/libexample.so ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS example
    VERBATIM
    )

ADD_CUSTOM_TARGET(run_test
    COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS prepare_test_libraryes
    VERBATIM
    )

ADD_TEST(NAME PyScript 
    COMMAND "${CMAKE_COMMAND}" --build . --target run_test
    )



