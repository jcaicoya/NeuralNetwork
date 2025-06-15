# Enable testing
enable_testing()

# Fetch GTest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add test executable
add_executable(tests
    test_network.cpp
)

target_include_directories(tests
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

# Link GTest libraries
target_link_libraries(tests
    PRIVATE neural_net
    PRIVATE gtest_main
)

# Add to CTest
include(GoogleTest)
gtest_discover_tests(tests)
