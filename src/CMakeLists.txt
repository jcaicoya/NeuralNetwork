add_library(neural_net
        network.cpp
)

# Include only what's needed: root include/ already added at top level
# This is not strictly required now, but safe for clarity
target_include_directories(neural_net PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)
