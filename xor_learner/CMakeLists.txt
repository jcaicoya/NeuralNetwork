project(xor_learner LANGUAGES CXX)

add_executable(${PROJECT_NAME}
        xor_learner.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE neural_net)
