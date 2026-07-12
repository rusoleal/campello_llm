cmake_minimum_required(VERSION 3.22.1)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(campello_llm_tests
    test_smoke.cpp
    test_safetensors_reader.cpp
    test_gguf_reader.cpp
    test_gguf_dequantize.cpp
    test_gguf_model_loader.cpp
    test_tokenizer.cpp
    test_llama_architecture.cpp
    test_gpt_architecture.cpp
    test_architecture_registry.cpp
    test_llama_config_reader.cpp
    test_chat_template.cpp
    test_model.cpp
)
target_include_directories(campello_llm_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_definitions(campello_llm_tests PRIVATE
    CAMPELLO_LLM_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(campello_llm_tests
    campello_llm
    GTest::gtest_main
)
set_target_properties(campello_llm_tests PROPERTIES
    CXX_STANDARD 20
)

include(GoogleTest)
gtest_discover_tests(campello_llm_tests)
