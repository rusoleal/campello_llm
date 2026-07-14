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

if(WIN32)
    # Windows has no rpath/install_name equivalent — campello_gpu (built SHARED
    # there, pulled in transitively via campello_nn) and DirectML.dll (always
    # linked into campello_nn's static lib on Windows, see campello_nn's
    # cmake/windows.cmake) must be physically copied next to this executable,
    # or it fails to even launch (STATUS_DLL_NOT_FOUND / exit code 0xc0000135)
    # when gtest_discover_tests()'s post-build step tries to run it. campello_gpu
    # is a real CMake target, resolvable directly; DirectML.dll isn't (linked via
    # a raw file path, not an IMPORTED target), so it's read back off a target
    # property campello_nn exposes for exactly this reason.
    get_target_property(_campello_llm_directml_dll campello_nn CAMPELLO_NN_DIRECTML_DLL)
    if(NOT _campello_llm_directml_dll)
        message(FATAL_ERROR
            "campello_llm: campello_nn's CAMPELLO_NN_DIRECTML_DLL target property is not set — "
            "the fetched campello_nn version (see the GIT_TAG in the top-level CMakeLists.txt) "
            "predates that property's introduction. Bump to a campello_nn version that has it.")
    endif()
    add_custom_command(TARGET campello_llm_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_campello_llm_directml_dll}"
            "$<TARGET_FILE:campello_gpu>"
            "$<TARGET_FILE_DIR:campello_llm_tests>"
    )
endif()

include(GoogleTest)
# DISCOVERY_TIMEOUT above the 5s default: gtest_discover_tests() runs this
# executable once (--gtest_list_tests) to enumerate tests, and on Windows CI
# runners a freshly-built .exe/.dll pair can get held up by antivirus/Defender
# scanning on first execution, occasionally exceeding 5s (seen in practice on
# campello_nn's equivalent step: "Process terminated due to timeout").
gtest_discover_tests(campello_llm_tests DISCOVERY_TIMEOUT 60)
