add_executable(campello_llm_cli_chat main.cpp)
target_link_libraries(campello_llm_cli_chat campello_llm)
set_target_properties(campello_llm_cli_chat PROPERTIES CXX_STANDARD 20)

if(WIN32)
    # Same reasoning as tests/CMakeLists.txt's copy step: campello_gpu/DirectML.dll
    # must sit next to this executable or it can't even launch on Windows.
    get_target_property(_campello_llm_cli_chat_directml_dll campello_nn CAMPELLO_NN_DIRECTML_DLL)
    if(NOT _campello_llm_cli_chat_directml_dll)
        message(FATAL_ERROR
            "campello_llm: campello_nn's CAMPELLO_NN_DIRECTML_DLL target property is not set — "
            "the fetched campello_nn version (see the GIT_TAG in the top-level CMakeLists.txt) "
            "predates that property's introduction. Bump to a campello_nn version that has it.")
    endif()
    add_custom_command(TARGET campello_llm_cli_chat POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_campello_llm_cli_chat_directml_dll}"
            "$<TARGET_FILE:campello_gpu>"
            "$<TARGET_FILE_DIR:campello_llm_cli_chat>"
    )
endif()
