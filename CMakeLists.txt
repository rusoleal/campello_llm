cmake_minimum_required(VERSION 3.22.1)

project(campello_llm VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(FetchContent)

# Single source of truth for the library version: the `project(VERSION ...)`
# call above sets campello_llm_VERSION_MAJOR/MINOR/PATCH, substituted here into
# a generated header so version.cpp doesn't hardcode a second copy.
configure_file(src/campello_llm_config.h.in campello_llm_config.h)

# ---------------------------------------------------------------------------
# campello_nn — the compute-graph library this project builds on top of.
# campello_nn has no tags yet, so this pins to a known-good commit; move to a
# real tag once campello_nn cuts one. campello_nn itself selects the right
# backend (CPU/MPSGraph/DirectML/...) per-platform internally, so no
# per-platform .cmake files are needed here.
#
# campello_nn's own CMakeLists.txt declares an `option(BUILD_TESTS ...)` with
# the same name as ours — force it off while fetching so a `-DBUILD_TESTS=ON`
# meant for campello_llm's own tests doesn't also build (and run, via ctest)
# campello_nn's universal test suite, whose fixture paths resolve against
# *this* project's source tree and fail for reasons unrelated to campello_llm.
# ---------------------------------------------------------------------------
set(CAMPELLO_LLM_BUILD_TESTS_REQUESTED ${BUILD_TESTS})
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    campello_nn
    GIT_REPOSITORY https://github.com/rusoleal/campello_nn.git
    GIT_TAG 95a8c01b586c854da6c4613a6f6cab6a69be8c28
)
FetchContent_MakeAvailable(campello_nn)

set(BUILD_TESTS ${CAMPELLO_LLM_BUILD_TESTS_REQUESTED} CACHE BOOL "" FORCE)

# ---------------------------------------------------------------------------
# Core sources
# ---------------------------------------------------------------------------
set(CAMPELLO_LLM_CORE_SOURCES
    src/version.cpp
)

add_library(${PROJECT_NAME} STATIC
    ${CAMPELLO_LLM_CORE_SOURCES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_include_directories(${PROJECT_NAME} PRIVATE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
)

target_link_libraries(${PROJECT_NAME} PUBLIC campello_nn::campello_nn)

add_library(campello_llm::campello_llm ALIAS ${PROJECT_NAME})

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
option(BUILD_TESTS "Build unit tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
