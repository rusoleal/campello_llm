#pragma once

#include <cstdint>

namespace systems::leal::campello_llm
{

    struct Version
    {
        std::uint32_t major;
        std::uint32_t minor;
        std::uint32_t patch;
    };

    /**
     * @brief The campello_llm library version (matches the CMake project version).
     */
    Version version();

} // namespace systems::leal::campello_llm
