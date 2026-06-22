#include <campello_llm/version.hpp>

#include "campello_llm_config.h"

namespace systems::leal::campello_llm
{

    Version version()
    {
        return {
            campello_llm_VERSION_MAJOR,
            campello_llm_VERSION_MINOR,
            campello_llm_VERSION_PATCH,
        };
    }

} // namespace systems::leal::campello_llm
