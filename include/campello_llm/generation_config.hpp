#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace systems::leal::campello_llm
{

    /**
     * @brief Sampling/generation parameters for `Model::generate()`.
     */
    struct GenerationConfig
    {
        std::uint32_t maxTokens = 256;
        // <= 0 selects deterministic greedy argmax (no RNG involved at all) rather
        // than dividing by an ~0 temperature; > 0 selects temperature/top-k/top-p
        // sampling.
        float temperature = 1.0f;
        float topP = 1.0f;
        std::uint32_t topK = 0; // 0 = disabled
        // Generation stops as soon as the text generated so far ends with any of
        // these (checked after every new token, alongside maxTokens/EOS).
        std::vector<std::string> stopSequences;
    };

} // namespace systems::leal::campello_llm
