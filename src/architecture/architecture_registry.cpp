#include <campello_llm/architecture.hpp>

#include <stdexcept>

namespace systems::leal::campello_llm
{

    ArchitectureGraphResult buildGraphForArchitecture(const std::string &architectureName,
                                                       std::shared_ptr<systems::leal::campello_nn::Context> context,
                                                       const WeightsFile &weights, const ArchitectureConfig &config,
                                                       std::int64_t seqLen)
    {
        if (architectureName == "llama")
        {
            if (!std::holds_alternative<LlamaConfig>(config))
            {
                throw std::runtime_error("campello_llm: architecture 'llama' requires a LlamaConfig");
            }
            return buildLlamaGraph(context, weights, std::get<LlamaConfig>(config), seqLen);
        }
        if (architectureName == "gpt2")
        {
            if (!std::holds_alternative<GptConfig>(config))
            {
                throw std::runtime_error("campello_llm: architecture 'gpt2' requires a GptConfig");
            }
            return buildGptGraph(context, weights, std::get<GptConfig>(config), seqLen);
        }
        throw std::runtime_error("campello_llm: unknown architecture '" + architectureName + "'");
    }

} // namespace systems::leal::campello_llm
