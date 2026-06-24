// Minimal interactive chat CLI: loads a LLaMA-family HuggingFace checkpoint
// directory (config.json, tokenizer.json, tokenizer_config.json,
// model.safetensors -- e.g. a local clone of TinyLlama/TinyLlama-1.1B-Chat-v1.0)
// and streams generated replies to a single-turn chat prompt, on the CPU
// backend. See CLAUDE.md for the "no KV-cache yet" performance caveat -- this
// is the pragmatic minimum-viable interactive example, not the final Phase 4
// design.
#include <cstdlib>
#include <iostream>
#include <string>

#include <campello_llm/model.hpp>
#include <campello_llm/tokenizer_config.hpp>
#include <campello_nn/context.hpp>

using namespace systems::leal::campello_llm;
namespace cnn = systems::leal::campello_nn;

// Hardcoded default for quick local testing against a predownloaded checkpoint
// -- override with `<model_directory>` as argv[1] for anyone else running this.
constexpr const char *kDefaultModelDirectory = "/Users/rubenleal/Documents/GitHub/TinyLlama-1.1B-Chat-v1.0";

int main(int argc, char **argv)
{
    std::string modelDir = argc > 1 ? argv[1] : kDefaultModelDirectory;
    std::int64_t maxSequenceLength = argc > 2 ? std::atoll(argv[2]) : 512;

    std::cout << "Loading model from " << modelDir << " (maxSequenceLength=" << maxSequenceLength << ")...\n";
    auto context = cnn::Context::create({cnn::DeviceType::Cpu});
    std::unique_ptr<Model> model;
    try
    {
        model = Model::load(context, modelDir, maxSequenceLength);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load model: " << e.what() << "\n";
        return 1;
    }

    SpecialTokenStrings specialTokens = loadSpecialTokenStringsFromFile(modelDir + "/tokenizer_config.json");
    std::cout << "Loaded. Type a message and press Enter (Ctrl+D to quit).\n";

    GenerationConfig generationConfig;
    generationConfig.maxTokens = 128;
    generationConfig.temperature = 0.0f; // greedy -- this is a correctness demo, not a creativity demo

    std::string line;
    while (true)
    {
        std::cout << "\n> ";
        if (!std::getline(std::cin, line))
        {
            break;
        }
        if (line.empty())
        {
            continue;
        }

        std::string prompt = formatSingleTurnChatPrompt("user", line, specialTokens.eosToken);
        try
        {
            model->generate(prompt, generationConfig, [](const std::string &chunk) {
                std::cout << chunk;
                std::cout.flush();
            });
        }
        catch (const std::exception &e)
        {
            std::cerr << "\n[generation error: " << e.what() << "]\n";
        }
        std::cout << "\n";
    }

    return 0;
}
