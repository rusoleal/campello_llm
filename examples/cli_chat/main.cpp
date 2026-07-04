// Minimal interactive chat CLI: loads a LLaMA-family HuggingFace checkpoint
// directory (config.json, tokenizer.json, tokenizer_config.json,
// model.safetensors -- e.g. a local clone of TinyLlama/TinyLlama-1.1B-Chat-v1.0)
// and streams generated replies to a single-turn chat prompt.
//
// Usage:
//   campello_llm_cli_chat [model_directory] [max_sequence_length] [options]
//
// Options:
//   --device=cpu|gpu-generic|mpsgraph
//   --prompt "<text>"   run one generation and exit (benchmark mode)
//   --max-tokens N      cap generated tokens (default: 128)
//   -h, --help          show this help
//
// Supported device types:
//   cpu         - CPU reference backend (default)
//   gpu-generic - campello_gpu-based cross-platform GPU backend
//   mpsgraph    - Apple MPSGraph backend (macOS/iOS only, maps to DeviceType::Gpu)
//
// See CLAUDE.md for the "no KV-cache yet" performance caveat -- this is the
// pragmatic minimum-viable interactive example, not the final Phase 4 design.
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>

#include <campello_llm/model.hpp>
#include <campello_llm/tokenizer_config.hpp>
#include <campello_nn/context.hpp>

using namespace systems::leal::campello_llm;
namespace cnn = systems::leal::campello_nn;

// Hardcoded default for quick local testing against a predownloaded checkpoint
// -- override with `<model_directory>` as argv[1] for anyone else running this.
constexpr const char *kDefaultModelDirectory = "/Users/rubenleal/Documents/GitHub/TinyLlama-1.1B-Chat-v1.0";

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

namespace
{
    struct ParsedArgs
    {
        std::string modelDir = kDefaultModelDirectory;
        std::int64_t maxSequenceLength = 512;
        cnn::DeviceType deviceType = cnn::DeviceType::Cpu;
        std::string prompt;
        std::uint32_t maxTokens = 128;
    };

    void printUsage(const char *programName)
    {
        std::cerr << "Usage: " << programName
                  << " [model_directory] [max_sequence_length] [--device=cpu|gpu-generic|mpsgraph] [--prompt=\"text\"] [--max-tokens=N]\n";
    }

    cnn::DeviceType deviceTypeFromString(const std::string &name)
    {
        static const std::unordered_map<std::string, cnn::DeviceType> kMap = {
            {"cpu", cnn::DeviceType::Cpu},
            {"gpu-generic", cnn::DeviceType::GpuGeneric},
            {"mpsgraph", cnn::DeviceType::Gpu},
        };
        auto it = kMap.find(name);
        if (it == kMap.end())
        {
            throw std::runtime_error("unknown device type: " + name);
        }
        return it->second;
    }

    std::string deviceTypeToString(cnn::DeviceType type)
    {
        switch (type)
        {
        case cnn::DeviceType::Cpu:
            return "cpu";
        case cnn::DeviceType::GpuGeneric:
            return "gpu-generic";
        case cnn::DeviceType::Gpu:
            return "mpsgraph";
        case cnn::DeviceType::Npu:
            return "npu";
        case cnn::DeviceType::Default:
            return "default";
        }
        return "unknown";
    }

    ParsedArgs parseArgs(int argc, char **argv)
    {
        ParsedArgs args;
        std::string deviceName;
        int positionalIndex = 0;

        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "--device" || arg == "-d")
            {
                if (i + 1 >= argc)
                {
                    throw std::runtime_error("missing value for " + arg);
                }
                deviceName = argv[++i];
            }
            else if (arg.rfind("--device=", 0) == 0)
            {
                deviceName = arg.substr(std::string("--device=").size());
            }
            else if (arg == "--prompt")
            {
                if (i + 1 >= argc)
                {
                    throw std::runtime_error("missing value for " + arg);
                }
                args.prompt = argv[++i];
            }
            else if (arg.rfind("--prompt=", 0) == 0)
            {
                args.prompt = arg.substr(std::string("--prompt=").size());
            }
            else if (arg == "--max-tokens")
            {
                if (i + 1 >= argc)
                {
                    throw std::runtime_error("missing value for " + arg);
                }
                args.maxTokens = static_cast<std::uint32_t>(std::atoi(argv[++i]));
            }
            else if (arg.rfind("--max-tokens=", 0) == 0)
            {
                args.maxTokens =
                    static_cast<std::uint32_t>(std::atoi(arg.substr(std::string("--max-tokens=").size()).c_str()));
            }
            else if (arg == "-h" || arg == "--help")
            {
                printUsage(argv[0]);
                std::exit(0);
            }
            else if (positionalIndex == 0)
            {
                args.modelDir = arg;
                ++positionalIndex;
            }
            else if (positionalIndex == 1)
            {
                args.maxSequenceLength = std::atoll(arg.c_str());
                ++positionalIndex;
            }
            else
            {
                throw std::runtime_error("unexpected positional argument: " + arg);
            }
        }

        if (!deviceName.empty())
        {
            args.deviceType = deviceTypeFromString(deviceName);
        }

        return args;
    }

    struct BenchmarkResult
    {
        double loadSeconds = 0.0;
        double generationSeconds = 0.0;
        std::uint32_t generatedTokens = 0;
        std::string generatedText;
    };

    BenchmarkResult runBenchmark(Model &model, const SpecialTokenStrings &specialTokens,
                                  const std::string &userPrompt, std::uint32_t maxTokens)
    {
        BenchmarkResult result;

        GenerationConfig generationConfig;
        generationConfig.maxTokens = maxTokens;
        generationConfig.temperature = 0.0f; // greedy -- deterministic for benchmarking

        std::string prompt = formatSingleTurnChatPrompt("user", userPrompt, specialTokens.eosToken);

        auto start = Clock::now();
        std::uint32_t tokenCount = 0;
        std::string text;
        try
        {
            text = model.generate(prompt, generationConfig, [&tokenCount](const std::string &) {
                // Tokens are streamed as decoded text chunks; for a rough count we rely
                // on the fact that generate() emits exactly one chunk per sampled token
                // in greedy mode. Counting Unicode tokens accurately would require the
                // tokenizer, but this is sufficient for a coarse throughput number.
                ++tokenCount;
            });
        }
        catch (const std::exception &e)
        {
            std::cerr << "\n[generation error: " << e.what() << "]\n";
            throw;
        }
        auto end = Clock::now();

        result.generationSeconds = Seconds(end - start).count();
        result.generatedTokens = tokenCount;
        result.generatedText = text;
        return result;
    }

    void printBenchmarkSummary(const BenchmarkResult &result)
    {
        std::cout << "\n--- Benchmark ---\n";
        std::cout << "Load time:          " << result.loadSeconds << " s\n";
        std::cout << "Generation time:    " << result.generationSeconds << " s\n";
        std::cout << "Generated tokens:   " << result.generatedTokens << "\n";
        if (result.generationSeconds > 0.0)
        {
            std::cout << "Throughput:         "
                      << static_cast<double>(result.generatedTokens) / result.generationSeconds
                      << " tok/s\n";
        }
        std::cout << "-----------------\n";
    }
} // namespace

int main(int argc, char **argv)
{
    ParsedArgs args;
    try
    {
        args = parseArgs(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Argument error: " << e.what() << "\n";
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "Loading model from " << args.modelDir
              << " (maxSequenceLength=" << args.maxSequenceLength
              << ", device=" << deviceTypeToString(args.deviceType) << ")...\n";

    auto loadStart = Clock::now();
    auto context = cnn::Context::create({args.deviceType});
    std::unique_ptr<Model> model;
    try
    {
        model = Model::load(context, args.modelDir, args.maxSequenceLength);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load model: " << e.what() << "\n";
        return 1;
    }
    auto loadEnd = Clock::now();

    SpecialTokenStrings specialTokens = loadSpecialTokenStringsFromFile(args.modelDir + "/tokenizer_config.json");

    if (!args.prompt.empty())
    {
        BenchmarkResult result;
        result.loadSeconds = Seconds(loadEnd - loadStart).count();
        result = runBenchmark(*model, specialTokens, args.prompt, args.maxTokens);
        result.loadSeconds = Seconds(loadEnd - loadStart).count();
        std::cout << result.generatedText << "\n";
        printBenchmarkSummary(result);
        return 0;
    }

    std::cout << "Loaded. Type a message and press Enter (Ctrl+D to quit).\n";

    GenerationConfig generationConfig;
    generationConfig.maxTokens = args.maxTokens;
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
