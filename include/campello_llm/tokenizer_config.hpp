#pragma once

#include <string>

namespace systems::leal::campello_llm
{

    /**
     * @brief The handful of `tokenizer_config.json` fields campello_llm needs:
     * BOS/EOS/UNK/PAD as literal token text (empty if the field is absent). These
     * aren't derivable from `tokenizer.json` alone — `Tokenizer::bosId()` only knows
     * about a post-processor's template token, and has no notion of EOS/PAD at all —
     * so this is a deliberately separate, small reader rather than something folded
     * into `Tokenizer`.
     *
     * Look up the corresponding id via `Tokenizer::tokenToId()`.
     */
    struct SpecialTokenStrings
    {
        std::string bosToken;
        std::string eosToken;
        std::string unkToken;
        std::string padToken;
    };

    /**
     * @brief Parses a `tokenizer_config.json` file already loaded into memory.
     */
    SpecialTokenStrings loadSpecialTokenStringsFromMemory(const void *data, std::size_t size);

    /**
     * @brief Reads `path` into memory and calls `loadSpecialTokenStringsFromMemory()`.
     */
    SpecialTokenStrings loadSpecialTokenStringsFromFile(const std::string &path);

} // namespace systems::leal::campello_llm
