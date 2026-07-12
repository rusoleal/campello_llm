#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace systems::leal::campello_llm
{

    class GgufFile; // Forward declaration for the GGUF-backed tokenizer factory.

    /**
     * @brief Internal discriminator for the tokenizer flavour loaded.
     */
    enum class TokenizerModel
    {
        LlamaSentencePiece, ///< SentencePiece-style BPE (LLaMA/TinyLlama), uses ▁ normalizer/decoder.
        Gpt2ByteLevel,      ///< GPT-2 byte-level BPE (Llama 3), uses bytes-to-unicode mapping.
    };

    /**
     * @brief Fills @p out with the standard GPT-2 bytes-to-unicode mapping.
     * @internal Used by the GGUF tokenizer loader.
     */
    void buildGpt2ByteToUnicode(std::array<char32_t, 256> &out);

    /**
     * @brief A HuggingFace "fast" tokenizer (`tokenizer.json`) or a GGUF embedded
     * tokenizer (`tokenizer.ggml.*` metadata), scoped to the SentencePiece-style
     * byte-fallback BPE shape LLaMA/TinyLlama-family models ship (the model chosen in
     * `TODO.md`'s Open Questions). Specifically requires:
     *   - `model.type == "BPE"` with `byte_fallback` support
     *   - `normalizer`: `null`, or a `Sequence` of exactly `[Prepend, Replace(" "->"▁")]`
     *   - `pre_tokenizer`: `null` (the whole normalized text is one BPE "word")
     *   - `decoder`: `null`, or a `Sequence` of exactly
     *     `[Replace("▁"->" "), ByteFallback, Fuse, Strip]`
     *   - `post_processor`: `null`, or `TemplateProcessing`'s `single` template
     *
     * Other tokenizer.json shapes (e.g. GPT-2/Llama-3's `ByteLevel` pre_tokenizer,
     * which uses a completely different initial-symbol algorithm) throw rather than
     * silently mistokenizing — see `CLAUDE.md` for exactly which fields were verified
     * against the real `tokenizers` Rust source before being trusted.
     */
    class Tokenizer
    {
    public:
        /**
         * @brief Encodes `text` to token ids.
         *
         * Literal occurrences of any added-token string (e.g. `</s>`) are recognized
         * and emitted as that token's id directly, regardless of `addSpecialTokens` —
         * matching the real tokenizer's behavior (`addSpecialTokens` only controls
         * whether the post-processor's template-defined tokens, e.g. a leading BOS,
         * are additionally applied).
         */
        std::vector<std::int32_t> encode(const std::string &text, bool addSpecialTokens = true) const;

        /**
         * @brief Decodes `ids` back to text. `skipSpecialTokens` drops ids marked
         * `"special": true` in `tokenizer.json`'s `added_tokens` (e.g. BOS/EOS) before
         * reconstructing text — matching the real tokenizer's default.
         */
        std::string decode(const std::vector<std::int32_t> &ids, bool skipSpecialTokens = true) const;

        /**
         * @brief The id the post-processor's `single` template inserts (commonly BOS),
         * or `-1` if there is no post-processor / it has no such token.
         */
        std::int32_t bosId() const { return bosId_; }

        /**
         * @brief `model.unk_token`'s id, or `-1` if the model has none.
         */
        std::int32_t unkId() const { return unkId_; }

        std::size_t vocabSize() const { return idToToken_.size(); }

        std::optional<std::int32_t> tokenToId(const std::string &token) const;

        /**
         * @brief Returns the literal vocab/added-token text for `id`.
         * @throws std::out_of_range if `id` isn't a valid token id.
         */
        const std::string &idToToken(std::int32_t id) const;

    private:
        friend std::unique_ptr<Tokenizer> loadTokenizerFromMemory(const void *data, std::size_t size);
        friend std::unique_ptr<Tokenizer> loadTokenizerFromGgufFile(const GgufFile &file);

        struct AddedToken
        {
            std::int32_t id;
            std::string content;
            bool special;
        };

        std::vector<std::string> idToToken_;
        std::unordered_map<std::string, std::int32_t> tokenToId_;
        // (leftId << 32 | rightId) -> (rank, mergedId), built once from `model.merges`.
        std::unordered_map<std::uint64_t, std::pair<std::int32_t, std::int32_t>> mergeRank_;
        std::vector<AddedToken> addedTokens_; // ordered longest-content-first for matching
        bool byteFallback_ = false;
        bool hasNormalizer_ = false; // Sequence[Prepend("▁"), Replace(" "->"▁")]
        bool hasDecoder_ = false;    // Sequence[Replace("▁"->" "), ByteFallback, Fuse, Strip(" ",1,0)]
        std::int32_t unkId_ = -1;
        std::int32_t bosId_ = -1;
        TokenizerModel model_ = TokenizerModel::LlamaSentencePiece;
        // GPT-2 byte-level BPE state. byteToUnicode_[b] is the Unicode char used to
        // represent byte b in the BPE vocab; unicodeToByte_ is the inverse map.
        std::array<char32_t, 256> byteToUnicode_{};
        std::unordered_map<char32_t, std::uint8_t> unicodeToByte_;

        std::vector<std::int32_t> bpeTokenizeWord(const std::string &word) const;
    };

    /**
     * @brief Parses a `tokenizer.json` file already loaded into memory.
     * @throws std::runtime_error if the file doesn't match the shape documented on
     * `Tokenizer`.
     */
    std::unique_ptr<Tokenizer> loadTokenizerFromMemory(const void *data, std::size_t size);

    /**
     * @brief Reads `path` into memory and calls `loadTokenizerFromMemory()`.
     */
    std::unique_ptr<Tokenizer> loadTokenizerFromFile(const std::string &path);

    /**
     * @brief Builds a Tokenizer from GGUF embedded tokenizer metadata
     * (`tokenizer.ggml.tokens`, `tokenizer.ggml.merges`, token ids, etc.).
     *
     * Only the SentencePiece-style BPE shape used by LLaMA-family GGUF models is
     * supported (same symbol table, byte-fallback, and normalizer/decoder semantics as
     * `loadTokenizerFromMemory()`). Throws for unsupported tokenizer.ggml.model values
     * or missing required fields.
     */
    std::unique_ptr<Tokenizer> loadTokenizerFromGgufFile(const GgufFile &file);

    /**
     * @brief Renders the literal `"<|{role}|>\n{content}{eosToken}\n<|assistant|>\n"`
     * single-turn prompt shape used by TinyLlama-Chat and other Zephyr-style chat
     * models — **not** a Jinja2 `chat_template` interpreter (see `TODO.md` Phase 2's
     * "basic chat template handling" scope). Verified against the real chat_template
     * rendered with Jinja2 using HuggingFace's default Environment settings
     * (`trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True`) — see
     * `CLAUDE.md`.
     */
    std::string formatSingleTurnChatPrompt(const std::string &role, const std::string &content,
                                            const std::string &eosToken);

} // namespace systems::leal::campello_llm
