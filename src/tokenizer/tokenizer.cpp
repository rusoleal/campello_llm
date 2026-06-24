#include <campello_llm/tokenizer.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

#include "../json/json_value.hpp"

using namespace systems::leal::campello_llm;
using namespace systems::leal::campello_llm::internal;

namespace
{
    constexpr const char *kSpaceMarker = "\xE2\x96\x81"; // "▁" U+2581, UTF-8

    std::uint64_t pairKey(std::int32_t left, std::int32_t right)
    {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
               static_cast<std::uint32_t>(right);
    }

    std::string formatByteFallbackToken(std::uint8_t b)
    {
        static const char *hex = "0123456789ABCDEF";
        std::string s = "<0x";
        s += hex[(b >> 4) & 0xF];
        s += hex[b & 0xF];
        s += '>';
        return s;
    }

    bool tryParseByteFallbackToken(const std::string &s, std::uint8_t &outByte)
    {
        if (s.size() != 6 || s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>')
        {
            return false;
        }
        auto hexVal = [](char c) -> int {
            if (c >= '0' && c <= '9')
                return c - '0';
            if (c >= 'A' && c <= 'F')
                return c - 'A' + 10;
            if (c >= 'a' && c <= 'f')
                return c - 'a' + 10;
            return -1;
        };
        int hi = hexVal(s[3]);
        int lo = hexVal(s[4]);
        if (hi < 0 || lo < 0)
        {
            return false;
        }
        outByte = static_cast<std::uint8_t>((hi << 4) | lo);
        return true;
    }

    std::size_t utf8CharLen(unsigned char leadByte)
    {
        if ((leadByte & 0x80) == 0x00)
            return 1;
        if ((leadByte & 0xE0) == 0xC0)
            return 2;
        if ((leadByte & 0xF0) == 0xE0)
            return 3;
        if ((leadByte & 0xF8) == 0xF0)
            return 4;
        return 1; // invalid lead byte: treat as a single raw byte
    }

    std::vector<std::string> splitUtf8Chars(const std::string &s)
    {
        std::vector<std::string> chars;
        std::size_t i = 0;
        while (i < s.size())
        {
            std::size_t n = utf8CharLen(static_cast<unsigned char>(s[i]));
            n = std::min(n, s.size() - i);
            chars.push_back(s.substr(i, n));
            i += n;
        }
        return chars;
    }

    // Replaces every literal ' ' byte with the 3-byte "▁" marker. Safe at the byte
    // level: UTF-8 continuation/lead bytes for non-ASCII characters are always >=
    // 0x80, so they can never be confused with the ASCII space byte (0x20).
    std::string replaceSpacesWithMarker(const std::string &s)
    {
        std::string out;
        out.reserve(s.size());
        for (char c : s)
        {
            if (c == ' ')
            {
                out += kSpaceMarker;
            }
            else
            {
                out.push_back(c);
            }
        }
        return out;
    }

    bool isValidUtf8(const std::vector<std::uint8_t> &bytes)
    {
        std::size_t i = 0;
        while (i < bytes.size())
        {
            std::size_t n = utf8CharLen(bytes[i]);
            if (bytes[i] < 0x80)
            {
                n = 1;
            }
            else if (n == 1 || i + n > bytes.size())
            {
                return false; // invalid lead byte, or truncated multi-byte sequence
            }
            for (std::size_t j = 1; j < n; ++j)
            {
                if ((bytes[i + j] & 0xC0) != 0x80)
                {
                    return false;
                }
            }
            i += n;
        }
        return true;
    }

    // U+FFFD REPLACEMENT CHARACTER, UTF-8 encoded, repeated once per invalid byte —
    // matches the real ByteFallback decoder's "each failed byte gets its own
    // placeholder" behavior.
    std::string replacementCharacters(std::size_t count)
    {
        std::string out;
        out.reserve(count * 3);
        for (std::size_t i = 0; i < count; ++i)
        {
            out += "\xEF\xBF\xBD";
        }
        return out;
    }

} // namespace

namespace systems::leal::campello_llm
{

    std::optional<std::int32_t> Tokenizer::tokenToId(const std::string &token) const
    {
        auto it = tokenToId_.find(token);
        if (it == tokenToId_.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    const std::string &Tokenizer::idToToken(std::int32_t id) const
    {
        if (id < 0 || static_cast<std::size_t>(id) >= idToToken_.size())
        {
            throw std::out_of_range("campello_llm: invalid token id " + std::to_string(id));
        }
        return idToToken_[static_cast<std::size_t>(id)];
    }

    std::vector<std::int32_t> Tokenizer::bpeTokenizeWord(const std::string &word) const
    {
        std::vector<std::int32_t> symbols;
        bool fuseUnkPending = false;

        for (const std::string &ch : splitUtf8Chars(word))
        {
            auto it = tokenToId_.find(ch);
            if (it != tokenToId_.end())
            {
                symbols.push_back(it->second);
                fuseUnkPending = false;
                continue;
            }

            bool fellBack = false;
            if (byteFallback_)
            {
                std::vector<std::int32_t> byteIds;
                byteIds.reserve(ch.size());
                bool allFound = true;
                for (unsigned char b : ch)
                {
                    auto byteIt = tokenToId_.find(formatByteFallbackToken(b));
                    if (byteIt == tokenToId_.end())
                    {
                        allFound = false;
                        break;
                    }
                    byteIds.push_back(byteIt->second);
                }
                if (allFound)
                {
                    symbols.insert(symbols.end(), byteIds.begin(), byteIds.end());
                    fuseUnkPending = false;
                    fellBack = true;
                }
            }

            if (!fellBack)
            {
                if (unkId_ < 0)
                {
                    throw std::runtime_error(
                        "campello_llm: tokenizer cannot represent a character and has no unk_token");
                }
                if (!fuseUnkPending)
                {
                    symbols.push_back(unkId_);
                }
                fuseUnkPending = true; // only matters if a later iteration also falls through
            }
        }

        // Repeatedly merge the lowest-rank adjacent pair until none remain — logically
        // equivalent to the real implementation's priority-queue approach (each pair
        // string has a unique rank by construction, so there's no tie-breaking
        // subtlety to replicate), just expressed as a simpler full rescan per merge
        // since BPE words here are short (a single chat message, not a whole corpus).
        while (symbols.size() > 1)
        {
            std::size_t bestPos = symbols.size();
            std::int32_t bestRank = 0;
            std::int32_t bestMergedId = 0;
            for (std::size_t i = 0; i + 1 < symbols.size(); ++i)
            {
                auto it = mergeRank_.find(pairKey(symbols[i], symbols[i + 1]));
                if (it == mergeRank_.end())
                {
                    continue;
                }
                if (bestPos == symbols.size() || it->second.first < bestRank)
                {
                    bestPos = i;
                    bestRank = it->second.first;
                    bestMergedId = it->second.second;
                }
            }
            if (bestPos == symbols.size())
            {
                break;
            }
            symbols[bestPos] = bestMergedId;
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(bestPos) + 1);
        }

        return symbols;
    }

    std::vector<std::int32_t> Tokenizer::encode(const std::string &text, bool addSpecialTokens) const
    {
        std::vector<std::int32_t> contentIds;

        // Split on literal occurrences of added-token strings first (e.g. "</s>" is
        // recognized even with addSpecialTokens=false — matches the real tokenizer;
        // only added_tokens themselves are addSpecialTokens-independent, the
        // post-processor's template token below is not).
        std::size_t i = 0;
        std::string pending;
        auto flushPending = [&]() {
            if (pending.empty())
            {
                return;
            }
            std::string normalized = hasNormalizer_ ? (kSpaceMarker + replaceSpacesWithMarker(pending))
                                                      : pending;
            auto ids = bpeTokenizeWord(normalized);
            contentIds.insert(contentIds.end(), ids.begin(), ids.end());
            pending.clear();
        };

        while (i < text.size())
        {
            const AddedToken *matched = nullptr;
            for (const AddedToken &added : addedTokens_)
            {
                if (text.compare(i, added.content.size(), added.content) == 0)
                {
                    matched = &added;
                    break;
                }
            }
            if (matched != nullptr)
            {
                flushPending();
                contentIds.push_back(matched->id);
                i += matched->content.size();
            }
            else
            {
                std::size_t n = utf8CharLen(static_cast<unsigned char>(text[i]));
                n = std::min(n, text.size() - i);
                pending += text.substr(i, n);
                i += n;
            }
        }
        flushPending();

        if (addSpecialTokens && bosId_ >= 0)
        {
            contentIds.insert(contentIds.begin(), bosId_);
        }
        return contentIds;
    }

    std::string Tokenizer::decode(const std::vector<std::int32_t> &ids, bool skipSpecialTokens) const
    {
        std::unordered_set<std::int32_t> specialIds;
        if (skipSpecialTokens)
        {
            for (const AddedToken &added : addedTokens_)
            {
                if (added.special)
                {
                    specialIds.insert(added.id);
                }
            }
        }

        std::vector<std::string> tokens;
        tokens.reserve(ids.size());
        for (std::int32_t id : ids)
        {
            if (skipSpecialTokens && specialIds.count(id) != 0)
            {
                continue;
            }
            tokens.push_back(idToToken(id));
        }

        if (!hasDecoder_)
        {
            std::string out;
            for (const std::string &t : tokens)
            {
                out += t;
            }
            return out;
        }

        // Replace("▁" -> " ")
        for (std::string &t : tokens)
        {
            std::string replaced;
            replaced.reserve(t.size());
            std::size_t pos = 0;
            while (pos < t.size())
            {
                if (t.compare(pos, 3, kSpaceMarker) == 0)
                {
                    replaced += ' ';
                    pos += 3;
                }
                else
                {
                    replaced.push_back(t[pos]);
                    ++pos;
                }
            }
            t = std::move(replaced);
        }

        // ByteFallback: merge consecutive "<0xXX>"-shaped tokens back into UTF-8 text.
        std::vector<std::string> afterByteFallback;
        std::vector<std::uint8_t> pendingBytes;
        auto flushBytes = [&]() {
            if (pendingBytes.empty())
            {
                return;
            }
            if (isValidUtf8(pendingBytes))
            {
                afterByteFallback.emplace_back(pendingBytes.begin(), pendingBytes.end());
            }
            else
            {
                afterByteFallback.push_back(replacementCharacters(pendingBytes.size()));
            }
            pendingBytes.clear();
        };
        for (const std::string &t : tokens)
        {
            std::uint8_t b;
            if (tryParseByteFallbackToken(t, b))
            {
                pendingBytes.push_back(b);
            }
            else
            {
                flushBytes();
                afterByteFallback.push_back(t);
            }
        }
        flushBytes();

        // Fuse
        std::string fused;
        for (const std::string &t : afterByteFallback)
        {
            fused += t;
        }

        // Strip(content=' ', start=1, stop=0)
        if (!fused.empty() && fused.front() == ' ')
        {
            fused.erase(fused.begin());
        }

        return fused;
    }

    std::unique_ptr<Tokenizer> loadTokenizerFromMemory(const void *data, std::size_t size)
    {
        JsonValue root = parseJson(static_cast<const char *>(data), size);
        if (root.type != JsonType::Object)
        {
            throw std::runtime_error("campello_llm: tokenizer.json is not a JSON object");
        }

        const JsonValue *model = root.find("model");
        if (model == nullptr || model->type != JsonType::Object)
        {
            throw std::runtime_error("campello_llm: tokenizer.json has no 'model' object");
        }
        const JsonValue *modelType = model->find("type");
        if (modelType == nullptr || modelType->type != JsonType::String || modelType->stringValue != "BPE")
        {
            throw std::runtime_error("campello_llm: only model.type 'BPE' is supported");
        }
        for (const char *unsupportedField : {"continuing_subword_prefix", "end_of_word_suffix"})
        {
            const JsonValue *v = model->find(unsupportedField);
            if (v != nullptr && v->type == JsonType::String)
            {
                throw std::runtime_error(std::string("campello_llm: unsupported non-null model.") + unsupportedField);
            }
        }
        const JsonValue *dropout = model->find("dropout");
        if (dropout != nullptr && dropout->type == JsonType::Number && dropout->numberValue != 0.0)
        {
            throw std::runtime_error("campello_llm: BPE dropout is not supported (deterministic merges only)");
        }

        const JsonValue *vocab = model->find("vocab");
        if (vocab == nullptr || vocab->type != JsonType::Object)
        {
            throw std::runtime_error("campello_llm: tokenizer.json model has no 'vocab' object");
        }
        const JsonValue *merges = model->find("merges");
        if (merges == nullptr || merges->type != JsonType::Array)
        {
            throw std::runtime_error("campello_llm: tokenizer.json model has no 'merges' array");
        }

        auto tokenizer = std::make_unique<Tokenizer>();

        std::int32_t maxId = -1;
        for (const auto &entry : vocab->objectValue)
        {
            if (entry.second.type != JsonType::Number)
            {
                throw std::runtime_error("campello_llm: vocab entry '" + entry.first + "' is not a number");
            }
            maxId = std::max(maxId, static_cast<std::int32_t>(entry.second.numberValue));
        }
        tokenizer->idToToken_.assign(static_cast<std::size_t>(maxId) + 1, std::string());
        for (const auto &entry : vocab->objectValue)
        {
            auto id = static_cast<std::int32_t>(entry.second.numberValue);
            tokenizer->idToToken_[static_cast<std::size_t>(id)] = entry.first;
            tokenizer->tokenToId_.emplace(entry.first, id);
        }

        for (std::size_t rank = 0; rank < merges->arrayValue.size(); ++rank)
        {
            const JsonValue &mergeEntry = merges->arrayValue[rank];
            std::string left;
            std::string right;
            std::string description; // for error messages only
            if (mergeEntry.type == JsonType::String)
            {
                // Older `tokenizers` versions (e.g. the one that produced TinyLlama's
                // real tokenizer.json) serialize merges as a single "left right"
                // string.
                std::size_t sep = mergeEntry.stringValue.find(' ');
                if (sep == std::string::npos)
                {
                    throw std::runtime_error("campello_llm: malformed merges entry '" + mergeEntry.stringValue + "'");
                }
                left = mergeEntry.stringValue.substr(0, sep);
                right = mergeEntry.stringValue.substr(sep + 1);
                description = mergeEntry.stringValue;
            }
            else if (mergeEntry.type == JsonType::Array && mergeEntry.arrayValue.size() == 2 &&
                     mergeEntry.arrayValue[0].type == JsonType::String && mergeEntry.arrayValue[1].type == JsonType::String)
            {
                // Newer `tokenizers` versions (confirmed against the real library,
                // version 0.22.2) serialize each merge as a 2-element [left, right]
                // array instead -- an equally real, valid shape, not a guess.
                left = mergeEntry.arrayValue[0].stringValue;
                right = mergeEntry.arrayValue[1].stringValue;
                description = left + " " + right;
            }
            else
            {
                throw std::runtime_error(
                    "campello_llm: unsupported merges entry shape (expected a string or a 2-element string array)");
            }
            auto leftIt = tokenizer->tokenToId_.find(left);
            auto rightIt = tokenizer->tokenToId_.find(right);
            auto mergedIt = tokenizer->tokenToId_.find(left + right);
            if (leftIt == tokenizer->tokenToId_.end() || rightIt == tokenizer->tokenToId_.end() ||
                mergedIt == tokenizer->tokenToId_.end())
            {
                throw std::runtime_error("campello_llm: merges entry '" + description +
                                          "' references a token missing from vocab");
            }
            tokenizer->mergeRank_[pairKey(leftIt->second, rightIt->second)] = {
                static_cast<std::int32_t>(rank), mergedIt->second};
        }

        const JsonValue *byteFallback = model->find("byte_fallback");
        tokenizer->byteFallback_ = byteFallback != nullptr && byteFallback->type == JsonType::Bool &&
                                    byteFallback->boolValue;

        const JsonValue *unkToken = model->find("unk_token");
        if (unkToken != nullptr && unkToken->type == JsonType::String)
        {
            auto it = tokenizer->tokenToId_.find(unkToken->stringValue);
            if (it == tokenizer->tokenToId_.end())
            {
                throw std::runtime_error("campello_llm: model.unk_token is missing from vocab");
            }
            tokenizer->unkId_ = it->second;
        }

        const JsonValue *addedTokens = root.find("added_tokens");
        if (addedTokens != nullptr)
        {
            if (addedTokens->type != JsonType::Array)
            {
                throw std::runtime_error("campello_llm: tokenizer.json 'added_tokens' is not an array");
            }
            for (const JsonValue &entry : addedTokens->arrayValue)
            {
                const JsonValue *idVal = entry.find("id");
                const JsonValue *contentVal = entry.find("content");
                const JsonValue *specialVal = entry.find("special");
                if (idVal == nullptr || idVal->type != JsonType::Number || contentVal == nullptr ||
                    contentVal->type != JsonType::String)
                {
                    throw std::runtime_error("campello_llm: malformed added_tokens entry");
                }
                for (const char *flag : {"normalized", "lstrip", "rstrip"})
                {
                    const JsonValue *flagVal = entry.find(flag);
                    if (flagVal != nullptr && flagVal->type == JsonType::Bool && flagVal->boolValue)
                    {
                        throw std::runtime_error(std::string("campello_llm: unsupported added_tokens entry with ") +
                                                  flag + "=true");
                    }
                }

                auto id = static_cast<std::int32_t>(idVal->numberValue);
                bool special = specialVal != nullptr && specialVal->type == JsonType::Bool && specialVal->boolValue;

                if (static_cast<std::size_t>(id) >= tokenizer->idToToken_.size())
                {
                    tokenizer->idToToken_.resize(static_cast<std::size_t>(id) + 1);
                }
                tokenizer->idToToken_[static_cast<std::size_t>(id)] = contentVal->stringValue;
                tokenizer->tokenToId_[contentVal->stringValue] = id;
                tokenizer->addedTokens_.push_back({id, contentVal->stringValue, special});
            }
            std::sort(tokenizer->addedTokens_.begin(), tokenizer->addedTokens_.end(),
                      [](const Tokenizer::AddedToken &a, const Tokenizer::AddedToken &b) {
                          return a.content.size() > b.content.size();
                      });
        }

        const JsonValue *normalizer = root.find("normalizer");
        if (normalizer == nullptr || normalizer->type == JsonType::Null)
        {
            tokenizer->hasNormalizer_ = false;
        }
        else
        {
            const JsonValue *normType = normalizer->find("type");
            const JsonValue *normalizers = normalizer->find("normalizers");
            bool shapeOk = normType != nullptr && normType->type == JsonType::String &&
                           normType->stringValue == "Sequence" && normalizers != nullptr &&
                           normalizers->type == JsonType::Array && normalizers->arrayValue.size() == 2;
            if (shapeOk)
            {
                const JsonValue &prepend = normalizers->arrayValue[0];
                const JsonValue &replace = normalizers->arrayValue[1];
                const JsonValue *prependType = prepend.find("type");
                const JsonValue *prependValue = prepend.find("prepend");
                const JsonValue *replaceType = replace.find("type");
                const JsonValue *replacePattern = replace.find("pattern");
                const JsonValue *replaceContent = replace.find("content");
                shapeOk = prependType != nullptr && prependType->type == JsonType::String &&
                          prependType->stringValue == "Prepend" && prependValue != nullptr &&
                          prependValue->type == JsonType::String && prependValue->stringValue == kSpaceMarker &&
                          replaceType != nullptr && replaceType->type == JsonType::String &&
                          replaceType->stringValue == "Replace" && replacePattern != nullptr &&
                          replacePattern->find("String") != nullptr &&
                          replacePattern->find("String")->stringValue == " " && replaceContent != nullptr &&
                          replaceContent->type == JsonType::String && replaceContent->stringValue == kSpaceMarker;
            }
            if (!shapeOk)
            {
                throw std::runtime_error(
                    "campello_llm: unsupported normalizer shape (only null or "
                    "Sequence[Prepend(\"\xE2\x96\x81\"), Replace(\" \"->\"\xE2\x96\x81\")] is supported)");
            }
            tokenizer->hasNormalizer_ = true;
        }

        const JsonValue *preTokenizer = root.find("pre_tokenizer");
        if (preTokenizer != nullptr && preTokenizer->type != JsonType::Null)
        {
            throw std::runtime_error("campello_llm: unsupported non-null pre_tokenizer (only null is supported)");
        }

        const JsonValue *decoder = root.find("decoder");
        if (decoder == nullptr || decoder->type == JsonType::Null)
        {
            tokenizer->hasDecoder_ = false;
        }
        else
        {
            const JsonValue *decType = decoder->find("type");
            const JsonValue *decoders = decoder->find("decoders");
            bool shapeOk = decType != nullptr && decType->type == JsonType::String &&
                           decType->stringValue == "Sequence" && decoders != nullptr &&
                           decoders->type == JsonType::Array && decoders->arrayValue.size() == 4;
            if (shapeOk)
            {
                const JsonValue &replace = decoders->arrayValue[0];
                const JsonValue &byteFallbackDec = decoders->arrayValue[1];
                const JsonValue &fuse = decoders->arrayValue[2];
                const JsonValue &strip = decoders->arrayValue[3];

                auto typeIs = [](const JsonValue &v, const char *expected) {
                    const JsonValue *t = v.find("type");
                    return t != nullptr && t->type == JsonType::String && t->stringValue == expected;
                };
                shapeOk = typeIs(replace, "Replace") && typeIs(byteFallbackDec, "ByteFallback") &&
                          typeIs(fuse, "Fuse") && typeIs(strip, "Strip");
                if (shapeOk)
                {
                    const JsonValue *stripContent = strip.find("content");
                    const JsonValue *stripStart = strip.find("start");
                    const JsonValue *stripStop = strip.find("stop");
                    shapeOk = stripContent != nullptr && stripContent->type == JsonType::String &&
                              stripContent->stringValue == " " && stripStart != nullptr &&
                              stripStart->type == JsonType::Number && stripStart->numberValue == 1.0 &&
                              stripStop != nullptr && stripStop->type == JsonType::Number &&
                              stripStop->numberValue == 0.0;
                }
            }
            if (!shapeOk)
            {
                throw std::runtime_error(
                    "campello_llm: unsupported decoder shape (only null or "
                    "Sequence[Replace, ByteFallback, Fuse, Strip(\" \",1,0)] is supported)");
            }
            tokenizer->hasDecoder_ = true;
        }

        const JsonValue *postProcessor = root.find("post_processor");
        if (postProcessor == nullptr || postProcessor->type == JsonType::Null)
        {
            tokenizer->bosId_ = -1;
        }
        else
        {
            const JsonValue *ppType = postProcessor->find("type");
            if (ppType == nullptr || ppType->type != JsonType::String || ppType->stringValue != "TemplateProcessing")
            {
                throw std::runtime_error("campello_llm: unsupported post_processor.type (only 'TemplateProcessing' is supported)");
            }
            const JsonValue *single = postProcessor->find("single");
            if (single == nullptr || single->type != JsonType::Array)
            {
                throw std::runtime_error("campello_llm: post_processor has no 'single' template array");
            }
            tokenizer->bosId_ = -1;
            for (const JsonValue &step : single->arrayValue)
            {
                const JsonValue *specialToken = step.find("SpecialToken");
                if (specialToken == nullptr)
                {
                    continue; // a "Sequence" step: where the content tokens go, nothing to record
                }
                const JsonValue *tokenText = specialToken->find("id");
                if (tokenText == nullptr || tokenText->type != JsonType::String)
                {
                    throw std::runtime_error("campello_llm: malformed post_processor SpecialToken step");
                }
                auto it = tokenizer->tokenToId_.find(tokenText->stringValue);
                if (it == tokenizer->tokenToId_.end())
                {
                    throw std::runtime_error("campello_llm: post_processor references a token missing from vocab");
                }
                tokenizer->bosId_ = it->second;
                break; // only the first auto-added special token is supported (see tokenizer.hpp)
            }
        }

        return tokenizer;
    }

    std::unique_ptr<Tokenizer> loadTokenizerFromFile(const std::string &path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream)
        {
            throw std::runtime_error("campello_llm: cannot open '" + path + "'");
        }
        std::streamsize size = stream.tellg();
        if (size < 0)
        {
            throw std::runtime_error("campello_llm: cannot determine size of '" + path + "'");
        }
        stream.seekg(0, std::ios::beg);

        std::vector<char> buffer(static_cast<std::size_t>(size));
        if (size > 0 && !stream.read(buffer.data(), size))
        {
            throw std::runtime_error("campello_llm: failed to read '" + path + "'");
        }

        return loadTokenizerFromMemory(buffer.data(), buffer.size());
    }

    std::string formatSingleTurnChatPrompt(const std::string &role, const std::string &content,
                                            const std::string &eosToken)
    {
        return "<|" + role + "|>\n" + content + eosToken + "\n<|assistant|>\n";
    }

} // namespace systems::leal::campello_llm
