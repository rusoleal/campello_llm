#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace systems::leal::campello_llm::internal
{

    /**
     * @brief Minimal hand-rolled JSON parser, scoped to reading a safetensors header.
     *
     * Safetensors headers are JSON written by Python's `json` module (objects, string
     * keys, string/integer-array values) — a small, fixed shape that doesn't justify a
     * full JSON dependency, same "hand-roll the minimal reader for a known, stable
     * format" call campello_nn's `proto_reader.hpp` made for ONNX's protobuf. Still a
     * real, generic recursive-descent JSON parser (objects/arrays/strings/numbers/
     * bools/null, with string escapes including `\uXXXX`) rather than a
     * safetensors-specific shortcut, since nothing about JSON syntax itself is
     * safetensors-specific.
     */
    enum class JsonType
    {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object,
    };

    class JsonValue
    {
    public:
        JsonType type = JsonType::Null;
        bool boolValue = false;
        double numberValue = 0.0;
        std::string stringValue;
        std::vector<JsonValue> arrayValue;
        std::vector<std::pair<std::string, JsonValue>> objectValue;

        /**
         * @brief Looks up `key` in `objectValue` (only valid when `type == Object`).
         * @return The value, or `nullptr` if `type != Object` or `key` isn't present.
         */
        const JsonValue *find(const std::string &key) const;
    };

    /**
     * @brief Parses `size` bytes of UTF-8 JSON text starting at `data`.
     * @throws std::runtime_error on any malformed input (every read is bounds-checked).
     */
    JsonValue parseJson(const char *data, std::size_t size);

} // namespace systems::leal::campello_llm::internal
