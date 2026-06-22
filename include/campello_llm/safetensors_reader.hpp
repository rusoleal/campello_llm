#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <campello_llm/weights_file.hpp>

namespace systems::leal::campello_llm
{

    /**
     * @brief A parsed `.safetensors` file: an 8-byte little-endian header length, a
     * JSON header (tensor name -> {dtype, shape, data_offsets}, plus an optional
     * `__metadata__` string map), followed by the raw tensor bytes.
     *
     * Owns one buffer holding every tensor's bytes (the header is parsed and
     * discarded, not retained); `TensorInfo::data` in `tensors()` are zero-copy
     * pointers into that buffer.
     */
    class SafetensorsFile : public WeightsFile
    {
    public:
        const std::vector<TensorInfo> &tensors() const override;

        const TensorInfo *find(const std::string &name) const override;

        /**
         * @brief The optional `__metadata__` string map (e.g. `format`), empty if the
         * file didn't have one.
         */
        const std::unordered_map<std::string, std::string> &metadata() const;

    private:
        friend std::unique_ptr<SafetensorsFile> loadSafetensorsFromMemory(const void *data, std::size_t size);

        std::vector<std::uint8_t> buffer_;
        std::vector<TensorInfo> tensors_;
        std::unordered_map<std::string, std::size_t> nameToIndex_;
        std::unordered_map<std::string, std::string> metadata_;
    };

    /**
     * @brief Parses a `.safetensors` file already loaded into memory.
     *
     * Copies `data`/`size` into the returned `SafetensorsFile`'s own buffer (callers
     * are free to discard `data` afterwards) — see Android `AAssetManager` rationale
     * in `CLAUDE.md`.
     */
    std::unique_ptr<SafetensorsFile> loadSafetensorsFromMemory(const void *data, std::size_t size);

    /**
     * @brief Reads `path` into memory and calls `loadSafetensorsFromMemory()`.
     */
    std::unique_ptr<SafetensorsFile> loadSafetensorsFromFile(const std::string &path);

} // namespace systems::leal::campello_llm
