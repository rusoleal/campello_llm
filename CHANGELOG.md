# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-04

### Changed
- Bumped `campello_nn` dependency from `v0.3.0` to `v0.4.0`.

### Fixed
- Ignored generated graph-cache files (`*.cache`) so they are no longer accidentally committed.

## [0.1.0] - 2026-06-24

### Added
- Initial release of `campello_llm`.
- CMake-based build with `FetchContent` for `campello_nn` and GoogleTest.
- Safetensors and GGUF weights-file readers.
- HuggingFace `tokenizer.json` BPE tokenizer.
- Architecture registry with LLaMA and GPT graph wiring.
- `Model::load()` / `Model::generate()` with KV-cache and graph caching.
- Cross-platform CI for macOS, Linux, and Windows.
