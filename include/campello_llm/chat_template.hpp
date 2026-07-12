#pragma once

#include <string>
#include <vector>

namespace systems::leal::campello_llm
{

    /**
     * @brief A single message in a multi-turn conversation.
     */
    struct ChatMessage
    {
        std::string role;
        std::string content;
    };

    /**
     * @brief Render a chat prompt from a Jinja2-like template string.
     *
     * Supported subset:
     *   - Text and `{{ expression }}` output.
     *   - `{% for message in messages %} ... {% endfor %}` with `loop.last`,
     *     `loop.first`, and `loop.index`.
     *   - `{% if expr %} ... {% elif expr %} ... {% else %} ... {% endif %}`.
     *   - Expressions: string literals, identifiers, member access (`message.role`
     *     and `message['role']`), `==`, `!=`, `and`, `or`, `not`, `+` for string
     *     concatenation.
     *
     * Variables available at the top level:
     *   - `messages`: the vector of ChatMessage.
     *   - `add_generation_prompt`: the value of the @p addGenerationPrompt parameter.
     *
     * @throws std::runtime_error on parse or evaluation errors.
     */
    std::string formatChatPrompt(const std::vector<ChatMessage> &messages,
                                  const std::string &chatTemplate,
                                  bool addGenerationPrompt = true);

    /**
     * @brief A default TinyLlama/Zephyr-style chat template.
     */
    std::string defaultChatTemplate();

} // namespace systems::leal::campello_llm
