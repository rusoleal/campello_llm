#include <campello_llm/chat_template.hpp>

#include <gtest/gtest.h>

using namespace systems::leal::campello_llm;

TEST(ChatTemplate, DefaultTemplateFormatsMultiTurnConversation)
{
    std::vector<ChatMessage> messages = {
        {"system", "You are helpful."},
        {"user", "Hello"},
    };

    std::string prompt = formatChatPrompt(messages, defaultChatTemplate(), true);

    std::string expected =
        "You are helpful."
        "<|user|>\n"
        "Hello\n"
        "<|assistant|>\n";

    EXPECT_EQ(prompt, expected);
}

TEST(ChatTemplate, DefaultTemplateWithoutGenerationPrompt)
{
    std::vector<ChatMessage> messages = {
        {"user", "Hello"},
    };

    std::string prompt = formatChatPrompt(messages, defaultChatTemplate(), false);

    std::string expected =
        "<|user|>\n"
        "Hello\n";

    EXPECT_EQ(prompt, expected);
}

TEST(ChatTemplate, DefaultTemplateIncludesAssistantHistory)
{
    std::vector<ChatMessage> messages = {
        {"user", "Hello"},
        {"assistant", "Hi there!"},
        {"user", "How are you?"},
    };

    std::string prompt = formatChatPrompt(messages, defaultChatTemplate(), true);

    std::string expected =
        "<|user|>\n"
        "Hello\n"
        "<|assistant|>\n"
        "Hi there!</s>\n"
        "<|user|>\n"
        "How are you?\n"
        "<|assistant|>\n";

    EXPECT_EQ(prompt, expected);
}

TEST(ChatTemplate, CustomSimpleTemplate)
{
    std::string tmpl = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}";
    std::vector<ChatMessage> messages = {
        {"user", "Hi"},
        {"assistant", "Hello"},
    };

    std::string prompt = formatChatPrompt(messages, tmpl, false);
    EXPECT_EQ(prompt, "user: Hi\nassistant: Hello\n");
}

TEST(ChatTemplate, LoopLastCondition)
{
    std::string tmpl = "{% for message in messages %}{{ message.content }}{% if not loop.last %}, {% endif %}{% endfor %}";
    std::vector<ChatMessage> messages = {
        {"user", "a"},
        {"user", "b"},
        {"user", "c"},
    };

    std::string prompt = formatChatPrompt(messages, tmpl, false);
    EXPECT_EQ(prompt, "a, b, c");
}

TEST(ChatTemplate, StringConcatenation)
{
    std::string tmpl = "{% for message in messages %}{{ 'prefix ' + message.content + ' suffix' }}{% endfor %}";
    std::vector<ChatMessage> messages = {
        {"user", "hello"},
    };

    std::string prompt = formatChatPrompt(messages, tmpl, false);
    EXPECT_EQ(prompt, "prefix hello suffix");
}

TEST(ChatTemplate, BracketMemberAccess)
{
    std::string tmpl = "{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% endif %}{% endfor %}";
    std::vector<ChatMessage> messages = {
        {"user", "ok"},
    };

    std::string prompt = formatChatPrompt(messages, tmpl, false);
    EXPECT_EQ(prompt, "ok");
}
