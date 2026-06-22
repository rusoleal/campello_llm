#include "json_value.hpp"

#include <stdexcept>

namespace systems::leal::campello_llm::internal
{
    namespace
    {
        class Parser
        {
        public:
            Parser(const char *data, std::size_t size) : data_(data), size_(size) {}

            JsonValue parse()
            {
                skipWhitespace();
                JsonValue v = parseValue();
                skipWhitespace();
                if (pos_ != size_)
                {
                    throw std::runtime_error("campello_llm: trailing data after JSON value");
                }
                return v;
            }

        private:
            const char *data_;
            std::size_t size_;
            std::size_t pos_ = 0;

            char peek() const
            {
                if (pos_ >= size_)
                {
                    throw std::runtime_error("campello_llm: unexpected end of JSON input");
                }
                return data_[pos_];
            }

            char advance()
            {
                char c = peek();
                ++pos_;
                return c;
            }

            void expect(char c)
            {
                if (advance() != c)
                {
                    throw std::runtime_error(std::string("campello_llm: expected '") + c + "' in JSON input");
                }
            }

            void skipWhitespace()
            {
                while (pos_ < size_)
                {
                    char c = data_[pos_];
                    if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                    {
                        ++pos_;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            bool consumeLiteral(const char *literal, std::size_t len)
            {
                if (pos_ + len > size_)
                {
                    return false;
                }
                for (std::size_t i = 0; i < len; ++i)
                {
                    if (data_[pos_ + i] != literal[i])
                    {
                        return false;
                    }
                }
                pos_ += len;
                return true;
            }

            JsonValue parseValue()
            {
                skipWhitespace();
                char c = peek();
                if (c == '{')
                {
                    return parseObject();
                }
                if (c == '[')
                {
                    return parseArray();
                }
                if (c == '"')
                {
                    JsonValue v;
                    v.type = JsonType::String;
                    v.stringValue = parseString();
                    return v;
                }
                if (consumeLiteral("true", 4))
                {
                    JsonValue v;
                    v.type = JsonType::Bool;
                    v.boolValue = true;
                    return v;
                }
                if (consumeLiteral("false", 5))
                {
                    JsonValue v;
                    v.type = JsonType::Bool;
                    v.boolValue = false;
                    return v;
                }
                if (consumeLiteral("null", 4))
                {
                    JsonValue v;
                    v.type = JsonType::Null;
                    return v;
                }
                if (c == '-' || (c >= '0' && c <= '9'))
                {
                    return parseNumber();
                }
                throw std::runtime_error("campello_llm: unexpected character in JSON input");
            }

            JsonValue parseObject()
            {
                expect('{');
                JsonValue v;
                v.type = JsonType::Object;
                skipWhitespace();
                if (pos_ < size_ && peek() == '}')
                {
                    ++pos_;
                    return v;
                }
                while (true)
                {
                    skipWhitespace();
                    std::string key = parseString();
                    skipWhitespace();
                    expect(':');
                    JsonValue value = parseValue();
                    v.objectValue.emplace_back(std::move(key), std::move(value));
                    skipWhitespace();
                    char c = advance();
                    if (c == ',')
                    {
                        continue;
                    }
                    if (c == '}')
                    {
                        break;
                    }
                    throw std::runtime_error("campello_llm: expected ',' or '}' in JSON object");
                }
                return v;
            }

            JsonValue parseArray()
            {
                expect('[');
                JsonValue v;
                v.type = JsonType::Array;
                skipWhitespace();
                if (pos_ < size_ && peek() == ']')
                {
                    ++pos_;
                    return v;
                }
                while (true)
                {
                    v.arrayValue.push_back(parseValue());
                    skipWhitespace();
                    char c = advance();
                    if (c == ',')
                    {
                        continue;
                    }
                    if (c == ']')
                    {
                        break;
                    }
                    throw std::runtime_error("campello_llm: expected ',' or ']' in JSON array");
                }
                return v;
            }

            static void appendUtf8(std::string &out, unsigned int codepoint)
            {
                if (codepoint <= 0x7F)
                {
                    out.push_back(static_cast<char>(codepoint));
                }
                else if (codepoint <= 0x7FF)
                {
                    out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
                    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
                }
                else if (codepoint <= 0xFFFF)
                {
                    out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
                    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
                }
                else
                {
                    out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
                    out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
                }
            }

            unsigned int parseHex4()
            {
                unsigned int value = 0;
                for (int i = 0; i < 4; ++i)
                {
                    char c = advance();
                    value <<= 4;
                    if (c >= '0' && c <= '9')
                    {
                        value |= static_cast<unsigned int>(c - '0');
                    }
                    else if (c >= 'a' && c <= 'f')
                    {
                        value |= static_cast<unsigned int>(c - 'a' + 10);
                    }
                    else if (c >= 'A' && c <= 'F')
                    {
                        value |= static_cast<unsigned int>(c - 'A' + 10);
                    }
                    else
                    {
                        throw std::runtime_error("campello_llm: invalid \\u escape in JSON string");
                    }
                }
                return value;
            }

            std::string parseString()
            {
                expect('"');
                std::string out;
                while (true)
                {
                    char c = advance();
                    if (c == '"')
                    {
                        break;
                    }
                    if (c == '\\')
                    {
                        char esc = advance();
                        switch (esc)
                        {
                        case '"':
                            out.push_back('"');
                            break;
                        case '\\':
                            out.push_back('\\');
                            break;
                        case '/':
                            out.push_back('/');
                            break;
                        case 'b':
                            out.push_back('\b');
                            break;
                        case 'f':
                            out.push_back('\f');
                            break;
                        case 'n':
                            out.push_back('\n');
                            break;
                        case 'r':
                            out.push_back('\r');
                            break;
                        case 't':
                            out.push_back('\t');
                            break;
                        case 'u':
                        {
                            unsigned int codepoint = parseHex4();
                            // Surrogate pair (UTF-16 high/low) -> single codepoint.
                            if (codepoint >= 0xD800 && codepoint <= 0xDBFF)
                            {
                                if (advance() != '\\' || advance() != 'u')
                                {
                                    throw std::runtime_error("campello_llm: unpaired UTF-16 surrogate in JSON string");
                                }
                                unsigned int low = parseHex4();
                                if (low < 0xDC00 || low > 0xDFFF)
                                {
                                    throw std::runtime_error("campello_llm: invalid UTF-16 low surrogate in JSON string");
                                }
                                codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
                            }
                            appendUtf8(out, codepoint);
                            break;
                        }
                        default:
                            throw std::runtime_error("campello_llm: invalid escape sequence in JSON string");
                        }
                    }
                    else
                    {
                        out.push_back(c);
                    }
                }
                return out;
            }

            JsonValue parseNumber()
            {
                std::size_t start = pos_;
                if (pos_ < size_ && data_[pos_] == '-')
                {
                    ++pos_;
                }
                while (pos_ < size_ && data_[pos_] >= '0' && data_[pos_] <= '9')
                {
                    ++pos_;
                }
                if (pos_ < size_ && data_[pos_] == '.')
                {
                    ++pos_;
                    while (pos_ < size_ && data_[pos_] >= '0' && data_[pos_] <= '9')
                    {
                        ++pos_;
                    }
                }
                if (pos_ < size_ && (data_[pos_] == 'e' || data_[pos_] == 'E'))
                {
                    ++pos_;
                    if (pos_ < size_ && (data_[pos_] == '+' || data_[pos_] == '-'))
                    {
                        ++pos_;
                    }
                    while (pos_ < size_ && data_[pos_] >= '0' && data_[pos_] <= '9')
                    {
                        ++pos_;
                    }
                }
                std::string text(data_ + start, pos_ - start);
                JsonValue v;
                v.type = JsonType::Number;
                v.numberValue = std::stod(text);
                return v;
            }
        };

    } // namespace

    const JsonValue *JsonValue::find(const std::string &key) const
    {
        if (type != JsonType::Object)
        {
            return nullptr;
        }
        for (const auto &entry : objectValue)
        {
            if (entry.first == key)
            {
                return &entry.second;
            }
        }
        return nullptr;
    }

    JsonValue parseJson(const char *data, std::size_t size)
    {
        Parser parser(data, size);
        return parser.parse();
    }

} // namespace systems::leal::campello_llm::internal
