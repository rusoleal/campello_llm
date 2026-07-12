#include <campello_llm/chat_template.hpp>

#include <cctype>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <variant>
#include <vector>

namespace systems::leal::campello_llm
{

    namespace
    {

        // ------------------------------------------------------------------
        // Value model
        // ------------------------------------------------------------------
        struct Value;
        using Array = std::vector<Value>;
        using Object = std::unordered_map<std::string, Value>;

        struct Value
        {
            std::variant<std::string, std::int64_t, bool, Array, Object> data;

            Value() = default;
            Value(std::string s) : data(std::move(s)) {}
            Value(const char *s) : data(std::string(s)) {}
            Value(std::int64_t i) : data(i) {}
            Value(bool b) : data(b) {}
            Value(Array a) : data(std::move(a)) {}
            Value(Object o) : data(std::move(o)) {}

            bool isString() const { return std::holds_alternative<std::string>(data); }
            bool isInt() const { return std::holds_alternative<std::int64_t>(data); }
            bool isBool() const { return std::holds_alternative<bool>(data); }
            bool isArray() const { return std::holds_alternative<Array>(data); }
            bool isObject() const { return std::holds_alternative<Object>(data); }

            const std::string &asString() const { return std::get<std::string>(data); }
            std::int64_t asInt() const { return std::get<std::int64_t>(data); }
            bool asBool() const { return std::get<bool>(data); }
            const Array &asArray() const { return std::get<Array>(data); }
            const Object &asObject() const { return std::get<Object>(data); }

            std::string toString() const
            {
                if (isString())
                    return asString();
                if (isInt())
                    return std::to_string(asInt());
                if (isBool())
                    return asBool() ? "true" : "false";
                return "";
            }

            bool toBool() const
            {
                if (isBool())
                    return asBool();
                if (isInt())
                    return asInt() != 0;
                if (isString())
                    return !asString().empty();
                if (isArray())
                    return !asArray().empty();
                return true;
            }
        };

        // ------------------------------------------------------------------
        // Lexer for control tags
        // ------------------------------------------------------------------
        enum class TokenType
        {
            Text,
            Ident,
            String,
            Number,
            For,
            In,
            EndFor,
            If,
            Elif,
            Else,
            EndIf,
            True,
            False,
            And,
            Or,
            Not,
            Eq,
            Ne,
            Plus,
            Dot,
            LBracket,
            RBracket,
            LParen,
            RParen,
            Eof
        };

        struct Token
        {
            TokenType type;
            std::string text;
            std::size_t pos;
        };

        class Lexer
        {
        public:
            explicit Lexer(std::string_view src) : src_(src), pos_(0) {}

            std::vector<Token> tokenize()
            {
                std::vector<Token> tokens;
                while (pos_ < src_.size())
                {
                    if (peek() == '{' && peek(1) == '%')
                    {
                        tokenizeControl(tokens);
                    }
                    else if (peek() == '{' && peek(1) == '{')
                    {
                        tokenizeExpression(tokens);
                    }
                    else
                    {
                        tokenizeText(tokens);
                    }
                }
                tokens.push_back({TokenType::Eof, "", pos_});
                return tokens;
            }

        private:
            std::string_view src_;
            std::size_t pos_;

            char peek(std::size_t offset = 0) const
            {
                if (pos_ + offset >= src_.size())
                    return '\0';
                return src_[pos_ + offset];
            }

            void advance(std::size_t n = 1) { pos_ += n; }

            void tokenizeText(std::vector<Token> &tokens)
            {
                std::size_t start = pos_;
                while (pos_ < src_.size())
                {
                    if (peek() == '{' && (peek(1) == '%' || peek(1) == '{'))
                        break;
                    advance();
                }
                if (pos_ > start)
                {
                    tokens.push_back({TokenType::Text, std::string(src_.substr(start, pos_ - start)), start});
                }
            }

            void tokenizeControl(std::vector<Token> &tokens)
            {
                advance(2); // {%
                skipWhitespace();
                while (pos_ < src_.size() - 1 && !(peek() == '%' && peek(1) == '}'))
                {
                    skipWhitespace();
                    if (pos_ >= src_.size() - 1 || (peek() == '%' && peek(1) == '}'))
                        break;
                    tokenizeWordOrSymbol(tokens);
                    skipWhitespace();
                }
                if (peek() == '%' && peek(1) == '}')
                    advance(2);
                else
                    throw std::runtime_error("Unterminated control tag at " + std::to_string(pos_));
            }

            void tokenizeExpression(std::vector<Token> &tokens)
            {
                advance(2); // {{
                skipWhitespace();
                while (pos_ < src_.size() - 1 && !(peek() == '}' && peek(1) == '}'))
                {
                    skipWhitespace();
                    if (pos_ >= src_.size() - 1 || (peek() == '}' && peek(1) == '}'))
                        break;
                    tokenizeWordOrSymbol(tokens);
                    skipWhitespace();
                }
                if (peek() == '}' && peek(1) == '}')
                    advance(2);
                else
                    throw std::runtime_error("Unterminated expression at " + std::to_string(pos_));
            }

            void tokenizeWordOrSymbol(std::vector<Token> &tokens)
            {
                if (std::isalpha(peek()) || peek() == '_')
                {
                    std::size_t start = pos_;
                    while (pos_ < src_.size() && (std::isalnum(peek()) || peek() == '_'))
                        advance();
                    std::string word(src_.substr(start, pos_ - start));
                    TokenType type = TokenType::Ident;
                    if (word == "for")
                        type = TokenType::For;
                    else if (word == "in")
                        type = TokenType::In;
                    else if (word == "endfor")
                        type = TokenType::EndFor;
                    else if (word == "if")
                        type = TokenType::If;
                    else if (word == "elif")
                        type = TokenType::Elif;
                    else if (word == "else")
                        type = TokenType::Else;
                    else if (word == "endif")
                        type = TokenType::EndIf;
                    else if (word == "true")
                        type = TokenType::True;
                    else if (word == "false")
                        type = TokenType::False;
                    else if (word == "and")
                        type = TokenType::And;
                    else if (word == "or")
                        type = TokenType::Or;
                    else if (word == "not")
                        type = TokenType::Not;
                    tokens.push_back({type, word, start});
                }
                else if (std::isdigit(peek()))
                {
                    std::size_t start = pos_;
                    while (pos_ < src_.size() && std::isdigit(peek()))
                        advance();
                    tokens.push_back({TokenType::Number, std::string(src_.substr(start, pos_ - start)), start});
                }
                else if (peek() == '"' || peek() == '\'')
                {
                    tokens.push_back(tokenizeString());
                }
                else
                {
                    switch (peek())
                    {
                    case '=':
                        if (peek(1) == '=')
                        {
                            tokens.push_back({TokenType::Eq, "==", pos_});
                            advance(2);
                        }
                        else
                        {
                            throw std::runtime_error("Unexpected '=' at " + std::to_string(pos_));
                        }
                        break;
                    case '!':
                        if (peek(1) == '=')
                        {
                            tokens.push_back({TokenType::Ne, "!=", pos_});
                            advance(2);
                        }
                        else
                        {
                            throw std::runtime_error("Unexpected '!' at " + std::to_string(pos_));
                        }
                        break;
                    case '+':
                        tokens.push_back({TokenType::Plus, "+", pos_});
                        advance();
                        break;
                    case '.':
                        tokens.push_back({TokenType::Dot, ".", pos_});
                        advance();
                        break;
                    case '[':
                        tokens.push_back({TokenType::LBracket, "[", pos_});
                        advance();
                        break;
                    case ']':
                        tokens.push_back({TokenType::RBracket, "]", pos_});
                        advance();
                        break;
                    case '(':
                        tokens.push_back({TokenType::LParen, "(", pos_});
                        advance();
                        break;
                    case ')':
                        tokens.push_back({TokenType::RParen, ")", pos_});
                        advance();
                        break;
                    default:
                        throw std::runtime_error(std::string("Unexpected character '") + peek() + "' at " + std::to_string(pos_));
                    }
                }
            }

            Token tokenizeString()
            {
                char quote = peek();
                std::size_t start = pos_;
                advance();
                std::string value;
                while (pos_ < src_.size() && peek() != quote)
                {
                    if (peek() == '\\' && pos_ + 1 < src_.size())
                    {
                        advance();
                        char c = peek();
                        switch (c)
                        {
                        case 'n':
                            value += '\n';
                            break;
                        case 't':
                            value += '\t';
                            break;
                        case 'r':
                            value += '\r';
                            break;
                        case '\\':
                            value += '\\';
                            break;
                        case '"':
                            value += '"';
                            break;
                        case '\'':
                            value += '\'';
                            break;
                        default:
                            value += c;
                            break;
                        }
                        advance();
                    }
                    else
                    {
                        value += peek();
                        advance();
                    }
                }
                if (peek() != quote)
                    throw std::runtime_error("Unterminated string at " + std::to_string(start));
                advance();
                return {TokenType::String, value, start};
            }

            void skipWhitespace()
            {
                while (pos_ < src_.size() && std::isspace(peek()))
                    advance();
            }
        };

        // ------------------------------------------------------------------
        // AST
        // ------------------------------------------------------------------
        struct Expr
        {
            virtual ~Expr() = default;
            virtual Value evaluate(const Object &vars) const = 0;
        };

        struct Stmt
        {
            virtual ~Stmt() = default;
            virtual void render(std::ostringstream &out, const Object &vars) const = 0;
        };

        using StmtPtr = std::unique_ptr<Stmt>;
        using ExprPtr = std::unique_ptr<Expr>;

        struct TextStmt : Stmt
        {
            std::string text;
            explicit TextStmt(std::string t) : text(std::move(t)) {}
            void render(std::ostringstream &out, const Object &) const override { out << text; }
        };

        struct OutputStmt : Stmt
        {
            ExprPtr expr;
            explicit OutputStmt(ExprPtr e) : expr(std::move(e)) {}
            void render(std::ostringstream &out, const Object &vars) const override
            {
                out << expr->evaluate(vars).toString();
            }
        };

        struct ForStmt : Stmt
        {
            std::string loopVar;
            std::string collectionVar;
            std::vector<StmtPtr> body;
            ForStmt(std::string lv, std::string cv, std::vector<StmtPtr> b)
                : loopVar(std::move(lv)), collectionVar(std::move(cv)), body(std::move(b)) {}

            void render(std::ostringstream &out, const Object &vars) const override
            {
                auto it = vars.find(collectionVar);
                if (it == vars.end() || !it->second.isArray())
                    return;
                const auto &arr = it->second.asArray();
                for (std::size_t i = 0; i < arr.size(); ++i)
                {
                    Object local = vars;
                    local[loopVar] = arr[i];
                    Object loop;
                    loop["index"] = Value(static_cast<std::int64_t>(i + 1));
                    loop["first"] = Value(i == 0);
                    loop["last"] = Value(i + 1 == arr.size());
                    local["loop"] = Value(loop);
                    for (const auto &stmt : body)
                        stmt->render(out, local);
                }
            }
        };

        struct IfStmt : Stmt
        {
            struct Branch
            {
                ExprPtr condition;
                std::vector<StmtPtr> body;
            };
            std::vector<Branch> branches;
            std::vector<StmtPtr> elseBody;

            void render(std::ostringstream &out, const Object &vars) const override
            {
                for (const auto &branch : branches)
                {
                    if (branch.condition->evaluate(vars).toBool())
                    {
                        for (const auto &stmt : branch.body)
                            stmt->render(out, vars);
                        return;
                    }
                }
                for (const auto &stmt : elseBody)
                    stmt->render(out, vars);
            }
        };

        struct StringExpr : Expr
        {
            std::string value;
            explicit StringExpr(std::string v) : value(std::move(v)) {}
            Value evaluate(const Object &) const override { return Value(value); }
        };

        struct IntExpr : Expr
        {
            std::int64_t value;
            explicit IntExpr(std::int64_t v) : value(v) {}
            Value evaluate(const Object &) const override { return Value(value); }
        };

        struct BoolExpr : Expr
        {
            bool value;
            explicit BoolExpr(bool v) : value(v) {}
            Value evaluate(const Object &) const override { return Value(value); }
        };

        struct VarExpr : Expr
        {
            std::string name;
            explicit VarExpr(std::string n) : name(std::move(n)) {}
            Value evaluate(const Object &vars) const override
            {
                auto it = vars.find(name);
                if (it == vars.end())
                    return Value("");
                return it->second;
            }
        };

        struct MemberExpr : Expr
        {
            ExprPtr object;
            std::string member;
            MemberExpr(ExprPtr o, std::string m) : object(std::move(o)), member(std::move(m)) {}
            Value evaluate(const Object &vars) const override
            {
                Value obj = object->evaluate(vars);
                if (obj.isObject())
                {
                    auto it = obj.asObject().find(member);
                    if (it != obj.asObject().end())
                        return it->second;
                }
                return Value("");
            }
        };

        struct IndexExpr : Expr
        {
            ExprPtr object;
            std::string key;
            IndexExpr(ExprPtr o, std::string k) : object(std::move(o)), key(std::move(k)) {}
            Value evaluate(const Object &vars) const override
            {
                Value obj = object->evaluate(vars);
                if (obj.isObject())
                {
                    auto it = obj.asObject().find(key);
                    if (it != obj.asObject().end())
                        return it->second;
                }
                return Value("");
            }
        };

        struct BinOpExpr : Expr
        {
            enum Op { Eq, Ne, And, Or, Plus } op;
            ExprPtr left;
            ExprPtr right;
            BinOpExpr(Op o, ExprPtr l, ExprPtr r) : op(o), left(std::move(l)), right(std::move(r)) {}

            Value evaluate(const Object &vars) const override
            {
                Value l = left->evaluate(vars);
                Value r = right->evaluate(vars);
                switch (op)
                {
                case Eq:
                    if (l.isString() && r.isString())
                        return Value(l.asString() == r.asString());
                    if (l.isInt() && r.isInt())
                        return Value(l.asInt() == r.asInt());
                    if (l.isBool() && r.isBool())
                        return Value(l.asBool() == r.asBool());
                    return Value(false);
                case Ne:
                    if (l.isString() && r.isString())
                        return Value(l.asString() != r.asString());
                    if (l.isInt() && r.isInt())
                        return Value(l.asInt() != r.asInt());
                    if (l.isBool() && r.isBool())
                        return Value(l.asBool() != r.asBool());
                    return Value(true);
                case And:
                    return Value(l.toBool() && r.toBool());
                case Or:
                    return Value(l.toBool() || r.toBool());
                case Plus:
                    return Value(l.toString() + r.toString());
                }
                return Value("");
            }
        };

        struct NotExpr : Expr
        {
            ExprPtr operand;
            explicit NotExpr(ExprPtr o) : operand(std::move(o)) {}
            Value evaluate(const Object &vars) const override
            {
                return Value(!operand->evaluate(vars).toBool());
            }
        };

        // ------------------------------------------------------------------
        // Parser
        // ------------------------------------------------------------------
        class Parser
        {
        public:
            explicit Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)), pos_(0) {}

            std::vector<StmtPtr> parseTemplate()
            {
                std::vector<StmtPtr> stmts;
                while (!check(TokenType::Eof))
                {
                    stmts.push_back(parseStmt());
                }
                return stmts;
            }

        private:
            std::vector<Token> tokens_;
            std::size_t pos_;

            const Token &peek() const { return tokens_[pos_]; }
            const Token &advance()
            {
                const Token &t = tokens_[pos_];
                if (pos_ + 1 < tokens_.size())
                    ++pos_;
                return t;
            }
            bool check(TokenType type) const { return peek().type == type; }
            bool consume(TokenType type)
            {
                if (check(type))
                {
                    advance();
                    return true;
                }
                return false;
            }
            void expect(TokenType type)
            {
                if (!consume(type))
                {
                    throw std::runtime_error("Expected token at position " + std::to_string(peek().pos));
                }
            }

            StmtPtr parseStmt()
            {
                if (check(TokenType::Text))
                {
                    std::string text = advance().text;
                    return std::make_unique<TextStmt>(std::move(text));
                }
                if (check(TokenType::For))
                {
                    return parseFor();
                }
                if (check(TokenType::If))
                {
                    return parseIf();
                }
                // Expression output: the lexer already emitted the expression tokens.
                auto expr = parseExpr();
                return std::make_unique<OutputStmt>(std::move(expr));
            }

            StmtPtr parseFor()
            {
                expect(TokenType::For);
                std::string loopVar = advance().text;
                expect(TokenType::In);
                std::string collection = advance().text;
                std::vector<StmtPtr> body;
                while (!check(TokenType::EndFor) && !check(TokenType::Eof))
                {
                    body.push_back(parseStmt());
                }
                expect(TokenType::EndFor);
                return std::make_unique<ForStmt>(std::move(loopVar), std::move(collection), std::move(body));
            }

            StmtPtr parseIf()
            {
                auto stmt = std::make_unique<IfStmt>();
                expect(TokenType::If);
                stmt->branches.push_back({parseExpr(), parseBranchBody()});
                while (check(TokenType::Elif))
                {
                    advance();
                    stmt->branches.push_back({parseExpr(), parseBranchBody()});
                }
                if (check(TokenType::Else))
                {
                    advance();
                    stmt->elseBody = parseBranchBody();
                }
                expect(TokenType::EndIf);
                return stmt;
            }

            std::vector<StmtPtr> parseBranchBody()
            {
                std::vector<StmtPtr> body;
                while (!check(TokenType::Elif) && !check(TokenType::Else) && !check(TokenType::EndIf) && !check(TokenType::Eof))
                {
                    body.push_back(parseStmt());
                }
                return body;
            }

            ExprPtr parseExpr()
            {
                return parseOr();
            }

            ExprPtr parseOr()
            {
                auto left = parseAnd();
                while (check(TokenType::Or))
                {
                    advance();
                    auto right = parseAnd();
                    left = std::make_unique<BinOpExpr>(BinOpExpr::Or, std::move(left), std::move(right));
                }
                return left;
            }

            ExprPtr parseAnd()
            {
                auto left = parseEquality();
                while (check(TokenType::And))
                {
                    advance();
                    auto right = parseEquality();
                    left = std::make_unique<BinOpExpr>(BinOpExpr::And, std::move(left), std::move(right));
                }
                return left;
            }

            ExprPtr parseEquality()
            {
                auto left = parseConcat();
                while (check(TokenType::Eq) || check(TokenType::Ne))
                {
                    TokenType op = advance().type;
                    auto right = parseConcat();
                    BinOpExpr::Op binOp = (op == TokenType::Eq) ? BinOpExpr::Eq : BinOpExpr::Ne;
                    left = std::make_unique<BinOpExpr>(binOp, std::move(left), std::move(right));
                }
                return left;
            }

            ExprPtr parseConcat()
            {
                auto left = parseUnary();
                while (check(TokenType::Plus))
                {
                    advance();
                    auto right = parseUnary();
                    left = std::make_unique<BinOpExpr>(BinOpExpr::Plus, std::move(left), std::move(right));
                }
                return left;
            }

            ExprPtr parseUnary()
            {
                if (check(TokenType::Not))
                {
                    advance();
                    return std::make_unique<NotExpr>(parseUnary());
                }
                return parsePrimary();
            }

            ExprPtr parsePrimary()
            {
                if (check(TokenType::String))
                {
                    return std::make_unique<StringExpr>(advance().text);
                }
                if (check(TokenType::Number))
                {
                    return std::make_unique<IntExpr>(std::stoll(advance().text));
                }
                if (check(TokenType::True))
                {
                    advance();
                    return std::make_unique<BoolExpr>(true);
                }
                if (check(TokenType::False))
                {
                    advance();
                    return std::make_unique<BoolExpr>(false);
                }
                if (check(TokenType::Ident))
                {
                    std::string name = advance().text;
                    ExprPtr expr = std::make_unique<VarExpr>(name);
                    while (check(TokenType::Dot) || check(TokenType::LBracket))
                    {
                        if (consume(TokenType::Dot))
                        {
                            std::string member = advance().text;
                            expr = std::make_unique<MemberExpr>(std::move(expr), std::move(member));
                        }
                        else if (consume(TokenType::LBracket))
                        {
                            std::string key = advance().text;
                            expect(TokenType::RBracket);
                            expr = std::make_unique<IndexExpr>(std::move(expr), std::move(key));
                        }
                    }
                    return expr;
                }
                if (check(TokenType::LParen))
                {
                    advance();
                    auto expr = parseExpr();
                    expect(TokenType::RParen);
                    return expr;
                }
                throw std::runtime_error("Unexpected token in expression at position " + std::to_string(peek().pos));
            }
        };

        // ------------------------------------------------------------------
        // Rendering
        // ------------------------------------------------------------------
        static std::string renderTemplate(const std::vector<StmtPtr> &stmts, const Object &vars)
        {
            std::ostringstream out;
            for (const auto &stmt : stmts)
            {
                stmt->render(out, vars);
            }
            return out.str();
        }

        static Object buildTopLevelVars(const std::vector<ChatMessage> &messages, bool addGenerationPrompt)
        {
            Object vars;

            Array msgs;
            for (const auto &m : messages)
            {
                Object obj;
                obj["role"] = Value(m.role);
                obj["content"] = Value(m.content);
                msgs.push_back(Value(obj));
            }
            vars["messages"] = Value(msgs);
            vars["add_generation_prompt"] = Value(addGenerationPrompt);

            return vars;
        }

    } // namespace

    std::string formatChatPrompt(const std::vector<ChatMessage> &messages,
                                  const std::string &chatTemplate,
                                  bool addGenerationPrompt)
    {
        Lexer lexer(chatTemplate);
        auto tokens = lexer.tokenize();
        Parser parser(std::move(tokens));
        auto stmts = parser.parseTemplate();
        auto vars = buildTopLevelVars(messages, addGenerationPrompt);
        return renderTemplate(stmts, vars);
    }

    std::string defaultChatTemplate()
    {
        return R"({% for message in messages %}{% if message.role == 'system' %}{{ message.content }}{% elif message.role == 'user' %}<|user|>
{{ message.content }}
{% elif message.role == 'assistant' %}<|assistant|>
{{ message.content }}</s>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>
{% endif %})";
    }

} // namespace systems::leal::campello_llm
