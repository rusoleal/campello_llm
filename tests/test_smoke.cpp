#include <gtest/gtest.h>

#include <campello_llm/version.hpp>
#include <campello_nn/context.hpp>

using namespace systems::leal::campello_llm;
using namespace systems::leal::campello_nn;

TEST(Smoke, VersionIsSet)
{
    auto v = version();
    EXPECT_EQ(v.major, 0u);
    EXPECT_EQ(v.minor, 2u);
    EXPECT_EQ(v.patch, 0u);
}

// Proves the FetchContent'd campello_nn dependency actually links and runs —
// the only thing this layer can rely on before Phase 1+ exist.
TEST(Smoke, CampelloNnContextCreatesOnCpu)
{
    auto context = Context::create({DeviceType::Cpu});
    ASSERT_NE(context, nullptr);
}
