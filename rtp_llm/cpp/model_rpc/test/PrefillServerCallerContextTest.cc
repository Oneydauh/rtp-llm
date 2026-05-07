#include "rtp_llm/cpp/model_rpc/PrefillServerCallerContext.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(PrefillServerCallerContextTest, PreserveFirstReadResponseSnapshot) {
    PrefillServerCallerContext context("127.0.0.1:1234", "test_unique_key");

    GenerateOutputsPB first_response;
    first_response.mutable_error_info()->set_error_code(ErrorCodePB::UNKNOWN_ERROR);
    first_response.mutable_error_info()->set_error_message("first chunk error");

    GenerateOutputsPB second_response;

    context.handleReadChunkLocked(first_response);
    context.handleReadChunkLocked(second_response);

    EXPECT_TRUE(context.response_received_);
    EXPECT_TRUE(context.response().has_error_info());
    EXPECT_EQ(context.response().error_info().error_code(), ErrorCodePB::UNKNOWN_ERROR);
    EXPECT_EQ(context.response().error_info().error_message(), "first chunk error");
}

}  // namespace
}  // namespace rtp_llm
