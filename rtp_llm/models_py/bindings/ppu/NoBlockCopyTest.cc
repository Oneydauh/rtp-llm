#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <array>
#include <exception>
#include <thread>

namespace rtp_llm {
namespace {

torch::TensorOptions bytesOn(torch::Device device) {
    return torch::TensorOptions().dtype(torch::kUInt8).device(device);
}

TEST(PpuNoBlockCopyTest, CopiesAllDirectionsAndInteriorViews) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto host        = torch::arange(256, torch::kInt32).to(torch::kUInt8).pin_memory();
    auto device      = torch::zeros({512}, bytesOn(torch::Device(torch::kCUDA, 0)));
    auto device_copy = torch::zeros_like(device);
    auto result      = torch::zeros_like(host);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    MultiCopyParams h2d;
    h2d.multi_src = {host.slice(0, 17, 217)};
    h2d.multi_dst = {device.slice(0, 129, 329)};
    auto owner    = acquireDeviceHostCopyStreams(0);
    execNoBlockCopy(h2d, {owner, DeviceHostCopyDirection::H2D});
    execNoBlockCopy({{device_copy.slice(0, 31, 231)}, h2d.multi_dst});
    execNoBlockCopy({{result.slice(0, 17, 217)}, {device_copy.slice(0, 31, 231)}},
                    {owner, DeviceHostCopyDirection::D2H});
    EXPECT_TRUE(torch::equal(result.slice(0, 17, 217), host.slice(0, 17, 217)));
    EXPECT_EQ(result.slice(0, 0, 17).sum().item<int64_t>(), 0);
    EXPECT_EQ(result.slice(0, 217).sum().item<int64_t>(), 0);
    auto host_copy = torch::empty_like(result);
    execNoBlockCopy({{host_copy}, {result}});
    EXPECT_TRUE(torch::equal(host_copy, result));
}

TEST(PpuNoBlockCopyTest, ValidatesBeforeSubmittingAndHandlesEmptyCopies) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto src = torch::full({16}, 7, bytesOn(torch::Device(torch::kCUDA, 0)));
    auto dst = torch::zeros({16}, torch::kUInt8).pin_memory();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // The first item must not be copied when a later item is invalid.
    EXPECT_THROW(execNoBlockCopy({{dst, dst.slice(0, 0, 15)}, {src, src}}), c10::Error);
    EXPECT_EQ(dst.sum().item<int64_t>(), 0);
    EXPECT_THROW(execNoBlockCopy({{dst}, {src.slice(0, 0, 16, 2)}}), c10::Error);
    EXPECT_THROW(execNoBlockCopy({{dst}, {src}}, {acquireDeviceHostCopyStreams(0), DeviceHostCopyDirection::H2D}),
                 c10::Error);
    EXPECT_NO_THROW(execNoBlockCopy({}));
    EXPECT_NO_THROW(execNoBlockCopy({{dst.slice(0, 0, 0)}, {src.slice(0, 0, 0)}}));
}

TEST(PpuNoBlockCopyTest, ConcurrentLayerStagingKeepsBytesAndOwnerLifetimes) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr int layers = 24;
    constexpr int bytes  = 8192;
    auto          pool   = torch::empty({layers, bytes + 256}, bytesOn(torch::Device(torch::kCUDA, 0)));
    for (int layer = 0; layer < layers; ++layer) {
        pool[layer].fill_(layer + 1);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::array<std::exception_ptr, layers> errors{};
    std::array<torch::Tensor, layers>      results;
    std::vector<std::thread>               workers;
    for (int layer = 0; layer < layers; ++layer) {
        workers.emplace_back([&, layer]() {
            try {
                auto            pinned = torch::empty({bytes}, torch::kUInt8).pin_memory();
                MultiCopyParams params;
                for (int block = 0; block < 4; ++block) {
                    // Match makeValidBlocks: non-owning byte tensors into one
                    // pinned allocation and an interior layer of a GPU pool.
                    params.multi_src.push_back(torch::from_blob(pool[layer].data_ptr<uint8_t>() + 128 + block * 2048,
                                                                {2048},
                                                                bytesOn(torch::Device(torch::kCUDA, 0))));
                    params.multi_dst.push_back(
                        torch::from_blob(pinned.data_ptr<uint8_t>() + block * 2048, {2048}, bytesOn(torch::kCPU)));
                }
                execNoBlockCopy(params);
                results[layer] = pinned;
            } catch (...) { errors[layer] = std::current_exception(); }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    pool = torch::Tensor();
    for (int layer = 0; layer < layers; ++layer) {
        ASSERT_EQ(errors[layer], nullptr) << "layer=" << layer;
        EXPECT_TRUE(torch::equal(results[layer], torch::full({bytes}, layer + 1, torch::kUInt8)));
    }
}

TEST(PpuNoBlockCopyTest, CopiesBeyondFourGiBAllocationOffset) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    // The smoke pool is 91 GiB; layer bases pass 32-bit offsets. Reserve only
    // 4 GiB here and touch a small tail, without filling the entire allocation.
    constexpr int64_t offset = (int64_t{1} << 32) + 256;
    auto              pool   = torch::empty({offset + 4096}, bytesOn(torch::Device(torch::kCUDA, 0)));
    auto              tail   = pool.slice(0, offset, offset + 4096);
    tail.fill_(173);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto pinned = torch::zeros({4096}, torch::kUInt8).pin_memory();
    auto source = torch::from_blob(tail.data_ptr(), {4096}, tail.options());
    execNoBlockCopy({{pinned}, {source}});
    EXPECT_TRUE(torch::equal(pinned, torch::full({4096}, 173, torch::kUInt8)));
}

}  // namespace
}  // namespace rtp_llm
