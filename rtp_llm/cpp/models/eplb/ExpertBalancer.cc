#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include <thread>
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

using namespace std;

namespace rtp_llm {

void EplbPlanBuffers::init(size_t    log_exp_num,
                           size_t    phy_exp_num,
                           size_t    hidden_size,
                           size_t    moe_size,
                           size_t    ep_size,
                           DataType  dtype,
                           QuantAlgo quant_algo,
                           bool      allocate_weight_buffers) {
    auto gpu_i32 = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
    auto cpu_i32 = torch::kInt32;

    layer_id_buf          = torch::zeros({1}, gpu_i32);
    logic_expert_cnt      = torch::zeros({(int64_t)log_exp_num}, gpu_i32);
    logic_expert_cnt_host = torch::zeros({(int64_t)log_exp_num}, cpu_i32).pin_memory();
    log2phy               = torch::zeros({(int64_t)log_exp_num, (int64_t)(phy_exp_num - log_exp_num + 1)}, gpu_i32);
    log2phy_host = torch::zeros({(int64_t)log_exp_num, (int64_t)(phy_exp_num - log_exp_num + 1)}, cpu_i32).pin_memory();
    phy2log      = torch::zeros({(int64_t)phy_exp_num}, gpu_i32);

    size_t expert_per_ep = phy_exp_num / ep_size;

    // NewLoader stages its postprocessed layer on the current rank and returns
    // CPU tensors for an in-place Python install.  It never uses these legacy
    // pointer-swap buffers, whose size is one complete local MoE layer.
    is_quantized = quant_algo.isFp8();
    if (!allocate_weight_buffers) {
        return;
    }

    // note: only support fp8 per token quant
    if (is_quantized) {
        int  group_size = quant_algo.getGroupSize();
        auto gpu_fp8    = torch::TensorOptions(torch::kFloat8_e4m3fn).device(torch::kCUDA);
        auto gpu_fp32   = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);

        moe_weight_1 = torch::zeros({(int64_t)expert_per_ep, (int64_t)(moe_size * 2), (int64_t)hidden_size}, gpu_fp8);
        moe_scale_1  = torch::zeros(
            {(int64_t)expert_per_ep, (int64_t)(moe_size * 2 / group_size), (int64_t)(hidden_size / group_size)},
            gpu_fp32);
        moe_weight_2 = torch::zeros({(int64_t)expert_per_ep, (int64_t)hidden_size, (int64_t)moe_size}, gpu_fp8);
        moe_scale_2  = torch::zeros(
            {(int64_t)expert_per_ep, (int64_t)(hidden_size / group_size), (int64_t)(moe_size / group_size)}, gpu_fp32);
    } else {
        auto gpu_dtype = torch::TensorOptions(dataTypeToTorchType(dtype)).device(torch::kCUDA);
        moe_weight_1 = torch::zeros({(int64_t)expert_per_ep, (int64_t)(moe_size * 2), (int64_t)hidden_size}, gpu_dtype);
        moe_weight_2 = torch::zeros({(int64_t)expert_per_ep, (int64_t)hidden_size, (int64_t)moe_size}, gpu_dtype);
    }
}

void BalanceStatsBuffers::init(int layer_num, int log_exp_num, int ep_size) {
    // host tensors
    log_stats = torch::zeros({layer_num, log_exp_num}, torch::kInt32);
    gpu_loads = torch::zeros({layer_num, ep_size}, torch::kInt32);

    // gpu tensors
    log_stats_gpu = torch::zeros({layer_num, log_exp_num}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    gpu_loads_gpu = torch::zeros({layer_num, ep_size}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
}

void BalanceStatsBuffers::reset() {
    log_stats.zero_();
    gpu_loads.zero_();
    log_stats_gpu.zero_();
    gpu_loads_gpu.zero_();
}

void LoadFlags::init() {
    // [terminal rank count, failed rank count].  Keeping the two counters
    // separate lets every rank wait for all background loaders to finish
    // before deciding whether the plan can be committed.
    flag_gpu  = torch::zeros({2}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    flag_sync = torch::zeros({2}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    flag_host = torch::zeros({2}, torch::kInt32).pin_memory();
}

void LoadFlags::setLoading() {
    flag_gpu.zero_();
}

void LoadFlags::setReady() {
    flag_gpu.zero_();
    flag_gpu.narrow(0, 0, 1).fill_(1);
}

void LoadFlags::setFailed() {
    flag_gpu.fill_(1);
}

LoadFlags::SyncStatus LoadFlags::sync(size_t world_size) {
    if (world_size > 1) {
        flag_sync = execAllReduce({flag_gpu, ReduceOp::Sum, false, ParallelMode::DP_AND_TP, flag_sync}).buffer;
    } else {
        flag_sync.copy_(flag_gpu);
    }

    flag_host.copy_(flag_sync);
    const int* values         = flag_host.data_ptr<int>();
    const int  terminal_ranks = values[0];
    const int  failed_ranks   = values[1];
    if (terminal_ranks != static_cast<int>(world_size)) {
        return SyncStatus::LOADING;
    }
    return failed_ranks == 0 ? SyncStatus::READY : SyncStatus::FAILED;
}

void EplbController::init(const EPLBConfig& eplb_control_data, const EPLBConfig& eplb_config) {
    this->eplb_control_data = eplb_control_data;

    control_step = eplb_config.eplb_control_step;
    RTP_LLM_LOG_INFO("EPLB control step: %d", control_step);

    auto eplb_control_data_list  = eplb_control_data.toList();
    eplb_control_data_buf_host   = torch::zeros({(int64_t)eplb_control_data_list.size()}, torch::kInt32).pin_memory();
    eplb_control_data_buf_device = torch::zeros({(int64_t)eplb_control_data_list.size()},
                                                torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
}

void EplbController::setData(const EPLBConfig& updated_control_data) {
    // lock mutex
    lock_guard<mutex> lock(eplb_control_mutex);
    eplb_control_data = updated_control_data;
}

bool EplbController::stepAndCheckSyncStep() {
    // check if current step is sync step
    cur_step++;
    if (cur_step >= control_step) {
        cur_step = 0;
        return true;
    }
    return false;
}

EPLBConfig EplbController::getAndSyncData(size_t world_size) {
    // copy control data to host buffer
    EPLBConfig cur_data;
    {
        lock_guard<mutex> lock(eplb_control_mutex);
        cur_data = eplb_control_data;
    }
    auto eplb_control_data_list     = cur_data.toList();
    int* eplb_control_data_host_ptr = eplb_control_data_buf_host.data_ptr<int>();
    for (size_t i = 0; i < eplb_control_data_list.size(); ++i) {
        eplb_control_data_host_ptr[i] = eplb_control_data_list[i];
    }

    // copy to device
    eplb_control_data_buf_device.copy_(eplb_control_data_buf_host, /*non_blocking=*/true);

    if (world_size > 1) {
        execBroadcast({{eplb_control_data_buf_device}, 0, ParallelMode::DP_AND_TP});
    }

    // copy to host
    eplb_control_data_buf_host.copy_(eplb_control_data_buf_device);

    // convert to EPLBConfig
    auto eplb_control_data = EPLBConfig::fromList(eplb_control_data_host_ptr);

    return eplb_control_data;
}

ExpertBalancer::ExpertBalancer(size_t                       log_exp_num,
                               size_t                       phy_exp_num,
                               size_t                       num_layers,
                               size_t                       moe_size,
                               size_t                       hidden_size,
                               size_t                       ep_rank,
                               size_t                       ep_size,
                               size_t                       world_size,
                               py::object                   py_eplb,
                               DataType                     dtype,
                               QuantAlgo                    quant_algo,
                               kmonitor::MetricsReporterPtr metrics_reporter,
                               const EPLBConfig&            eplb_config):
    num_logic_experts_(log_exp_num),
    num_physic_experts_(phy_exp_num),
    ep_rank_(ep_rank),
    ep_size_(ep_size),
    world_size_(world_size),
    metrics_reporter_(metrics_reporter),
    eplb_python_wrapper_(py_eplb) {
    cout << "ExpertBalancer constructed with " << log_exp_num << " logical experts" << endl;
    printf("DEBUG: ExpertBalancer constructor called for linker debug\n");
    eplb_control_data_ = eplb_config;

    // init memory
    stats_.init(num_layers, log_exp_num, ep_size_);
    const bool python_weight_install = eplb_python_wrapper_.supportsPythonWeightInstall();
    eplb_plan_buffers_.init(
        log_exp_num, phy_exp_num, hidden_size, moe_size, ep_size, dtype, quant_algo, !python_weight_install);
    if (python_weight_install) {
        RTP_LLM_LOG_INFO("NewLoader EPLB skips legacy full-layer GPU staging buffers");
    }
    eplb_plan_tensors_.init(log_exp_num, phy_exp_num);
    load_flags_.init();
    load_flags_.setLoading();
    eplb_controller_.init(eplb_control_data_, eplb_config);

    test_mode_ = eplb_config.eplb_test_mode;

    balance_layer_per_step_ = eplb_config.eplb_balance_layer_per_step;

    resetPlan(true);
}

ExpertBalancer::~ExpertBalancer() {
    joinPlanWeightLoad();
}

void ExpertBalancer::stepForward(ModelBase& model, RtpLLMExecutorMetricsCollector& executor_collector) {
    syncController();

    if (eplb_control_data_.checkEplbMode(eplb_control_data_.eplb_mode, EplbMode::NONE)) {
        return;
    }

    OverallExpertStats& stats = model.overall_expert_stats_;

    // report stats
    reportStats(stats);

    // eplb plan
    excuteEplbPlan(stats, model);
}

bool ExpertBalancer::updateEplbConfig(const EPLBConfig& config) {
    eplb_controller_.setData(config);
    return true;
}

void ExpertBalancer::syncController() {
    // sync control data
    if (eplb_controller_.stepAndCheckSyncStep()) {
        auto eplb_control_data = eplb_controller_.getAndSyncData(world_size_);
        if (eplb_control_data.eplb_mode != eplb_control_data_.eplb_mode
            || eplb_control_data.eplb_update_time != eplb_control_data_.eplb_update_time) {
            eplb_control_data_ = eplb_control_data;
            RTP_LLM_LOG_INFO("EPLB config changed to %s", eplb_control_data_.to_string().c_str());
        }
    }
}

void ExpertBalancer::reportStats(OverallExpertStats& stats) {
    if (metrics_reporter_
        && eplb_control_data_.checkEplbMode(eplb_control_data_.eplb_mode, EplbMode::STATS, EplbMode::ALL)) {
        int layer_num = stats.layer_num;
        executor_collector_.gpu_loads.resize(layer_num);
        executor_collector_.ep_rank = ep_rank_;

        auto gpu_loads_tensor = stats.stats_buf.gpu_loads_buf.cpu();
        int* gpu_loads        = gpu_loads_tensor.data_ptr<int>();

        for (int i = 0; i < layer_num; ++i) {
            executor_collector_.gpu_loads[i] = gpu_loads[i * ep_size_ + ep_rank_];
        }

        metrics_reporter_->report<RtpLLmEplbMetrics, RtpLLmEplbMetricsCollector>(nullptr, &executor_collector_);
    }
}

void ExpertBalancer::setPlanStatus(EplbPlanStatus status) {
    lock_guard<mutex> lock(eplb_plan_status_mutex_);
    eplb_plan_status_ = status;
}

EplbPlanStatus ExpertBalancer::getPlanStatus() const {
    lock_guard<mutex> lock(eplb_plan_status_mutex_);
    auto              status = eplb_plan_status_;
    return status;
}

void ExpertBalancer::excuteEplbPlan(OverallExpertStats& stats, ModelBase& model) {
    if (eplb_control_data_.checkEplbMode(eplb_control_data_.eplb_mode, EplbMode::EPLB, EplbMode::ALL)) {
        EplbPlanStatus status = getPlanStatus();
        switch (status) {
            case EplbPlanStatus::INIT:
                update_cnt_++;
                updateStats(stats);
                if (update_cnt_ >= eplb_control_data_.eplb_update_time) {
                    setPlanStatus(EplbPlanStatus::PREPARING);
                }
                break;
            case EplbPlanStatus::PREPARING: {
                createPlan();
                setPlanStatus(EplbPlanStatus::LOADING);
                startPlanWeightLoad();
                break;
            }
            case EplbPlanStatus::LOADING:
            case EplbPlanStatus::LOADED:
            case EplbPlanStatus::FAILED: {
                const auto sync_status = syncPlanWeightsLoadStatus(status);
                if (sync_status == LoadFlags::SyncStatus::LOADING) {
                    break;
                }
                joinPlanWeightLoad();
                if (sync_status == LoadFlags::SyncStatus::FAILED) {
                    rollbackPlan();
                    break;
                }
                bool install_ok = true;
                try {
                    applyPlanWeights(model);
                } catch (const std::exception& e) {
                    setLoadError(string("failed to install loaded EPLB plan: ") + e.what());
                    install_ok = false;
                } catch (...) {
                    setLoadError("unknown exception while installing loaded EPLB plan");
                    install_ok = false;
                }
                // Plan installation changes routing and executor aggregation.
                // Keep every EP rank on the same generation before allowing
                // the next model forward to enter a cross-rank collective.
                // Failed ranks must enter the same collective as successful
                // ranks. Otherwise a local exception strands peers in the
                // install barrier and leaves them on a different placement.
                const auto install_sync_status = syncPlanWeightsLoadStatus(
                    install_ok ? EplbPlanStatus::LOADED : EplbPlanStatus::FAILED);
                if (install_sync_status == LoadFlags::SyncStatus::FAILED) {
                    if (!eplb_python_wrapper_.supportsPythonWeightInstall()) {
                        throw std::runtime_error("Legacy EPLB weight installation failed; cannot resume inference");
                    }
                    rollbackPlan();
                    break;
                }
                RTP_LLM_CHECK_WITH_INFO(install_sync_status == LoadFlags::SyncStatus::READY,
                                        "EPLB install barrier did not observe every rank as ready");
                if (eplb_python_wrapper_.supportsPythonWeightInstall()) {
                    eplb_python_wrapper_.commitBalancePlan(eplb_plan_tensors_.layer_id);
                }

                load_flags_.setLoading();
                balance_layer_cnt_++;

                if (balance_layer_cnt_ >= balance_layer_per_step_) {
                    update_cnt_        = 0;
                    balance_layer_cnt_ = 0;
                } else {
                    update_cnt_ = eplb_control_data_.eplb_update_time;  // quick update
                }

                setPlanStatus(EplbPlanStatus::INIT);
                resetPlan(true);
                break;
            }
        }
    }
}

void ExpertBalancer::resetPlan(bool force_clean) {
    stats_.reset();
    if (force_clean) {
        eplb_plan_tensors_.moe_weight_1 = torch::Tensor();
        eplb_plan_tensors_.moe_weight_2 = torch::Tensor();
        eplb_plan_tensors_.moe_scale_1  = torch::Tensor();
        eplb_plan_tensors_.moe_scale_2  = torch::Tensor();
    }
}

void ExpertBalancer::copyFromTensor(const torch::Tensor& src, torch::Tensor& dst) {
    dst.copy_(src, /*non_blocking=*/true);
}

void ExpertBalancer::copyToTensor(const torch::Tensor& src, torch::Tensor& dst) {
    dst.copy_(src, /*non_blocking=*/true);
}

void ExpertBalancer::createPlan() {
    // pre run
    if (world_size_ > 1) {
        execAllReduce({stats_.log_stats_gpu, ReduceOp::Sum, false, ParallelMode::DP_AND_TP});
        execAllReduce({stats_.gpu_loads_gpu, ReduceOp::Sum, false, ParallelMode::DP_AND_TP});
    }

    // copy stats gpu tensor to host tensor [implicit sync]
    stats_.log_stats.copy_(stats_.log_stats_gpu);
    stats_.gpu_loads.copy_(stats_.gpu_loads_gpu);

    if (ep_rank_ == 0) {
        eplb_python_wrapper_.createBalancePlan(stats_.log_stats, stats_.gpu_loads, eplb_plan_tensors_);

        // copy tensor(host) to gpu tensor
        // note: it's ok to use async copy, since the tensor host ptr will not be released
        copyFromTensor(eplb_plan_tensors_.layer_id_buf, eplb_plan_buffers_.layer_id_buf);
        copyFromTensor(eplb_plan_tensors_.logic_expert_cnt, eplb_plan_buffers_.logic_expert_cnt);
        copyFromTensor(eplb_plan_tensors_.log2phy, eplb_plan_buffers_.log2phy);
        copyFromTensor(eplb_plan_tensors_.phy2log, eplb_plan_buffers_.phy2log);
    }

    if (world_size_ > 1) {
        execBroadcast({{eplb_plan_buffers_.layer_id_buf,
                        eplb_plan_buffers_.logic_expert_cnt,
                        eplb_plan_buffers_.log2phy,
                        eplb_plan_buffers_.phy2log},
                       0,
                       ParallelMode::DP_AND_TP});
    }

    // copy plan gpu tensor to host tensor [implicit sync]
    copyToTensor(eplb_plan_buffers_.layer_id_buf, eplb_plan_tensors_.layer_id_buf);
    copyToTensor(eplb_plan_buffers_.logic_expert_cnt, eplb_plan_tensors_.logic_expert_cnt);
    copyToTensor(eplb_plan_buffers_.log2phy, eplb_plan_tensors_.log2phy);
    copyToTensor(eplb_plan_buffers_.phy2log, eplb_plan_tensors_.phy2log);
}

void ExpertBalancer::processPlanWeights() {
    eplb_plan_buffers_.layer_id = eplb_plan_tensors_.layer_id;
    if (eplb_plan_buffers_.is_quantized) {
        copyFromTensor(eplb_plan_tensors_.moe_weight_1, eplb_plan_buffers_.moe_weight_1);
        copyFromTensor(eplb_plan_tensors_.moe_scale_1, eplb_plan_buffers_.moe_scale_1);
        copyFromTensor(eplb_plan_tensors_.moe_weight_2, eplb_plan_buffers_.moe_weight_2);
        copyFromTensor(eplb_plan_tensors_.moe_scale_2, eplb_plan_buffers_.moe_scale_2);
    } else {
        copyFromTensor(eplb_plan_tensors_.moe_weight_1, eplb_plan_buffers_.moe_weight_1);
        copyFromTensor(eplb_plan_tensors_.moe_weight_2, eplb_plan_buffers_.moe_weight_2);
    }

    executor_collector_.update_weights_qps = true;
    executor_collector_.update_layer_id    = eplb_plan_buffers_.layer_id;
}

void ExpertBalancer::loadPlanWeights() {
    int64_t start_time_ms = autil::TimeUtility::currentTimeInMilliSeconds();

    eplb_python_wrapper_.loadBalanceWeight(ep_rank_, ep_size_, eplb_plan_tensors_);

    executor_collector_.update_weights_latency_ms = autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms;
}

void ExpertBalancer::startPlanWeightLoad() {
    RTP_LLM_CHECK_WITH_INFO(!load_thread_.joinable(), "previous EPLB loading thread was not joined");
    load_flags_.setLoading();
    setLoadError("");
    load_thread_ = thread([this]() {
        try {
            loadPlanWeights();
            setPlanStatus(EplbPlanStatus::LOADED);
        } catch (const std::exception& e) {
            setLoadError(e.what());
            setPlanStatus(EplbPlanStatus::FAILED);
        } catch (...) {
            setLoadError("unknown exception while loading EPLB weights");
            setPlanStatus(EplbPlanStatus::FAILED);
        }
    });
    if (test_mode_) {
        joinPlanWeightLoad();
    }
}

void ExpertBalancer::joinPlanWeightLoad() {
    if (load_thread_.joinable()) {
        // The loading worker acquires the Python GIL.  Test-mode calls and
        // teardown may arrive from a Python-owned thread, so do not wait while
        // retaining the GIL.
        if (PyGILState_Check()) {
            py::gil_scoped_release release;
            load_thread_.join();
        } else {
            load_thread_.join();
        }
    }
}

void ExpertBalancer::rollbackPlan() {
    const int layer_id = eplb_plan_tensors_.layer_id_buf.defined() ? eplb_plan_tensors_.layer_id_buf.item<int>() : -1;
    try {
        if (layer_id >= 0) {
            eplb_python_wrapper_.rollbackBalancePlan(layer_id);
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("Failed to rollback NewLoader EPLB controller state for layer %d: %s", layer_id, e.what());
        // Continuing after a failed rollback would expose mixed weights or
        // placement. The executor must treat this as a fatal step failure.
        throw;
    }

    const auto error = getLoadError();
    RTP_LLM_LOG_ERROR("EPLB plan for layer %d was not installed; keeping current model weights and routing: %s",
                      layer_id,
                      error.empty() ? "one or more ranks failed to load the plan" : error.c_str());
    load_flags_.setLoading();
    update_cnt_        = 0;
    balance_layer_cnt_ = 0;
    setPlanStatus(EplbPlanStatus::INIT);
    resetPlan(true);
}

void ExpertBalancer::setLoadError(std::string error) {
    lock_guard<mutex> lock(load_error_mutex_);
    load_error_ = std::move(error);
}

std::string ExpertBalancer::getLoadError() const {
    lock_guard<mutex> lock(load_error_mutex_);
    return load_error_;
}

void ExpertBalancer::applyPlanWeights(ModelBase& model) {
    // NewLoader's Python model owns stable runtime tensor addresses.  Its
    // background callback already returned the exact postprocessed layout and
    // validated it against the live layer, so install those CPU tensors
    // directly at the request boundary.  This also avoids forcing non-block
    // quantization layouts through the legacy FP8-block staging formula.
    auto& loaded = eplb_plan_tensors_;
    if (model.applyEplbPlan(loaded.layer_id,
                            loaded.phy2log,
                            loaded.logic_expert_cnt,
                            loaded.log2phy,
                            loaded.moe_weight_1,
                            loaded.moe_weight_2,
                            loaded.moe_scale_1,
                            loaded.moe_scale_2)) {
        executor_collector_.update_weights_qps = true;
        executor_collector_.update_layer_id    = loaded.layer_id;
        return;
    }

    // Legacy models retain their existing GPU staging + pointer-swap path.
    processPlanWeights();
    auto& result         = eplb_plan_buffers_;
    auto& balanced_layer = model.weights_.layers[result.layer_id].ffn_weights;

    // Exchange DenseWeights kernel/scales/zeros with eplb plan tensors.
    auto exchange_dense_weight = [&result](DenseWeights& dense, torch::Tensor& plan_weight, torch::Tensor& plan_scale) {
        auto old_kernel = dense.kernel;
        auto old_scales = dense.scales;
        auto old_zeros  = dense.zeros;

        // Set model weights from plan tensors
        dense.kernel = plan_weight;
        if (result.is_quantized) {
            dense.scales = plan_scale;
            dense.zeros  = torch::Tensor();
        } else {
            dense.scales = torch::Tensor();
            dense.zeros  = torch::Tensor();
        }

        // Store old model weights back into plan tensors for next cycle
        plan_weight = old_kernel;
        if (result.is_quantized && old_scales.defined()) {
            plan_scale = old_scales;
        }
    };

    // Exchange torch::Tensor directly (for non-quantized fields)
    auto exchange_tensors = [](torch::Tensor& model_tensor, torch::Tensor& plan_tensor) {
        std::swap(model_tensor, plan_tensor);
    };

    exchange_dense_weight(*balanced_layer.moe_gate_weight, result.moe_weight_1, result.moe_scale_1);
    exchange_dense_weight(*balanced_layer.moe_down_weight, result.moe_weight_2, result.moe_scale_2);
    exchange_tensors(balanced_layer.log2phy, result.log2phy);
    exchange_tensors(balanced_layer.logic_expert_cnt, result.logic_expert_cnt);
}

LoadFlags::SyncStatus ExpertBalancer::syncPlanWeightsLoadStatus(EplbPlanStatus status) {
    if (status == EplbPlanStatus::LOADED) {
        load_flags_.setReady();
    } else if (status == EplbPlanStatus::FAILED) {
        load_flags_.setFailed();
    } else {
        load_flags_.setLoading();
    }
    return load_flags_.sync(world_size_);
}

void ExpertBalancer::updateStats(OverallExpertStats& stats) {
    stats_.log_stats_gpu.add_(stats.stats_buf.log_stats_buf);
    stats_.gpu_loads_gpu.add_(stats.stats_buf.gpu_loads_buf);
}

}  // namespace rtp_llm
