#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"

using namespace rtp_llm;

namespace unittest {

class EplbPyWrapperOP: public torch::jit::CustomClassHolder {
public:
    void init(py::object py_eplb) {
        eplb_py_wrapper_ = ExpertBalancerPythonWrapper(py_eplb);
    }

    void createBalancePlan(torch::Tensor log_stats, torch::Tensor gpu_loads) {
        eplb_py_wrapper_.createBalancePlan(log_stats, gpu_loads, eplb_plan_);
    }

    void loadMoeWeight(int64_t ep_rank, int64_t ep_size) {
        eplb_py_wrapper_.loadBalanceWeight(ep_rank, ep_size, eplb_plan_);
    }

    bool supportsPythonWeightInstall() {
        return eplb_py_wrapper_.supportsPythonWeightInstall();
    }

    void rollbackBalancePlan(int64_t layer_id) {
        eplb_py_wrapper_.rollbackBalancePlan(layer_id);
    }

    void commitBalancePlan(int64_t layer_id) {
        eplb_py_wrapper_.commitBalancePlan(layer_id);
    }

    std::tuple<bool, bool, bool> planBufferState(bool allocate_weight_buffers) {
        EplbPlanBuffers buffers;
        buffers.init(4, 6, 8, 8, 2, DataType::TYPE_FP16, QuantAlgo(), allocate_weight_buffers);
        const bool metadata_defined = buffers.layer_id_buf.defined() && buffers.logic_expert_cnt.defined()
                                      && buffers.log2phy.defined() && buffers.phy2log.defined();
        return {metadata_defined, buffers.moe_weight_1.defined(), buffers.moe_weight_2.defined()};
    }

    EplbPlanTensors::tuple_type getResult() {
        return eplb_plan_.to_tuple();
    }

private:
    ExpertBalancerPythonWrapper eplb_py_wrapper_;
    EplbPlanTensors             eplb_plan_;
};

PYBIND11_MODULE(libth_eplb_py_wrapper_test, m) {
    py::class_<EplbPyWrapperOP>(m, "EplbPyWrapperOP")
        .def(py::init<>())
        .def("init", &EplbPyWrapperOP::init)
        .def("create_balance_plan", &EplbPyWrapperOP::createBalancePlan, py::arg("log_stats"), py::arg("gpu_loads"))
        .def("load_moe_weight", &EplbPyWrapperOP::loadMoeWeight, py::arg("ep_rank"), py::arg("ep_size"))
        .def("supports_python_weight_install", &EplbPyWrapperOP::supportsPythonWeightInstall)
        .def("rollback_balance_plan", &EplbPyWrapperOP::rollbackBalancePlan, py::arg("layer_id"))
        .def("commit_balance_plan", &EplbPyWrapperOP::commitBalancePlan, py::arg("layer_id"))
        .def("plan_buffer_state", &EplbPyWrapperOP::planBufferState, py::arg("allocate_weight_buffers"))
        .def("get_result", &EplbPyWrapperOP::getResult);
}
}  // namespace unittest
