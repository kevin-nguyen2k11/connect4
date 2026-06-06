#ifndef C___CONNECT_4_MODEL_H
#define C___CONNECT_4_MODEL_H

#include <Config.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/lib/core/stringpiece.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/lib/strings/str_util.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/session.h>
#include <string>
#include <vector>
#include <array>
#include <utility>

class Model {
public:
    explicit Model(const std::string& directory) { load_new(directory); }

    void load_new(const std::string& directory)
    {
        tensorflow::Status load_graph_status = tensorflow::LoadSavedModel(
                session_options,run_options,directory,{"serve"},&model);
        if (!load_graph_status.ok()) {
            LOG(ERROR) << load_graph_status;
        }
    }

    void predict(std::array<tensorflow::Tensor, Config::num_threads>&& inputs, std::vector<tensorflow::Tensor>* outputs)
    {
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        std::vector<tensorflow::Input> test;
        for (int i{0}; i<Config::num_threads; ++i) {
            test.emplace_back(std::move(inputs[i]));
        }
        tensorflow::InputList input_list{test};
        auto stacked{tensorflow::ops::Stack(root,std::move(input_list))};
        std::vector<tensorflow::Tensor> temp;
        tensorflow::ClientSession session(root);
        TF_CHECK_OK(session.Run({stacked},&temp));
        tensorflow::Tensor result{std::move(temp[0])};

        tensorflow::Status status = model.session->Run({{"serving_default_board:0",result}},
                                                       {"StatefulPartitionedCall:0",
                                                        "StatefulPartitionedCall:1"},
                                                       {},outputs);
        if (!status.ok()) { LOG(ERROR) << status; }
    }

private:

    const tensorflow::RunOptions run_options{};
    const tensorflow::SessionOptions session_options{};
    tensorflow::SavedModelBundle model;
};

#endif //C___CONNECT_4_MODEL_H
