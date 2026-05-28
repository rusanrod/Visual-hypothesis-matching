#include <rclcpp/rclcpp.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <map>

#include <nlohmann/json.hpp>
#include "vhm_interfaces/srv/generate_references.hpp"

using json = nlohmann::json;
using GenerateReferences = vhm_interfaces::srv::GenerateReferences;

class ReferenceBankPipeline : public rclcpp::Node
{
public:
    ReferenceBankPipeline()
    : Node("reference_bank_pipeline")
    {
        this->declare_parameter<int>("seed", 42);

        service_name_ = "/vhm_core/generate_references";
        seed_ = this->get_parameter("seed").as_int();

        image_counts_ = {5, 10, 15};

        client_ = this->create_client<GenerateReferences>(service_name_);

        load_instances();

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&ReferenceBankPipeline::start, this)
        );
    }

private:
    void load_instances()
    {
        std::ifstream file("/home/rusanrod/vhm_ws/src/vhm_results/instances.json");

        if (!file.is_open()) {
            throw std::runtime_error("Could not open instances json: " + instances_json_);
        }

        json data;
        file >> data;

        for (auto& item : data.items()) {
            instances_.push_back({item.key(), item.value().get<std::string>()});
        }

        RCLCPP_INFO(this->get_logger(), "Loaded %ld instances.", instances_.size());
    }


    void start()
    {
        timer_->cancel();

        if (!client_->wait_for_service(std::chrono::seconds(10))) {
            RCLCPP_ERROR(this->get_logger(), "Service not available: %s", service_name_.c_str());
            rclcpp::shutdown();
            return;
        }
    
        current_instance_idx_ = 0;
        current_count_idx_ = 0;

        send_next_request();

    }
    void send_next_request()
        {
            if (current_instance_idx_ >= instances_.size()) {
                RCLCPP_INFO(this->get_logger(), "Reference bank generation finished.");
                rclcpp::shutdown();
                return;
            }

            const auto& [instance_id, description] = instances_[current_instance_idx_];
            const int num_images = image_counts_[current_count_idx_];

            const std::string experiment_id = instance_id + "_" + std::to_string(num_images);

            auto request = std::make_shared<GenerateReferences::Request>();

            request->prompt = description;
            request->num_images = num_images;
            request->seed = seed_;

            request->save_reference_bank = true;
            request->experiment_id = experiment_id;

            RCLCPP_INFO(
                this->get_logger(),
                "Generating bank: %s | num_images=%d",
                experiment_id.c_str(),
                num_images
            );

            client_->async_send_request(
                request,
                [this, experiment_id]( rclcpp::Client<GenerateReferences>::SharedFuture future) {
                    this->handle_response(future, experiment_id);
                }
            );
        }

    
    void handle_response(
        rclcpp::Client<GenerateReferences>::SharedFuture future,
        const std::string& experiment_id)
        {
            auto response = future.get();

            if (!response->success) {
                RCLCPP_WARN(
                    this->get_logger(),
                    "Generation failed for %s: %s",
                    experiment_id.c_str(),
                    response->message.c_str()
                );
            } else {
                RCLCPP_INFO(
                    this->get_logger(),
                    "Generated %d references for %s",
                    response->generated_reference_count,
                    experiment_id.c_str()
                );
            }

            advance_indices();
            send_next_request();
        }

    void advance_indices()
        {
            current_count_idx_++;

            if (current_count_idx_ >= image_counts_.size()) {
                current_count_idx_ = 0;
                current_instance_idx_++;
            }
        }

    std::string instances_json_;
    std::string service_name_;
    int seed_;

    std::vector<int> image_counts_;
    size_t current_instance_idx_ = 0;
    size_t current_count_idx_ = 0;
    std::vector<std::pair<std::string, std::string>> instances_;

    rclcpp::Client<GenerateReferences>::SharedPtr client_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ReferenceBankPipeline>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}