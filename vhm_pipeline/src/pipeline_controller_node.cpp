#include "rclcpp/rclcpp.hpp"
#include <chrono>
#include <memory>

using namespace std::chrono_literals;

enum class PipelineState
{
    IDLE,
    WAIT_FOR_COMMAND,
    VALIDATE_COMMAND, //Maybe  PARSE_COMMAND
    GENERATE_REFERENCES,
    SEGMENT_SCENE,
    COMPUTE_MATCHES,
    SELECT_BEST_MATCH,
    ESTIMATE_POSE,
    SAVE_RESULTS,
    EXECUTE_MANIPULATION,
    VERIFY_EXECUTION,
    SUCCESS,
    FAILURE

};

class PipelineControllerNode : public rclcpp::Node
{
public:
  PipelineControllerNode() : Node("pipeline_controller_node")
  {
    cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    auto sub_options = rclcpp::SubscriptionOptions();
    sub_options.callback_group = cb_group_;

    current_state_ = PipelineState::IDLE;
    control_timer_ = this->create_wall_timer(
      100ms, // Control loop runs every 100ms
      std::bind(&PipelineControllerNode::control_loop, this), cb_group_);
    RCLCPP_INFO(this->get_logger(), "Pipeline Controller Node has been initialized.");
  }

private:

  void control_loop(){
    // TODO* Implement the control loop for the pipeline state machine
    // This function will be called periodically to check the current state and perform actions accordingly
    switch (current_state_)
    {
    case PipelineState::IDLE:
      RCLCPP_INFO(this->get_logger(), "State: IDLE, wainting for command...");
      /* code */
      break;
    
    default:
    RCLCPP_INFO(this->get_logger(), "State: %d, executing state actions...", static_cast<int>(current_state_));
      break;
    }
  }



  // RCLCPP timers, publishers, subscribers, and service clients
  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  // FSM state variables
  PipelineState current_state_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PipelineControllerNode>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}