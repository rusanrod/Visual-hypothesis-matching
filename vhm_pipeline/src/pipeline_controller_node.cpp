#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "vhm_interfaces/action/execute_vhm.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

using namespace std::chrono_literals;

enum class PipelineState
{
  IDLE,
  VALIDATE_GOAL,
  PREPARE_INPUTS,
  PREPARE_REFERENCES,
  SEGMENT_SCENE,
  COMPUTE_MATCHES,
  SELECT_BEST_MATCH,
  ESTIMATE_POSE,
  SAVE_RESULTS,
  WAIT_FOR_RESPONSE,
  SUCCESS,
  FAILURE,
  CANCELED
};

struct PipelineContext
{
  std::string raw_command;

  std::string positive_prompt;
  std::string negative_prompt;
  std::string class_name;

  bool use_image_enhancement{false};
  bool estimate_pose{false};

  int32_t num_synthetic_images{10};
  int32_t seed{-1};

  bool save_generated_images{false};
  bool save_debug_artifacts{false};

  bool save_reference_bank{false};
  bool use_reference_bank{false};
  std::string reference_bank_id;

  float match_threshold{0.0f};

  int32_t generated_reference_count{0};
  bool used_reference_bank{false};

  int32_t best_candidate_index{-1};
  float best_score{0.0f};
  bool pose_estimated{false};

  std::string failure_message;
};

class PipelineControllerNode : public rclcpp::Node
{
public:
  using ExecuteVHM = vhm_interfaces::action::ExecuteVHM;
  using GoalHandleExecuteVHM = rclcpp_action::ServerGoalHandle<ExecuteVHM>;

  PipelineControllerNode() : Node("pipeline_controller_node")
  {
    cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    current_state_ = PipelineState::IDLE;
    busy_ = false;

    control_timer_ = this->create_wall_timer(
      100ms,
      std::bind(&PipelineControllerNode::control_loop, this),
      cb_group_);

    pipeline_action_server_ = rclcpp_action::create_server<ExecuteVHM>(
      this,
      "execute_vhm",
      std::bind(&PipelineControllerNode::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&PipelineControllerNode::handle_cancel, this, std::placeholders::_1),
      std::bind(&PipelineControllerNode::handle_accepted, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), "Pipeline Controller Node initialized.");
  }

private:

  // === Action server handlers ===
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & /*uuid*/,
    std::shared_ptr<const ExecuteVHM::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received ExecuteVHM goal request.");

    if (busy_) {
      RCLCPP_WARN(this->get_logger(), "Pipeline is busy. Rejecting new goal.");
      return rclcpp_action::GoalResponse::REJECT;
    }

    if (!goal) {
      RCLCPP_WARN(this->get_logger(), "Received null goal.");
      return rclcpp_action::GoalResponse::REJECT;
    }

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleExecuteVHM> goal_handle)
  {
    (void)goal_handle;
    RCLCPP_WARN(this->get_logger(), "Received request to cancel current goal.");

    if (!busy_) {
      return rclcpp_action::CancelResponse::REJECT;
    }

    cancel_requested_ = true;
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleExecuteVHM> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Goal accepted.");

    active_goal_handle_ = goal_handle;
    load_goal_into_context(goal_handle->get_goal());

    busy_ = true;
    cancel_requested_ = false;
    current_state_ = PipelineState::VALIDATE_GOAL;
  }

  // === Context managment and control loop ===
  void load_goal_into_context(const std::shared_ptr<const ExecuteVHM::Goal> goal)
  {
    context_ = PipelineContext();  // reset limpio

    context_.raw_command = goal->raw_command;

    context_.use_image_enhancement = goal->use_image_enhancement;
    context_.estimate_pose = goal->estimate_pose;

    context_.num_synthetic_images = goal->num_synthetic_images;
    context_.seed = goal->seed;

    context_.save_generated_images = goal->save_generated_images;
    context_.save_debug_artifacts = goal->save_debug_artifacts;

    context_.save_reference_bank = goal->save_reference_bank;
    context_.use_reference_bank = goal->use_reference_bank;
    context_.reference_bank_id = goal->reference_bank_id;

    context_.match_threshold = goal->match_threshold;
  }

  void control_loop()
  {
    if (!busy_) {
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 10000,
        "State: IDLE, waiting for ExecuteVHM goals.");
      return;
    }

    if (cancel_requested_) {
      current_state_ = PipelineState::CANCELED;
    }

    //publish_feedback( "Pipeline running");

    switch (current_state_) {
      case PipelineState::VALIDATE_GOAL:
        step_validate_goal();
        break;

      case PipelineState::PREPARE_INPUTS:
        step_prepare_inputs();
        break;

      case PipelineState::PREPARE_REFERENCES:
        step_prepare_references();
        break;

      case PipelineState::SEGMENT_SCENE:
        step_segment_scene();
        break;

      case PipelineState::COMPUTE_MATCHES:
        step_compute_matches();
        break;

      case PipelineState::SELECT_BEST_MATCH:
        step_select_best_match();
        break;

      case PipelineState::ESTIMATE_POSE:
        step_estimate_pose();
        break;

      case PipelineState::SAVE_RESULTS:
        step_save_results();
        break;

      case PipelineState::SUCCESS:
        finalize_success();
        break;

      case PipelineState::FAILURE:
        finalize_failure(context_.failure_message);
        break;

      case PipelineState::CANCELED:
        finalize_canceled();
        break;

      case PipelineState::IDLE:
      default:
        break;
    }
  }

    // === Pipeline steps ===
    void step_validate_goal()
    {
      RCLCPP_INFO(this->get_logger(), "State: VALIDATE_GOAL");

      if (context_.raw_command.empty()) {
          context_.failure_message = "raw_command is empty.";
          current_state_ = PipelineState::FAILURE;
          return;
        }


      if (context_.num_synthetic_images <= 0 && !context_.use_reference_bank) {
        context_.failure_message = "num_synthetic_images must be > 0 when not using reference bank.";
        current_state_ = PipelineState::FAILURE;
        return;
      }

      current_state_ = PipelineState::PREPARE_INPUTS;
    }

  void step_prepare_inputs()
  {
    RCLCPP_INFO(this->get_logger(), "State: PREPARE_INPUTS");

    // TODO:
    // 1. Parse raw_command
    // 2. Build positive_prompt y negative_prompt

    // Parseo provisional
    context_.positive_prompt = context_.raw_command;
    context_.negative_prompt = "blurry, low quality, distorted, occluded";

    current_state_ = PipelineState::PREPARE_REFERENCES;
  }

  void step_prepare_references()
  {
    RCLCPP_INFO(this->get_logger(), "State: PREPARE_REFERENCES");

    if (context_.use_reference_bank) {
      //request_load_reference_bank();
    } else {
      //request_generate_references();
    }

    current_state_ = PipelineState::SEGMENT_SCENE;
  }

  void step_segment_scene()
  {
    RCLCPP_INFO(this->get_logger(), "State: SEGMENT_SCENE");

    // Aquí luego:
    // - llamas a FastSAM
    // - obtienes masks/crops candidatos
    current_state_ = PipelineState::COMPUTE_MATCHES;
  }

  void step_compute_matches()
  {
    RCLCPP_INFO(this->get_logger(), "State: COMPUTE_MATCHES");

    // Aquí luego:
    // - computes embeddings
    // - comparas
    // - llenas score provisional
    context_.best_candidate_index = 0;
    context_.best_score = 0.85f;

    current_state_ = PipelineState::SELECT_BEST_MATCH;
  }

  void step_select_best_match()
  {
    RCLCPP_INFO(this->get_logger(), "State: SELECT_BEST_MATCH");

    if (context_.best_candidate_index < 0) {
      context_.failure_message = "No candidate was selected.";
      current_state_ = PipelineState::FAILURE;
      return;
    }

    if (context_.best_score < context_.match_threshold) {
      context_.failure_message = "Best candidate score is below threshold.";
      current_state_ = PipelineState::FAILURE;
      return;
    }

    if (context_.estimate_pose) {
      current_state_ = PipelineState::ESTIMATE_POSE;
    } else {
      current_state_ = PipelineState::SAVE_RESULTS;
    }
  }

  void step_estimate_pose()
  {
    RCLCPP_INFO(this->get_logger(), "State: ESTIMATE_POSE");

    // Aquí luego:
    // - mask/crop -> centroid/pose con nube de puntos
    context_.pose_estimated = true;

    current_state_ = PipelineState::SAVE_RESULTS;
  }

  void step_save_results()
  {
    RCLCPP_INFO(this->get_logger(), "State: SAVE_RESULTS");

    // Aquí luego:
    // - guardar máscara, crop, logs, overlays, .pth, etc.
    current_state_ = PipelineState::SUCCESS;
  }

  void publish_feedback(
    const std::string & current_stage,
    float progress,
    const std::string & status_message)
  {
    if (!active_goal_handle_) {
      return;
    }

    auto feedback = std::make_shared<ExecuteVHM::Feedback>();
    feedback->current_stage = current_stage;
    feedback->progress = progress;
    feedback->status_message = status_message;
    feedback->current_candidate_count = 0;
    feedback->current_reference_count = context_.generated_reference_count;

    active_goal_handle_->publish_feedback(feedback);
  }

  void finalize_success()
  {
    if (!active_goal_handle_) {
      reset_pipeline();
      return;
    }

    auto result = std::make_shared<ExecuteVHM::Result>();
    result->success = true;
    result->final_state = "SUCCESS";
    result->message = "Pipeline executed successfully.";
    result->class_name = context_.class_name;
    result->generated_reference_count = context_.generated_reference_count;
    result->used_reference_bank = context_.used_reference_bank;
    result->best_candidate_index = context_.best_candidate_index;
    result->best_score = context_.best_score;
    result->pose_estimated = context_.pose_estimated;

    active_goal_handle_->succeed(result);
    RCLCPP_INFO(this->get_logger(), "Pipeline finished successfully.");

    reset_pipeline();
  }

  void finalize_failure(const std::string & message)
  {
    if (!active_goal_handle_) {
      reset_pipeline();
      return;
    }

    auto result = std::make_shared<ExecuteVHM::Result>();
    result->success = false;
    result->final_state = "FAILURE";
    result->message = message;
    result->class_name = context_.class_name;
    result->generated_reference_count = context_.generated_reference_count;
    result->used_reference_bank = context_.used_reference_bank;
    result->best_candidate_index = -1;
    result->best_score = 0.0f;
    result->pose_estimated = false;

    active_goal_handle_->abort(result);
    RCLCPP_ERROR(this->get_logger(), "Pipeline failed: %s", message.c_str());

    reset_pipeline();
  }

  void finalize_canceled()
  {
    if (!active_goal_handle_) {
      reset_pipeline();
      return;
    }

    auto result = std::make_shared<ExecuteVHM::Result>();
    result->success = false;
    result->final_state = "CANCELED";
    result->message = "Pipeline execution canceled.";
    result->class_name = context_.class_name;
    result->generated_reference_count = context_.generated_reference_count;
    result->used_reference_bank = context_.used_reference_bank;
    result->best_candidate_index = -1;
    result->best_score = 0.0f;
    result->pose_estimated = false;

    active_goal_handle_->canceled(result);
    RCLCPP_WARN(this->get_logger(), "Pipeline was canceled.");

    reset_pipeline();
  }

  void reset_pipeline()
  {
    active_goal_handle_.reset();
    context_ = PipelineContext();
    busy_ = false;
    cancel_requested_ = false;
    current_state_ = PipelineState::IDLE;
  }


  rclcpp_action::Server<ExecuteVHM>::SharedPtr pipeline_action_server_;
  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  std::shared_ptr<GoalHandleExecuteVHM> active_goal_handle_;
  PipelineContext context_;

  PipelineState current_state_;
  bool busy_{false};
  bool cancel_requested_{false};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PipelineControllerNode>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}