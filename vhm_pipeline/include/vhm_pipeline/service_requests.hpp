// include/vhm_pipeline/service_requests.hpp

#pragma once

#include "rclcpp/rclcpp.hpp"
//#include "vhm_interfaces/srv/generate_references.hpp"
// #include "vhm_interfaces/srv/load_reference_bank.hpp"
// #include "vhm_interfaces/srv/segment_scene.hpp"

#include <chrono>
#include <memory>
#include <string>

namespace vhm_pipeline
{

template<typename ServiceT, typename ResponseHandler, typename FailureHandler>
void call_service_async(
  typename rclcpp::Client<ServiceT>::SharedPtr client,
  typename ServiceT::Request::SharedPtr request,
  const std::string & service_name,
  ResponseHandler on_success,
  FailureHandler on_failure,
  std::chrono::seconds wait_timeout = std::chrono::seconds(2))
{
  if (!client) {
    on_failure("Client is null: " + service_name);
    return;
  }

  if (!client->wait_for_service(wait_timeout)) {
    on_failure("Service not available: " + service_name);
    return;
  }

  client->async_send_request(
    request,
    [service_name, on_success, on_failure](
      typename rclcpp::Client<ServiceT>::SharedFuture future) mutable
    {
      try {
        auto response = future.get();

        if (!response) {
          on_failure("Null response from service: " + service_name);
          return;
        }

        if (!response->success) {
          on_failure("Service failed [" + service_name + "]: " + response->message);
          return;
        }

        on_success(response);
      }
      catch (const std::exception & e) {
        on_failure("Exception from service [" + service_name + "]: " + std::string(e.what()));
      }
    });
}
/*
template<typename SuccessHandler, typename FailureHandler>
void request_image_generation(
  rclcpp::Client<vhm_interfaces::srv::GenerateReferences>::SharedPtr client,
  const std::string & positive_prompt,
  const std::string & negative_prompt,
  int32_t num_images,
  int32_t seed,
  bool save_generated_images,
  bool save_reference_bank,
  const std::string & reference_bank_id,
  SuccessHandler on_success,
  FailureHandler on_failure)
{
  auto request = std::make_shared<vhm_interfaces::srv::GenerateReferences::Request>();

  request->positive_prompt = positive_prompt;
  request->negative_prompt = negative_prompt;
  request->num_images = num_images;
  request->seed = seed;
  request->save_generated_images = save_generated_images;
  request->save_reference_bank = save_reference_bank;
  request->reference_bank_id = reference_bank_id;

  call_service_async<vhm_interfaces::srv::GenerateReferences>(
    client,
    request,
    "generate_references",
    on_success,
    on_failure);
}*/

}  // namespace vhm_pipeline