from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

class GetScene(Node):
    def __init__(self):
        super().__init__('get_scene')

        self.declare_parameter(
            'image_topic',
            '/camera/image_raw'
        )
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        self.default_output_dir = Path.joinpath(Path.home(), "vhm_ws", "src", "vhm_results", "scene_images")
        self.default_output_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.img_msg = None

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
            )
        
        self.get_scene_srv = self.create_service(
            Trigger, 
            'vhm_visualization/get_scene', 
            self.get_scene_callback
        )

        self.get_logger().info(f"Ready to save scenes.")

    def image_callback(self, msg):
        self.img_msg = msg

    def get_scene_callback(self, request, response: Trigger.Response):

        if self.img_msg is None:
            response.success = False
            response.message = "No image received yet"
            return response
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                self.img_msg, 
                desired_encoding='bgr8'
            )
        except Exception as e:
            response.success = False
            response.message = f"Error converting image: {str(e)}"
            return response

        file_count = len(list(self.default_output_dir.glob("scene_*.png")))
        output_dir = self.default_output_dir
        cv2.imwrite(f'{output_dir}/scene_{file_count:03d}.png', cv_image)

        response.success = True
        response.message = f"Scene image saved as scene_{file_count:03d}.png"

        return response


def main(args=None):
    rclpy.init(args=args)
    node = GetScene()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()