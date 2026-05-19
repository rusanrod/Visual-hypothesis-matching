import rclpy
from rclpy.node import Node

from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from vhm_core.utlis.result_utils import VHMResultsManager


class GetSceneNode(Node):
    def __init__(self):
        super().__init__('get_scene_node')

        self.last_img_msg = None
        self.bridge = CvBridge()

        self.result_manager = VHMResultsManager(experiment_id="scenes")
        self.result_manager.prepare_scene_images_dir()

        self.img_sub = self.create_subscription(
            Image,
            '/head_rgbd_sensor/rgb/image_rect_color',
            self.scene_image_callback,
            10
        )

        self.get_scene_srv = self.create_service(
            Trigger,
            'vhm_core/get_scene',
            self.handle_get_scene
        )

        self.get_logger().info('Get Scene Node ready.')

    def scene_image_callback(self, msg: Image):
        self.last_img_msg = msg

    def handle_get_scene(self, request, response: Trigger.Response):
        if self.last_img_msg is None:
            response.success = False
            response.message = "No scene image available."
            return response

        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                self.last_img_msg,
                desired_encoding='bgr8'
            )

            scene_path = self.result_manager.make_scene_image_path(".png")

            ok = cv2.imwrite(str(scene_path), cv_image)

            if not ok:
                response.success = False
                response.message = f"Failed to save scene image: {scene_path}"
                return response

            response.success = True
            response.message = f"Scene image saved: {scene_path}"

            self.get_logger().info(response.message)
            return response

        except Exception as e:
            response.success = False
            response.message = f"Error saving scene image: {str(e)}"
            self.get_logger().error(response.message)
            return response


def main(args=None):
    rclpy.init(args=args)
    node = GetSceneNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()