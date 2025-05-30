import cv2
import numpy as np
import struct


def image_proc(self):
    # if self.DEBUG==1:
    #     self.get_logger().info('Received an image! ')
    ros_rgb_image, ros_depth_image, depth_camera_info = self.image_queue.get(block=True)

    # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # rgb, depth convert nudarry
        # self.get_logger().info(f'Numbers: {ros_rgb_image.height}, {ros_rgb_image.width}, {ros_depth_image.height}, {ros_depth_image.width}')
    rgb_image = np.ndarray(shape=(ros_rgb_image.height, ros_rgb_image.width, 3), dtype=np.uint8,
                               buffer=ros_rgb_image.data)
    depth_image = np.ndarray(shape=(ros_depth_image.height, ros_depth_image.width), dtype=np.uint16,
                                 buffer=ros_depth_image.data)
    result_image = np.copy(rgb_image)
        # self.get_logger().info(str(depth_image.shape))
        # 将深度都图像展平，将无效深度信息设为0
    h, w = depth_image.shape[:2]
    depth = np.copy(depth_image).reshape((-1,))
    depth[depth <= 0] = 0

        # Convert depth image to gray scale image
    sim_depth_image = np.clip(depth_image, 0, 4000).astype(np.float64)
    sim_depth_image = sim_depth_image / 2000.0 * 255.0
        # Map depth to color image
    depth_color_map = cv2.applyColorMap(sim_depth_image.astype(np.uint8), cv2.COLORMAP_JET)

    boxes, confs, classes, masks, result_mask = self.yolov8.infer(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    i = 0
    cls_conf = None
    possiable_locations = []
    for box, cls_conf, cls_id in zip(boxes, confs, classes):
            # print(box)
        x1 = box[0]  # + self.roi[2]
        y1 = box[1]  # + self.roi[0]
        x2 = box[2]  # + self.roi[2]
        y2 = box[3]  # + self.roi[0]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        depth_value = depth_image[int(center_y), int(center_x)]
            # get class name
        if (cls_id >= self.cls_len):
            cls_name = "unknow"
            print(cls_id)
        else:
            # cls_name = TRT_CLASS_STRAWBERRY[cls_id]
            pass
        result_image = cv2.putText(result_image, str(i) + " " + str(float(cls_conf))[:4], (int(x1), int(y1) - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        result_image = cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
        i = i + 1
        if cls_conf > self.yolo_threshold:
            possiable_locations.append((cls_conf, [x1, y1, x2, y2]))
        # 相机内参信息
    self.K = depth_camera_info.k
    self.D = depth_camera_info.d
    self.fx = self.K[0]
    self.fy = self.K[4]
    self.cx = self.K[2]
    self.cy = self.K[5]


    if len(result_mask) != 0:
            # 将所有mask相加
        combined_mask = np.sum(result_mask, axis=0)
        binary_mask = (combined_mask > 0).astype(np.uint8)
            # print(len(result_mask))
            # print(result_mask[0].shape)
            # print(type(result_mask[0]))

        points = []
        colors = []
        for v in range(h):
            for u in range(w):
                if (binary_mask[v, u]):
                    z = depth_image[v, u]
                    if z == 0:
                        continue
                        # 计算3D坐标
                    x = (u - self.cx) * z / self.fx
                    y = (v - self.cy) * z / self.fy

                        # 获取对应的RGB值
                    color = rgb_image[v, u]
                    r, g, b = color[2], color[1], color[0]

                        # 将RGB转为float格式
                    rgb_float = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]

                        # 将点添加到点云列表
                    points.append([x / 1000, y / 1000, z / 1000])  # Store in meters
                    colors.append([r / 255.0, g / 255.0, b / 255.0])  # Normalize RGB
