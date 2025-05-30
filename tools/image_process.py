import struct
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import open3d as o3d

def image_proc(self):
    # if self.DEBUG==1:
    #     self.get_logger().info('Received an image! ')
    ros_rgb_image, ros_depth_image, depth_camera_info = self.image_queue.get(block=True)

    # Convert ROS images to numpy arrays
    rgb_image = np.ndarray(shape=(ros_rgb_image.height, ros_rgb_image.width, 3), dtype=np.uint8,
                           buffer=ros_rgb_image.data)
    depth_image = np.ndarray(shape=(ros_depth_image.height, ros_depth_image.width), dtype=np.uint16,
                             buffer=ros_depth_image.data)
    result_image = np.copy(rgb_image)

    # Flatten depth image and set invalid depth to 0
    h, w = depth_image.shape[:2]
    depth = np.copy(depth_image).reshape((-1,))
    depth[depth <= 0] = 0

    # Convert depth image to gray scale image
    sim_depth_image = np.clip(depth_image, 0, 4000).astype(np.float64)
    sim_depth_image = sim_depth_image / 2000.0 * 255.0

    # Map depth to color image
    depth_color_map = cv2.applyColorMap(sim_depth_image.astype(np.uint8), cv2.COLORMAP_JET)

    # Perform object detection
    boxes, confs, classes, masks, result_mask = self.yolov8.infer(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    i = 0
    possiable_locations = []
    for box, cls_conf, cls_id in zip(boxes, confs, classes):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        depth_value = depth_image[int(center_y), int(center_x)]

        # Get class name
        # if cls_id >= self.cls_len:
        #     cls_name = "unknown"
        # else:
        #     cls_name = TRT_CLASS_STRAWBERRY[cls_id]

        result_image = cv2.putText(result_image, str(i) + " " + str(float(cls_conf))[:4], (int(x1), int(y1) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        result_image = cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
        i += 1
        if cls_conf > self.yolo_threshold:
            possiable_locations.append((cls_conf, [x1, y1, x2, y2]))

    # Camera intrinsic parameters
    self.K = depth_camera_info.k
    self.D = depth_camera_info.d
    self.fx = self.K[0]
    self.fy = self.K[4]
    self.cx = self.K[2]
    self.cy = self.K[5]

    # Transformation matrix from base to camera (example)
    T_base_to_cam = np.array([
        [1, 0, 0, 0.1],
        [0, 1, 0, 0.2],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1]
    ])

    if len(result_mask) != 0:
        # Combine all masks
        combined_mask = np.sum(result_mask, axis=0)
        binary_mask = (combined_mask > 0).astype(np.uint8)

        points = []
        colors = []
        for v in range(h):
            for u in range(w):
                if binary_mask[v, u]:
                    z = depth_image[v, u]
                    if z == 0:
                        continue
                    # Calculate 3D coordinates in camera frame
                    x = (u - self.cx) * z / self.fx
                    y = (v - self.cy) * z / self.fy

                    # Convert to homogeneous coordinates
                    point_cam = np.array([x, y, z, 1])

                    # Transform to base frame
                    point_base = np.dot(point_cam, T_base_to_cam)

                    # Get corresponding RGB values
                    color = rgb_image[v, u]
                    r, g, b = color[2], color[1], color[0]

                    # Convert RGB to float format
                    rgb_float = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]

                    # Append to point cloud lists
                    points.append([point_base[0] / 1000, point_base[1] / 1000, point_base[2] / 1000])  # Store in meters
                    colors.append([r / 255.0, g / 255.0, b / 255.0])  # Normalize RGB

        return points, colors


def find_grasp_points(points, colors, stem_mask, min_samples=10, eps=0.05):
    """
    Find the grasp points on strawberry stems.

    :param points: List of 3D points in the point cloud.
    :param colors: List of RGB colors corresponding to each point.
    :param stem_mask: Binary mask indicating which points belong to the stem.
    :param min_samples: Minimum number of samples required to form a cluster.
    :param eps: Maximum distance between two samples for them to be considered as in the same neighborhood.
    :return: List of tuples containing the position and orientation of each grasp point.
    """
    # Filter points based on the stem mask
    stem_points = [points[i] for i in range(len(points)) if stem_mask[i]]
    stem_colors = [colors[i] for i in range(len(colors)) if stem_mask[i]]

    # Convert to numpy arrays for easier manipulation
    stem_points = np.array(stem_points)
    stem_colors = np.array(stem_colors)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(stem_points)

    unique_labels = set(labels)
    grasp_points = []

    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue

        # Extract points in the current cluster
        cluster_indices = np.where(labels == label)[0]
        cluster_points = stem_points[cluster_indices]

        # Fit a line to the cluster points
        _, _, v = np.linalg.svd(cluster_points - np.mean(cluster_points, axis=0))
        line_direction = v[-1, :]  # The last row of V is the direction of the line

        # Find the midpoint of the cluster
        midpoint = np.mean(cluster_points, axis=0)

        # Calculate the orientation of the grasp point
        orientation = line_direction

        # Add the grasp point and its orientation to the list
        grasp_points.append((midpoint, orientation))

    return grasp_points
