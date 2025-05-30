import numpy as np
import open3d as o3d

# 生成模拟草莓茎形状的细长、轻微弯曲的圆柱状噪声点云
def generate_strawberry_stem_points(num_points=1000):
    # 定义圆柱体的参数
    height = 0.05  # 更长的长度
    radius = 0.01  # 更细的半径
    curvature = 0.01  # 控制轻微的弯曲程度

    # 生成轻微弯曲的细长圆柱体的点
    theta = np.linspace(0, 2 * np.pi, num_points)
    z = np.linspace(0, height, num_points)

    x = np.zeros_like(z)
    y = np.zeros_like(z)

    # 合并x, y, z坐标
    points = np.vstack((x, y, z)).T

    # 添加随机噪声
    noise = np.random.normal(0, 0.0005, points.shape)  # 减少噪声以使其更光滑
    points += noise

    # 将点云数据旋转和平移到xy-z象限
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 4, np.pi / 6, np.pi / 3))
    points = points @ rotation_matrix[:3, :3].T
    points += np.array([0.05, 0.05, 0.05])

    return points

# 计算向量之间的夹角
def compute_angle(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

# 创建简单的文本标签
def create_text_label(text, position, color=[0, 0, 0]):
    # 创建一个简单的立方体作为文本标签的占位符
    cube = o3d.geometry.TriangleMesh.create_box(width=0.01, height=0.01, depth=0.01)
    cube.translate(position)
    cube.paint_uniform_color(color)
    return cube

# 主函数
def main(trim_ratio=0.1):
    # 生成点云数据
    points = generate_strawberry_stem_points()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 估计法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

    # 计算点云中心
    center = pcd.get_center()

    # 视角中心点
    view_point = np.array([0, 2, 0])

    # 计算视角中心点到点云中心的连线
    center_vector = center - view_point

    # 计算视角中心点到点云每个点法向量的连线
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    view_to_point_vectors = points - view_point

    # 计算夹角
    angles = [compute_angle(center_vector, view_to_point_vector) for view_to_point_vector in view_to_point_vectors]

    # 排除两端一定比例的点
    sorted_indices = np.argsort(angles)
    num_points = len(sorted_indices)
    trim_start = int(trim_ratio * num_points)
    trim_end = int((1 - trim_ratio) * num_points)
    trimmed_indices = sorted_indices[trim_start:trim_end]

    # 找到最小夹角对应的点
    min_angle_index = trimmed_indices[np.argmin(np.array(angles)[trimmed_indices])]
    pickup_point = points[min_angle_index]

    # 可视化
    pcd.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色表示所有点

    # 可视化夹取点
    pickup_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    pickup_sphere.translate(pickup_point)
    pickup_sphere.paint_uniform_color([1, 0, 0])  # 夹取点为红色

    # 可视化夹取点法向量
    pickup_normal = normals[min_angle_index]
    pickup_line_points = np.array([pickup_point, pickup_point + 0.05 * pickup_normal])
    pickup_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pickup_line_points),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    pickup_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色表示夹取方向

    # 创建坐标轴
    axis_length = 0.1
    axis_points = np.array([
        [0, 0, 0], [axis_length, 0, 0],
        [0, 0, 0], [0, axis_length, 0],
        [0, 0, 0], [0, 0, axis_length]
    ])
    axis_lines = np.array([
        [0, 1], [0, 2], [0, 3]
    ])
    axis_colors = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ])
    axis = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axis_points),
        lines=o3d.utility.Vector2iVector(axis_lines)
    )
    axis.colors = o3d.utility.Vector3dVector(axis_colors)

    # 创建坐标轴标签
    # text_x = create_text_label("X", [axis_length, 0, 0], color=[1, 0, 0])
    # text_y = create_text_label("Y", [0, axis_length, 0], color=[0, 1, 0])
    # text_z = create_text_label("Z", [0, 0, axis_length], color=[0, 0, 1])

    # 可视化所有元素
    o3d.visualization.draw_geometries([pcd, pickup_sphere, pickup_line, axis])

if __name__ == "__main__":
    main(trim_ratio=0.1)  # 设置排除两端点的比例为10%
