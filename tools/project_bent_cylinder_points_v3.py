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

    # x = radius * np.sin(theta)
    # y = radius * np.cos(theta) + curvature * z  # 在y方向添加轻微的弯曲

    # 合并x, y, z坐标
    points = np.vstack((x, y, z)).T

    # 添加随机噪声
    noise = np.random.normal(0, 0.0005, points.shape)  # 减少噪声以使其更光滑
    points += noise

    return points

# 使用Open3D的法向量估计和平面分割来计算平面和抓取点
def compute_grasp_point_and_plane(pcd):
    # 估计法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

    # 使用RANSAC平面分割
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model
    plane_normal = np.array([a, b, c])  # 平面法向量
    inlier_cloud = pcd.select_by_index(inliers)

    # 找到最优抓取点，即法向量与平面法线最接近的点
    normals = np.asarray(inlier_cloud.normals)
    points = np.asarray(inlier_cloud.points)
    dot_products = np.abs(np.dot(normals, plane_normal))
    best_index = np.argmax(dot_products)  # 最大化法向量与平面法线的对齐程度
    grasp_point = points[best_index]
    grasp_normal = normals[best_index]

    # 生成平面网格用于可视化
    plane_center = inlier_cloud.get_center()
    plane_size = 0.05  # 平面的大小，可以根据需求调整
    v1 = np.cross(plane_normal, [1, 0, 0])  # 平面内的一个方向
    if np.linalg.norm(v1) == 0:
        v1 = np.cross(plane_normal, [0, 1, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(plane_normal, v1)

    plane_points = np.array([
        plane_center + plane_size * (v1 + v2),
        plane_center + plane_size * (v1 - v2),
        plane_center - plane_size * (v1 + v2),
        plane_center - plane_size * (v1 - v2)
    ])

    # 创建平面网格
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(plane_points)
    plane_mesh.triangles = o3d.utility.Vector3iVector([
        [0, 1, 2], [0, 2, 3]
    ])
    plane_mesh.compute_vertex_normals()
    plane_mesh.paint_uniform_color([1, 0.7, 0])  # 橙色表示平面

    return grasp_point, grasp_normal, plane_mesh

# 主函数
def main():
    # 生成点云数据
    points = generate_strawberry_stem_points()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 使用法向量估计和平面分割确定草莓茎的平面和抓取点
    grasp_point, grasp_normal, plane_mesh = compute_grasp_point_and_plane(pcd)

    # 可视化
    pcd.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色表示所有点

    # 可视化抓取点
    grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    grasp_sphere.translate(grasp_point)
    grasp_sphere.paint_uniform_color([1, 0, 0])  # 抓取点为红色

    # 可视化抓取点法向量
    grasp_line_points = np.array([grasp_point, grasp_point + 0.05 * grasp_normal])
    grasp_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grasp_line_points),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    grasp_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色表示抓取方向

    # 可视化平面法向量
    plane_center = plane_mesh.get_center()
    plane_line_points = np.array([plane_center, plane_center + 0.05 * grasp_normal])
    plane_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(plane_line_points),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    plane_line.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # 蓝色表示平面法线

    # 可视化所有元素
    # o3d.visualization.draw_geometries([pcd, plane_mesh, grasp_sphere, grasp_line, plane_line])
    o3d.visualization.draw_geometries([pcd, plane_mesh, grasp_sphere, grasp_line])

if __name__ == "__main__":
    main()
