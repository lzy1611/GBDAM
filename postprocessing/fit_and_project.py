import numpy as np
import time
import math
def best_fit_plane(points, direction_vector, axispoint):
    """
    计算给定点集的最佳拟合平面的法向量，并根据给定的方向向量调整法向量的方向。
    """
    # 计算3d点集的质心
    centroid = axispoint
    # 计算相对于质心的偏移点
    centered_points = points - centroid
    # 计算协方差矩阵
    cov_matrix = np.cov(centered_points.T)
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 最小特征值对应的特征向量即为法向量
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
    # 归一化法向量
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    print(f"normal_vector in func : {normal_vector}")

    # 计算法向量和接收的方向向量之间的夹角
    dot_product = np.dot(normal_vector, direction_vector)
    norm_product = np.linalg.norm(normal_vector) * np.linalg.norm(direction_vector)
    angle_cosine = dot_product / norm_product

    # 夹角的余弦值范围是 [-1, 1]，因此我们需要确保角度在有效范围内
    angle_cosine = max(min(angle_cosine, 1), -1)
    print(f"angle_cosine: {angle_cosine}")
    angle_radians = np.arccos(angle_cosine)
    angle_degrees = np.degrees(angle_radians)
    print(f"angle_degrees: {angle_degrees}")

    # 如果夹角大于180度，反转法向量的方向
    if angle_degrees > 90:
        normal_vector = -normal_vector
    print(f"normal_vector after: {normal_vector}")
    # 归一化normal_vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector, centroid

def create_orthogonal_basis(normal_vector):
    """
    创建一个与给定法向量正交的基。
    """
    print(f"Starting to create orthogonal basis")

    # 确保法向量已归一化
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # 选择一个坐标轴
    if np.abs(normal_vector[0]) < 1e-6:  # 如果法向量接近于垂直于 x 轴
        axis_vector = np.array([1.0, 0.0, 0.0])
    elif np.abs(normal_vector[1]) < 1e-6:  # 如果法向量接近于垂直于 y 轴
        axis_vector = np.array([0.0, 1.0, 0.0])
    else:  # 如果法向量接近于垂直于 z 轴
        axis_vector = np.array([0.0, 0.0, 1.0])

    # 通过叉乘来创建第一个正交向量
    v = np.cross(normal_vector, axis_vector)
    # 归一化第一个正交向量
    v = v / np.linalg.norm(v)

    # 创建第二个正交向量
    w = np.cross(normal_vector, v)
    # 归一化第二个正交向量
    w = w / np.linalg.norm(w)
    # 验证正交性
    dot_product_v_w = np.dot(v, w)
    print(f"The dot product of v and w is: {dot_product_v_w}")

    # 验证与法向量的正交性
    dot_product_v_n = np.dot(v, normal_vector)
    dot_product_w_n = np.dot(w, normal_vector)
    print(f"The dot product of v and normal vector is: {dot_product_v_n}")
    print(f"The dot product of w and normal vector is: {dot_product_w_n}")
    print(f"Finished creating orthogonal basis")
    return v, w


def project_to_plane(points, axispoint, v, w):
    """
    将三维点投影到由法向量定义的平面上。
    """
    # 创建变换矩阵
    transformation_matrix = np.column_stack((v, w))
    # 投影点到平面
    projected_points = np.dot(points - axispoint, transformation_matrix)
    return projected_points

def fit_and_project(points, direction_vector):
    """
    计算最佳拟合平面的法向量并投影点到该平面上。
    """
    # 计算法向量
    start_time = time.time()
    normal_vector, centroid = best_fit_plane(points, direction_vector)
    end_time = time.time()
    print(f"Time to compute normal vector: {end_time - start_time:.4f} seconds")

    # 投影
    start_time = time.time()
    projected_points = project_to_plane(points, normal_vector)
    end_time = time.time()
    print(f"Time to project points: {end_time - start_time:.4f} seconds")

    return projected_points, centroid, normal_vector

def sort_points_by_angle(points):
    """
    对给定的点集按照它们与x轴的逆时针角度进行排序。

    参数:
    points : list of tuples
        一个包含 (x, y) 坐标的点集列表。

    返回:
    sorted_points : list of tuples
        按照角度排序后的点集列表。
    """
    # 计算每个点相对于原点的角度
    angles = [(math.atan2(y, x), (x, y)) for x, y in points]

    # 按照角度排序
    sorted_angles = sorted(angles)

    # 提取排序后的点
    sorted_points = [point for _, point in sorted_angles]

    return sorted_points
# 示例使用
if __name__ == "__main__":
    # 生成示例点
    points = np.random.rand(100, 3)
    direction_vector = np.array([0, 0, 1])

    # 使用 fit_and_project 函数
    projected_points, centroid, normal_vector = fit_and_project(points, direction_vector)

    print("Projected points:")
    print(projected_points)
    print("Centroid:", centroid)
    print("Normal vector:", normal_vector)