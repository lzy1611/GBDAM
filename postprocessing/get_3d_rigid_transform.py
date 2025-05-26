def get_3d_rigid_transform(src_points, dst_points):
    import numpy as np
    if len(src_points) != len(dst_points) or len(src_points) < 3:
        return None

    points_num = len(src_points)
    src_sum_x = np.sum([src_point[0] for src_point in src_points])
    src_sum_y = np.sum([src_point[1] for src_point in src_points])
    src_sum_z = np.sum([src_point[2] for src_point in src_points])

    dst_sum_x = np.sum([dst_point[0] for dst_point in dst_points])
    dst_sum_y = np.sum([dst_point[1] for dst_point in dst_points])
    dst_sum_z = np.sum([dst_point[2] for dst_point in dst_points])

    center_src = np.array([src_sum_x / points_num, src_sum_y / points_num, src_sum_z / points_num])
    center_dst = np.array([dst_sum_x / points_num, dst_sum_y / points_num, dst_sum_z / points_num])

    src_mat = np.zeros((3, points_num))
    dst_mat = np.zeros((3, points_num))

    for i in range(points_num):
        src_mat[:, i] = src_points[i] - center_src
        dst_mat[:, i] = dst_points[i] - center_dst

    mat_s = np.dot(src_mat, dst_mat.T)
    mat_u, mat_w, mat_v = np.linalg.svd(mat_s)
    mat_temp = np.dot(mat_v.T, np.dot(mat_u, mat_v))

    det = np.linalg.det(mat_temp)
    mat_m = np.eye(3, dtype=np.double)
    mat_m[2, 2] = det

    mat_r = np.dot(mat_v.T, np.dot(mat_m, mat_u.T))


    # 计算变换后的源点集的质心
    transformed_center_src = np.dot(mat_r, center_src)

    # 计算平移向量
    delta = center_dst - transformed_center_src

    # 创建齐次变换矩阵
    r_t = np.eye(4)
    r_t[:3, :3] = mat_r
    r_t[:3, 3] = delta

    return r_t

#Example usage:
# src_points and dst_points should be lists of tuples or lists, e.g., [(x1, y1, z1), (x2, y2, z2), ...]
# src_points = [...]
# dst_points = [...]
# transform_matrix = get_3d_rigid_transform(src_points, dst_points)
import numpy as np


# 定义源点集和目标点集
# 假设我们有三个三维空间中的点
src_points = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

dst_points = np.array([
    [1.1, 2, 3],
    [4.1, 5, 6],
    [7.1, 8, 9]
])

# 调用函数计算变换矩阵
transform_matrix = get_3d_rigid_transform(src_points, dst_points)

# 打印变换矩阵
print("Transform Matrix:")
print(transform_matrix)