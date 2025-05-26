import numpy as np
from scipy.optimize import least_squares
from geomdl import NURBS
from geomdl.visualization import VisMPL
import matplotlib.pyplot as plt
from geomdl import utilities
from scipy.spatial.distance import cdist

def fit_points_to_nurbs(points, num_ctrlpts, degree=3, plot_results=True):
    # 定义输入参数
    num_points = len(points)

    # 生成角度从0到2π
    angles = np.linspace(0, 2 * np.pi, num_ctrlpts, endpoint=False)

    # 创建控制点（圆形）
    x_circle = np.cos(angles)
    y_circle = np.sin(angles)

    # 创建控制点（圆形）
    control_points = np.vstack((x_circle, y_circle)).T

    # 为每个控制点分配相同的权重（对于圆，权重为1）
    weights = np.ones(len(control_points))

    # 由于我们需要周期性曲线，需要复制前degree个控制点
    control_points_periodic = np.concatenate((control_points, control_points[:degree]), axis=0)
    weights_periodic = np.concatenate((weights, weights[:degree]))

    # 创建NURBS曲线
    curve = NURBS.Curve()

    # 设置曲线的阶数
    curve.degree = degree

    # 设置控制点和权重
    curve.ctrlpts = control_points_periodic.tolist()
    curve.weights = weights_periodic.tolist()

    # 设置周期性的结点向量
    num_ctrlpts_total = len(control_points_periodic)
    curve.knotvector = utilities.generate_knot_vector(curve.degree, num_ctrlpts_total, clamped=False)

    # 将控制点从直角坐标转换为极坐标
    r_circle, theta_circle = np.linalg.norm(control_points, axis=1), np.arctan2(control_points[:, 1], control_points[:, 0])
    print(f"theta_circle before update: {theta_circle}")

    # 定义残差函数
    def residual(params):
        n = len(control_points)  # 控制点的数量
        radii = params[:n]
        knotvector = params[n:n + len(curve.knotvector) - 2]
        weights = params[n + len(curve.knotvector) - 2:]

        # 使用优化后的极径更新控制点
        new_ctrlpts = np.column_stack((radii * np.cos(theta_circle), radii * np.sin(theta_circle)))

        # 复制前degree个控制点以满足周期性曲线的要求
        new_ctrlpts_periodic = np.concatenate((new_ctrlpts, new_ctrlpts[:degree]), axis=0)
        new_weights_periodic = np.concatenate((weights, weights[:degree]))

        # 重建结点向量
        knot_start = [0]
        knot_end = [1]
        new_knotvector = np.hstack([knot_start, knotvector, knot_end])

        # 更新NURBS曲线的控制点、权重和结点向量
        curve.ctrlpts = new_ctrlpts_periodic.tolist()
        curve.weights = new_weights_periodic.tolist()
        curve.knotvector = new_knotvector.tolist()

        # 使用均匀分布的参数值来评估曲线上的点
        num_samples = 2000
        u_values = np.linspace(curve.knotvector[degree], curve.knotvector[-degree - 1], num=num_samples)
        curve_points_uniform = np.array([curve.evaluate_single(u) for u in u_values])

        # 遍历椭圆上的数据点，找到每个数据点最接近的曲线上的点
        closest_points = []
        for point in points:
            distances = cdist([point], curve_points_uniform)
            closest_index = np.argmin(distances)
            closest_points.append(curve_points_uniform[closest_index])

        # 计算残差，即曲线点和椭圆点之间的平方差之和
        closest_points_array = np.array(closest_points)
        distance_residual = (closest_points_array - points).flatten()
        return distance_residual

    # 使用初始极径和结点向量作为优化的初始参数
    initial_radii = r_circle
    initial_knotvector = curve.knotvector[1:-1]  # 取出内部结点向量
    initial_weights = weights

    initial_params = np.hstack((initial_radii, initial_knotvector, initial_weights))

    # 定义上下限
    lb_radii = initial_radii - 20
    ub_radii = initial_radii + 90
    lb_knotvector = np.array(initial_knotvector) - 0.01
    ub_knotvector = np.array(initial_knotvector) + 0.01
    lb_weights = initial_weights - 0.5
    ub_weights = initial_weights + 0.5
    lb = np.hstack((lb_radii, lb_knotvector, lb_weights))
    ub = np.hstack((ub_radii, ub_knotvector, ub_weights))

    result = least_squares(residual, initial_params, bounds=(lb, ub), method='trf', loss='huber', max_nfev=5000, verbose=2, ftol=1e-10, xtol=1e-10)

    optimized_params = result.x
    n = len(control_points)  # 控制点的数量
    new_radii = optimized_params[:n]
    new_knotvector = optimized_params[n:n + len(initial_knotvector)]
    new_weights = optimized_params[n + len(initial_knotvector):]

    # 重建结点向量，确保周期性
    new_knotvector = np.hstack([[0], new_knotvector, [1]])
    print(f"theta_circle after update: {theta_circle}")

    # 使用优化后的极径更新控制点
    new_ctrlpts = np.column_stack((new_radii * np.cos(theta_circle), new_radii * np.sin(theta_circle)))
    print(f"new_ctrlpts: {new_ctrlpts}")
    new_ctrlpts_periodic = np.concatenate((new_ctrlpts, new_ctrlpts[:degree]), axis=0)
    new_weights_periodic = np.concatenate((new_weights, new_weights[:degree]))
    # 更新最终控制点的极角
    updated_theta_circle = np.arctan2(new_ctrlpts[:, 1], new_ctrlpts[:, 0])
    print(f"Updated theta_circle: {updated_theta_circle}")

    curve.ctrlpts = new_ctrlpts_periodic.tolist()
    curve.weights = new_weights_periodic.tolist()
    curve.knotvector = new_knotvector.tolist()

    # 如果需要绘制结果
    if plot_results:
        # 设置曲线的渲染参数
        curve.delta = 0.01
        curve.vis = VisMPL.VisCurve2D()

        # 渲染曲线
        #curve.render()

        # 绘制椭圆上的点
        plt.figure(figsize=(10, 10))
        plt.scatter(points[:, 0], points[:, 1], color='red', label='Input Points')

        # 将curve.evalpts转换为NumPy数组
        curve_evalpts = np.array(curve.evalpts)

        # 绘制拟合的曲线
        plt.plot(curve_evalpts[:, 0], curve_evalpts[:, 1], color='blue', label='Fitted Curve')

        # 获取控制点坐标
        ctrlpts = np.array(curve.ctrlpts)

        # 绘制控制点
        plt.scatter(ctrlpts[:, 0], ctrlpts[:, 1], color='green', marker='x', s=100, label='Control Points')

        # 连接控制点间的虚线
        plt.plot(ctrlpts[:-2, 0], ctrlpts[:-2, 1], color='green', linestyle='--', linewidth=1, alpha=0.7)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Input Points, Fitted NURBS Curve, and Control Points with Lines')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    return new_radii, new_weights_periodic, new_knotvector, np.array(curve.evalpts)

# 在这里调用函数
if __name__ == "__main__":
    # 示例：使用椭圆上的点
    # 定义椭圆的长轴和短轴
    a = 1.5  # 长轴
    b = 1.0  # 短轴

    # 计算椭圆上的点
    ellipse_angles = np.linspace(0, 2 * np.pi, 88, endpoint=False)
    x_ellipse = a * np.cos(ellipse_angles)
    y_ellipse = b * np.sin(ellipse_angles)
    ellipse_points = np.vstack((x_ellipse, y_ellipse)).T

    # 调用函数
    print(f"ellipse_points: {ellipse_points}")
    radii, weights, knotvector, curve_evalpts = fit_points_to_nurbs(ellipse_points, degree=3, num_ctrlpts=12, plot_results=True)