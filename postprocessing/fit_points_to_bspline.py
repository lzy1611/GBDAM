import numpy as np
from scipy.optimize import least_squares
from geomdl import NURBS
from geomdl.visualization import VisMPL
import matplotlib.pyplot as plt
from geomdl import utilities
from scipy.spatial.distance import cdist

def fit_points_to_bspline(points, num_ctrlpts,degree=3, plot_results=True):
    print("调用fit_points_to_nurbs")
    # 定义输入参数
    num_points = len(points)

    # 生成角度从0到2π
    angles = np.linspace(0, 2 * np.pi, num_ctrlpts, endpoint=False)

    # 创建控制点（圆形）
    x_circle = np.cos(angles)
    y_circle = np.sin(angles)

    # 创建控制点（圆形）
    control_points = np.vstack((x_circle, y_circle)).T
    print("control_points:", control_points)

    # 为每个控制点分配相同的权重（对于圆，权重为1）
    weights = np.ones(len(control_points))

    # 由于我们需要周期性曲线，需要复制前degree个控制点
    control_points_periodic = np.concatenate((control_points, control_points[:degree]), axis=0)
    print("control_points_periodic:", control_points_periodic)
    weights_periodic = np.concatenate((weights, weights[:degree]))

    # 创建NURBS曲线
    curve = NURBS.Curve()

    # 设置曲线的阶数
    curve.degree = degree

    # 设置控制点和权重
    curve.ctrlpts = control_points_periodic.tolist()
    curve.weights = weights_periodic.tolist()

    # 设置周期性的结点向量
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(control_points_periodic), clamped=False)
    print("curve.knotvector:", curve.knotvector)

    # 将控制点从直角坐标转换为极坐标
    r_circle, theta_circle = np.abs(np.linalg.norm(control_points, axis=1)), np.arctan2(control_points[:, 1], control_points[:, 0])

    # 定义残差函数
    def residual(radii):
        # 构建新的控制点，使用优化后的极径
        new_ctrlpts = np.column_stack((radii * np.cos(theta_circle), radii * np.sin(theta_circle)))

        # 保留控制点的角度不变
        new_ctrlpts_periodic = np.concatenate((new_ctrlpts, new_ctrlpts[:degree]), axis=0)
        new_weights_periodic = np.concatenate((weights, weights[:degree]))

        curve.ctrlpts = new_ctrlpts_periodic.tolist()
        curve.weights = new_weights_periodic.tolist()

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

    # 拼接极径作为初始参数
    initial_radii = r_circle

    # 进行最小二乘法优化
    result = least_squares(residual, initial_radii, method='trf', loss='huber', max_nfev=1000, verbose=2, ftol=1e-15, xtol=1e-15)

    # 使用优化后的结果更新曲线
    new_radii = result.x
    new_ctrlpts = np.column_stack((new_radii * np.cos(theta_circle), new_radii * np.sin(theta_circle)))
    new_ctrlpts_periodic = np.concatenate((new_ctrlpts, new_ctrlpts[:degree]), axis=0)
    new_weights_periodic = np.concatenate((weights, weights[:degree]))
    curve.ctrlpts = new_ctrlpts_periodic.tolist()
    curve.weights = new_weights_periodic.tolist()
    curve_evalpts = np.array(curve.evalpts)
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
    #print("Returning:", new_radii, new_weights_periodic, curve.knotvector, curve_evalpts)
    # 返回控制点的极径、权重和结点向量
    return new_radii, new_weights_periodic, curve.knotvector, curve_evalpts

# 在这里调用函数
if __name__ == "__main__":
    # 示例：使用椭圆上的点
    # 定义椭圆的长轴和短轴
    a = 48  # 长轴
    b = 46  # 短轴

    # 计算椭圆上的点
    ellipse_angles = np.linspace(0, 2 * np.pi, 88, endpoint=False)
    x_ellipse = a * np.cos(ellipse_angles)
    y_ellipse = b * np.sin(ellipse_angles)
    ellipse_points = np.vstack((x_ellipse, y_ellipse)).T

    # 调用函数
    print(f"ellipse_points: {ellipse_points}")
    radii, weights, knotvector, curve_evalpts = fit_points_to_bspline(ellipse_points, degree=3, num_ctrlpts=12, plot_results=True)
    print(f"radii: {radii}")