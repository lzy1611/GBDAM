import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline
from numpy.linalg import det
import get_3d_rigid_transform
from itertools import combinations
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from fit_and_project import *
from fit_points_to_bspline import *
from fit_points_to_nurbs import *


#folder_path = 'E:\\muti-rdb\\postprocessing dataset'
coord_tf_folder_path = 'G:\\mrdbdataset4\\job_coord_tf_csv'
axispoint_folder_path = 'G:\\mrdbdataset4\\axispoint_csv'
job_folder_path = 'G:\\mrdbdataset4\\job_csv'
output_folder_path = 'G:\\mrdbdataset4\\dataset0826'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

def axispoint_catch(coord_tf_folder_path,axispoint_folder_path,plot):
    for filename in os.listdir(coord_tf_folder_path):
        if filename.endswith('coordinates_tf.csv'):
            file_path = os.path.join(coord_tf_folder_path, filename)
            data_df = pd.read_csv(file_path)
            grouped_means = data_df.groupby('Z-initial')[['X-final_transformed', 'Y-final_transformed', 'Z-final_transformed' ]].mean().reset_index().sort_values(by='Z-initial', ascending=False)
            print(f"Grouped means for {filename}:")
            #print(grouped_means)

            output_filename = f"{os.path.splitext(filename)[0]}_axispoint.csv"
            output_file_path = os.path.join(axispoint_folder_path, output_filename)
            grouped_means.to_csv(output_file_path, index=False)
            print(f"Grouped means have been written to {output_filename}")

            if plot == True :
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')


                scatter = ax.scatter(grouped_means['X-final_transformed'], grouped_means['Y-final_transformed'], grouped_means['Z-final_transformed'],
                                     c=grouped_means['Z-initial'], cmap='viridis')
                ax.set_title(f'3D Scatter plot of final coordinates grouped by Z-initial ({filename})')
                ax.set_xlabel('X-final')
                ax.set_ylabel('Y-final')
                ax.set_zlabel('Z-final')


                ax.set_box_aspect([1, 1, 1])
                xmin,xmax=[-2000,2000]
                ymin,ymax=[-2000,2000]
                zmin,zmax=[-2000,2000]
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ ymin, ymax])
                ax.set_zlim([ zmin, zmax])
                cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
                cbar.set_label('Z-initial')

                plt.show()
                plt.close()



def calculate_vectors(x, y, z):
    vectors_prev = np.array([x[:-1], y[:-1], z[:-1]]) - np.array([x[1:], y[1:], z[1:]]) # 减去前一个点
    vectors_next = np.array([x[1:], y[1:], z[1:]]) - np.array([x[:-1], y[:-1], z[:-1]]) # 减去后一个点
    vectors_prev = vectors_prev.T
    vectors_next = vectors_next.T
    print(f'vectors_prev:{vectors_prev.shape}')
    return vectors_prev, vectors_next

def calculate_normals(vectors_prev, vectors_next):
    normals = np.cross(vectors_prev, vectors_next)
    return normals



def points2circle(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    num1 = len(p1)
    num2 = len(p2)
    num3 = len(p3)

    if (num1 == num2) and (num2 == num3):
        if num1 == 2:
            p1 = np.append(p1, 0)
            p2 = np.append(p2, 0)
            p3 = np.append(p3, 0)
        elif num1 != 3:
            print('\t仅支持二维或三维坐标输入')
            return None
    else:
        print('\t输入坐标的维数不一致')
        return None

    # 共线检查
    temp01 = p1 - p2
    temp02 = p3 - p2
    temp03 = np.cross(temp01, temp02)
    temp = (temp03 @ temp03) / (temp01 @ temp01) / (temp02 @ temp02)
    if temp < 10**-18:
        print('\t三点共线, 无法确定圆')
        return None

    temp1 = np.vstack((p1, p2, p3))
    temp2 = np.ones((3, 1))
    mat1 = np.hstack((temp1, temp2))  # size = 3x4

    m = +det(mat1[:, 1:])
    n = -det(np.delete(mat1, 1, axis=1))
    p = +det(np.delete(mat1, 2, axis=1))
    q = -det(temp1)

    temp3 = np.array([p1 @ p1, p2 @ p2, p3 @ p3]).reshape(3, 1)
    temp4 = np.hstack((temp3, mat1))
    temp5 = np.array([2 * q, -m, -n, -p, 0])
    mat2 = np.vstack((temp4, temp5))  # size = 4x5

    A = +det(mat2[:, 1:])
    B = -det(np.delete(mat2, 1, axis=1))
    C = +det(np.delete(mat2, 2, axis=1))
    D = -det(np.delete(mat2, 3, axis=1))
    E = +det(mat2[:, :-1])

    pc = -np.array([B, C, D]) / 2 / A
    r = np.sqrt(B * B + C * C + D * D - 4 * A * E) / 2 / abs(A)

    return pc, r



def visualize_points_with_labels(x, y, z, labels, intersection_points ,jobname):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')


    for i in range(len(x)):
        if labels[i] == 0:
            color = 'blue'
        elif labels[i] == 1:
            color = 'red'
        else:
            color = 'gray'
        ax.scatter(x[i], y[i], z[i], color=color)

    intersection_color = 'green'
    intersection_marker = 'x'


    for point in intersection_points:
        ax.scatter(point[0], point[1], point[2], color=intersection_color, marker=intersection_marker,s=200)

    ax.set_title(f'Points Visualization with Specific Colors and Intersection Points:{jobname}',)
    ax.set_xlabel('X-final')
    ax.set_ylabel('Y-final')
    ax.set_zlabel('Z-final')
    # 设置相同的比例和范围
    ax.set_box_aspect([1, 1, 1])  # 设置 x、y 和 z 轴的相同比例
    xmin, xmax = [-2000, 2000]
    ymin, ymax = [-2000, 2000]
    zmin, zmax = [-2000, 2000]
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    plt.show()


def moving_average_filter(data, window_size):
    """Apply a moving average filter to the data."""
    window_size = min(window_size, len(data))
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def median_filter(data, window_size):
    """
    Apply a median filter to the data.

    :param data: 1D numpy array, the input data to be filtered.
    :param window_size: int, the size of the filtering window.
    :return: 1D numpy array, the filtered data.
    """
    window_size = min(window_size, len(data))
    kernel = np.ones(window_size) / window_size
    # Use convolution with 'same' mode to keep the array size same as input
    # Note: scipy.signal.medfilt or np.median with appropriate indexing could be used for a more direct implementation
    padded_data = np.pad(data, (window_size // 2, window_size // 2), mode='reflect')
    convolved = np.convolve(padded_data, kernel, mode='same')
    # Since np.convolve doesn't directly apply median, we use reflection padding to mimic scipy's behavior
    return np.median(padded_data.reshape(-1, window_size), axis=1).flatten()


def gaussian_filter_function(data, sigma):
    """
    Apply a Gaussian filter to the data.

    :param data: 1D numpy array, the input data to be filtered.
    :param sigma: float, standard deviation for Gaussian kernel.
    :return: 1D numpy array, the filtered data.
    """
    # Scipy's gaussian_filter can directly apply Gaussian smoothing
    return gaussian_filter(data, sigma=sigma)


def filter_labels(labels, threshold):
    labels_list = labels.tolist()
    start = 0
    while start < len(labels_list):
        if labels_list[start] == 1:
            end = start

            while end < len(labels_list) and labels_list[end] == 1:
                end += 1

            if end - start < threshold:
                labels_list[start:end] = [0] * (end - start)
            start = end
        else:
            start += 1

    # 将修改后的列表转换回numpy数组
    return np.array(labels_list)

def closest_points_between_lines(line1, line2):
    # Convert points to numpy arrays for easier manipulation
    p1, p2 = np.array(line1)
    q1, q2 = np.array(line2)

    # Direction vectors for each line
    v1 = p2 - p1
    v2 = q2 - q1

    # Define matrices A and B for the equations
    A = np.vstack((v1, -v2)).T
    B = q1 - p1

    # Solve the equation A * [t, s] = B
    t, s = np.linalg.lstsq(A, B, rcond=None)[0]

    # Calculate closest points on each line
    closest_point_line1 = p1 + t * v1
    closest_point_line2 = q1 + s * v2

    # Calculate the distance
    distance = np.linalg.norm(closest_point_line1 - closest_point_line2, ord=2)

    intersection_point = (closest_point_line1 + closest_point_line2) / 2

    return intersection_point ,closest_point_line1, closest_point_line2, distance


def get_z_initial_values_at_equally_spaced_points(curved_segments, z_initial, n):
    indices = []
    points_z_initial = []
    for start, end in curved_segments:
        step = (end - start) / n
        for i in range(n):
            index = int(start + step * i)
            if 0 <= index < len(z_initial):
                indices.append(index)
                points_z_initial.append(z_initial[index])
    return points_z_initial, indices

def get_tail_points(last_start_point, last_end_point,tail_length):
    direction_vector = last_end_point - last_start_point
    unit_direction = direction_vector / np.linalg.norm(direction_vector)
    tail_point = last_start_point + unit_direction * tail_length

    return tail_point

def extract_matching_rows(csv_filename, z_initial_values, columns, ):
    tolerance = 1e-5
    extracted_data = []

    with open(csv_filename, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            z_value = float(row['Z-initial'])
            for z_initial in z_initial_values:
                if abs(z_value - z_initial) <= tolerance:
                    matching_data = {column: row[column] for column in columns}
                    extracted_data.append(matching_data)
                    break

    return extracted_data

def find_corresponding_parameter_file(base_folder, jobname, suffix='_parameters.csv'):
    target_filename = f"{jobname}{suffix}"
    for root, dirs, files in os.walk(base_folder):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None


def reshape_dict_values(values_list, target_rows):
    num_items = len(values_list)
    items_per_row = num_items // target_rows
    remainder = num_items % target_rows
    reshaped_data = np.zeros((target_rows, items_per_row))
    for i in range(target_rows):
        start_idx = i * items_per_row
        end_idx = start_idx + items_per_row
        reshaped_data[i] = values_list[start_idx:end_idx]
    return reshaped_data

axispoint_catch(coord_tf_folder_path, axispoint_folder_path, plot=False)
for filename in os.listdir(axispoint_folder_path):
    if filename.endswith('tf_axispoint.csv'):
        file_path = os.path.join(axispoint_folder_path, filename)
        print(filename)
        jobname = filename.split('_merged_coordinates_tf_axispoint.csv')[0]
        print("Job Name:", jobname)
        grouped_means = pd.read_csv(file_path)
        corresponding_parameter_csv_path = find_corresponding_parameter_file(job_folder_path, jobname)
        if corresponding_parameter_csv_path:
            print(f"Found corresponding CSV at: {corresponding_parameter_csv_path}")
            job_parameters = pd.read_csv(corresponding_parameter_csv_path,)
            print('job_parameters:', job_parameters)
        else:
            print(f"No corresponding CSV found for job {jobname}")

        z_initial = grouped_means['Z-initial'].values
        x_final = grouped_means['X-final_transformed'].values
        y_final = grouped_means['Y-final_transformed'].values
        z_final = grouped_means['Z-final_transformed'].values
        #print(f'x_final: {x_final.shape}')

        curvature_threshold = 350
        radii = np.array([])
        labels = np.array([])

        for i in range(len(x_final) - 2):
            p1 = np.array([x_final[i], y_final[i], z_final[i]])
            p2 = np.array([x_final[i+1], y_final[i+1], z_final[i+1]])
            p3 = np.array([x_final[i + 2], y_final[i + 2], z_final[i + 2]])

            circle_center, circle_radius = points2circle(p1, p2, p3)


            label = 1 if circle_radius < curvature_threshold else 0
            radii = np.append(radii, circle_radius)
            labels = np.append(labels, label)

        radii = gaussian_filter_function(radii, 2)

        for i, circle_radius in enumerate(radii):
            labels[i] = 1 if circle_radius < curvature_threshold else 0

        labels = np.insert(labels, 0, 0)
        labels = np.append(labels, 0)
        labels = filter_labels(labels,threshold=15)
        line_segments = []
        straight_line_start = None
        straight_segments = []
        curved_segments = []

        in_straight_segment = False
        current_start_index = None
        for i in range(len(labels)):
            if labels[i] == 0:
                if not in_straight_segment:
                    in_straight_segment = True
                    current_start_index = i
            else:
                if in_straight_segment:
                    straight_segments.append((current_start_index, i - 1))
                    in_straight_segment = False

        if in_straight_segment:
            straight_segments.append((current_start_index, len(labels) - 1))

        if len(straight_segments) > 1:
            for i in range(len(straight_segments) - 1):
                start_index = straight_segments[i][1] + 1
                end_index = straight_segments[i + 1][0] - 1
                curved_segments.append((start_index, end_index))

            if straight_segments[-1][1] < len(labels) - 1:
                start_index = straight_segments[-1][1] + 1
                end_index = len(labels) - 1
                curved_segments.append((start_index, end_index))

        intersection_points = []
        for i in range(1, len(straight_segments)):
            start1, end1 = straight_segments[i - 1]
            start2, end2 = straight_segments[i]
            line1_points = [(x_final[start1], y_final[start1], z_final[start1]), ((x_final[end1], y_final[end1], z_final[end1]))]
            line2_points = [(x_final[start2], y_final[start2], z_final[start2]), ((x_final[end2], y_final[end2], z_final[end2]))]
            intersection_point, _, _, _ = closest_points_between_lines(line1_points,line2_points)
            intersection_points.append(intersection_point)
        last_start_point = np.array(
            [x_final[straight_segments[-1][0]], y_final[straight_segments[-1][0]], z_final[straight_segments[-1][0]]])
        last_end_point = np.array(
            [x_final[straight_segments[-1][1]], y_final[straight_segments[-1][1]], z_final[straight_segments[-1][1]]])

        tail_point = get_tail_points(last_start_point,last_end_point,tail_length=job_parameters.iloc[-1,1])

        intersection_points.append(tail_point)
        zero_row = np.zeros(3)
        intersection_points = np.insert(intersection_points, 0, zero_row, axis=0)
        print('intersection_points:', intersection_points)
        print('shape intersection:', intersection_points.shape)
        if intersection_points.shape[0] == job_parameters.shape[0]:
            curved_segments_num = 6
            z_initial_values,indices= get_z_initial_values_at_equally_spaced_points(curved_segments, z_initial, curved_segments_num)
            csv_filename = filename.replace("_axispoint", "")
            csv_file_path = os.path.join(coord_tf_folder_path, csv_filename)
            columns = ['Outside-Label','Z-initial', 'X-final_transformed', 'Y-final_transformed', 'Z-final_transformed']
            crosssection_data = extract_matching_rows(csv_file_path, z_initial_values, columns)
            known_thickness = job_parameters.iloc[-1, 5]*job_parameters.iloc[-1,14]
            known_outer_diameter = job_parameters.iloc[-1, 5]
            print('known_outer_diameter:', known_outer_diameter)

            grouped_by_z_initial = {}

            for item in crosssection_data:
                z_initial = float(item['Z-initial'])
                label = str(int(float(item['Outside-Label'])))

                if z_initial not in grouped_by_z_initial:
                    grouped_by_z_initial[z_initial] = {'0': [], '1': []}

                grouped_by_z_initial[z_initial][label].append((float(item['X-final_transformed']),
                                                               float(item['Y-final_transformed']),
                                                               float(item['Z-final_transformed'])))

            projected_points_by_z_initial = {}
            forward_distance = 20
            backward_distance = 20

            job_parameters_rows = job_parameters.shape[0] - 2
            all_inner_radii = []
            all_outer_radii = []


            for z_initial, label_groups in grouped_by_z_initial.items():
                inner_points = label_groups.get('0', [])
                print(f"Inner points for z_initial {z_initial}: {inner_points}")
                outer_points = label_groups.get('1', [])
                all_points = inner_points + outer_points
                matching_rows = grouped_means[grouped_means['Z-initial'] == z_initial]

                if not matching_rows.empty:
                    axis_point = matching_rows.iloc[0][[
                        'X-final_transformed', 'Y-final_transformed', 'Z-final_transformed']].values
                else:
                    closest_index = np.argmin(np.abs(grouped_means['Z-initial'] - z_initial))
                    closest_z_initial = grouped_means.iloc[closest_index]['Z-initial']
                    axis_point = grouped_means.iloc[closest_index][[
                        'X-final_transformed', 'Y-final_transformed', 'Z-final_transformed']].values

                    print(f"No exact match for z_initial {z_initial}. Using closest match {closest_z_initial} instead.")

                mask_next = (grouped_means['Z-initial'] > z_initial)
                next_axis_points = grouped_means[mask_next]

                if not next_axis_points.empty:
                    distances = np.abs(next_axis_points['Z-initial'] - (z_initial + forward_distance))
                    min_distance_index = distances.idxmin()
                    next_axis_point = next_axis_points.loc[min_distance_index][
                        ['X-final_transformed', 'Y-final_transformed', 'Z-final_transformed']].values
                else:
                    next_axis_point = None

                mask_prev = (grouped_means['Z-initial'] < z_initial)
                prev_axis_points = grouped_means[mask_prev]

                if not prev_axis_points.empty:
                    distances = np.abs(prev_axis_points['Z-initial'] - (z_initial - backward_distance))
                    min_distance_index = distances.idxmin()
                    prev_axis_point = prev_axis_points.loc[min_distance_index][
                        ['X-final_transformed', 'Y-final_transformed', 'Z-final_transformed']].values
                else:
                    prev_axis_point = None


                forward_direction_vector = next_axis_point - axis_point if next_axis_point is not None else None
                backward_direction_vector = axis_point - prev_axis_point if prev_axis_point is not None else None


                if forward_direction_vector is not None:
                    forward_direction_vector = forward_direction_vector / np.linalg.norm(forward_direction_vector)
                if backward_direction_vector is not None:
                    backward_direction_vector = backward_direction_vector / np.linalg.norm(backward_direction_vector)

                print(f"Forward direction vector for z_initial {z_initial}: {forward_direction_vector}")
                print(f"Backward direction vector for z_initial {z_initial}: {backward_direction_vector}")


                if forward_direction_vector is not None and backward_direction_vector is not None:
                    bending_normal_vector = np.cross(backward_direction_vector,forward_direction_vector)
                    bending_normal_vector = bending_normal_vector / np.linalg.norm(bending_normal_vector)
                    print(f"Bending normal vector for z_initial {z_initial}: {bending_normal_vector}")

                normal_vector, _ = best_fit_plane(all_points, forward_direction_vector, axis_point)#投影
                print(f"Normal vector for z_initial {z_initial}: {normal_vector}")
                radial_vector = np.cross(normal_vector, bending_normal_vector)
                radial_vector = radial_vector / np.linalg.norm(radial_vector)
                print(f"Radial vector for z_initial {z_initial}: {radial_vector}")
                x_axis_vector = np.cross(radial_vector, normal_vector)
                x_axis_vector = x_axis_vector / np.linalg.norm(x_axis_vector)
                print(f"x_axis_vector for z_initial {z_initial}: {x_axis_vector}")
                def check_orthogonality(v1, v2, tolerance=1e-6):
                    dot_product = np.dot(v1, v2)
                    return abs(dot_product) < tolerance

                inner_projected_points = project_to_plane(inner_points, axis_point, x_axis_vector, radial_vector)
                outer_projected_points = project_to_plane(outer_points, axis_point, x_axis_vector, radial_vector)

                sorted_inner_points = sort_points_by_angle(inner_projected_points)
                sorted_outer_points = sort_points_by_angle(outer_projected_points)
                sorted_inner_points_array = np.array(sorted_inner_points)
                sorted_outer_points_array = np.array(sorted_outer_points)
                print(f"Sorted inner points for z_initial {z_initial}type: {type(sorted_inner_points)}")
                inner_curve_radii, inner_curve_weights, inner_curve_knotvector, inner_curve_evalpts = fit_points_to_nurbs(sorted_inner_points_array, degree=3, num_ctrlpts=12, plot_results=False)
                outer_curve_radii, outer_curve_weights, outer_curve_knotvector, outer_curve_evalpts = fit_points_to_nurbs(sorted_outer_points_array, degree=3, num_ctrlpts=12, plot_results=False)

                if np.array_equal(inner_curve_evalpts, outer_curve_evalpts)==True:
                    print("inner_curve_evalpts and outer_curve_evalpts are the same.")

                plt.figure(figsize=(10, 10))

                plt.plot(inner_curve_evalpts[:, 0], inner_curve_evalpts[:, 1], color='blue', label='Inner Curve')

                plt.plot(outer_curve_evalpts[:, 0], outer_curve_evalpts[:, 1], color='red', label='Outer Curve')



                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title(f'NURBS Curves for z_initial = {z_initial}')
                plt.legend()
                plt.grid(True)
                plt.axis('equal')
                plt.show()

                all_inner_radii.append(inner_curve_radii)
                all_outer_radii.append(outer_curve_radii)

            job_parameters_rows = job_parameters.shape[0] - 2
            all_inner_radii_reshaped = np.array(all_inner_radii).reshape(job_parameters_rows, -1)
            all_outer_radii_reshaped = np.array(all_outer_radii).reshape(job_parameters_rows, -1)

            print(f"all_inner_radii_reshaped:, {all_inner_radii_reshaped}")

            num_columns = intersection_points.shape[1]

            print('type of intersection_points:',type(intersection_points))
            intersection_points_df = pd.DataFrame(intersection_points)
            print(intersection_points_df)
            new_columns_df = pd.DataFrame()

            for i in range(1, num_columns + 1):
                column_name = f'intersection_point_{i}'
                data_for_column = intersection_points_df.iloc[:, i - 1].tolist()
                new_columns_df[column_name] = data_for_column

            job_parameters = pd.concat([job_parameters, new_columns_df], axis=1)

            items_per_file = curved_segments_num

            def split_list_by_file_count(lst, items_per_file):
                chunks = [lst[i:i + items_per_file] for i in range(0, len(lst), items_per_file)]
                if len(chunks[-1]) < items_per_file:
                    chunks.append(lst[-items_per_file:])
                return chunks


            all_inner_radii_splits = split_list_by_file_count(all_inner_radii, items_per_file)
            all_outer_radii_splits = split_list_by_file_count(all_outer_radii, items_per_file)
            print(f"all_inner_radii_splits:, {all_inner_radii_splits}")
            bending_id = 1
            num_radii = 12
            inner_columns = [f'inner_curve_radii_{i}' for i in range(num_radii)]
            outer_columns = [f'outer_curve_radii_{i}' for i in range(num_radii)]
            for inner_chunk, outer_chunk in zip(all_inner_radii_splits, all_outer_radii_splits):
                rows = []
                for inner_row, outer_row in zip(inner_chunk, outer_chunk):
                    row = {**dict(zip(inner_columns, inner_row)), **dict(zip(outer_columns, outer_row))}
                    rows.append(row)

                df = pd.DataFrame(rows)

                print('df:')
                print(df)

                crosssection_output_folder_path = os.path.join(output_folder_path, 'cross_section')
                output_file_name = f'{jobname}-bending{bending_id}.csv'
                output_csv_path = os.path.join(crosssection_output_folder_path, output_file_name)

                df.to_csv(output_csv_path, index=False)
                bending_id += 1

            column_names = inner_columns+outer_columns
            data_length = curved_segments_num 
            data = {name: [0.0] * data_length for name in column_names}
            df = pd.DataFrame(data)
            print('df:')
            print(df)
            crosssection_output_folder_path = os.path.join(output_folder_path, 'cross_section')
            output_file_name_first = f'{jobname}-bending0.csv'
            output_csv_path_first = os.path.join(crosssection_output_folder_path, output_file_name_first)
            df.to_csv(output_csv_path_first, index=False)

            last_bending_id = bending_id
            output_file_name_last = f'{jobname}-bending{last_bending_id}.csv'
            output_csv_path_last = os.path.join(crosssection_output_folder_path, output_file_name_last)
            df.to_csv(output_csv_path_last, index=False)
            print(f'Output path for jobx-bending{last_bending_id}.csv: {output_csv_path_last}')
            print('job_parameters_with_new_columns:', job_parameters)
            job_output_folder_path = os.path.join(output_folder_path, 'job')
            output_csv_name = os.path.join(job_output_folder_path, jobname+'.csv')
            job_parameters.to_csv(output_csv_name, index=False)

