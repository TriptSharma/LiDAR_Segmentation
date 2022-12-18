# RBE549: Sexy Semantics
# Tript Sharma
# Wrapper to perform semantic mapping
import open3d as o3d
import numpy as np
import os
import glob
import time


def parse_velo_scans(data_dir):
    '''
    Parse velodyne scan *.bin files to pcd using Open3D
    Input: relative path directory containing the .bin files
    Output: list containing open3d pcd  
    '''
    # parse point cloud
    absolute_dir = os.path.join(os.getcwd(),data_dir)
    pcd_paths = sorted(glob.glob(os.path.join(absolute_dir, "*.bin")))

    # Read field using numpy array
    # TODO: parse all files instead of only first 1000
    pcd_data_np = [np.fromfile(pcd, dtype=np.float32).reshape((-1, 4))[:,:3] for pcd in pcd_paths]
    # print(pcd_data_np)

    return pcd_data_np

def np_to_o3d(np_pcd):
    # Convert to Open3D point cloud
    #! o3d takes array of size (N,3)
    o3d_pcd = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)) for points in np_pcd]
    # print(o3d_pcd[0].points)
    return o3d_pcd

def downsample_pcds(pcd_list, voxel_size=0.05): 
    '''
    Downsample a list of pcds to procecss them faster and store less data'''
    downsampled_pcd_list = [pcd.voxel_down_sample(voxel_size=voxel_size) for pcd in pcd_list]
    return downsampled_pcd_list

def pairwise_registration(source, target):
    trans_init = np.eye(4)
    #! between 1x-3x of the voxel size
    max_correspondence_distance = 0.25
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=0.00001,
            relative_rmse=0.00001,
            max_iteration=100
        )

    # def callback(output):
    #     print("Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
    #         output["iteration_index"].item(),
    #         output["fitness"].item(),
    #         output["inlier_rmse"].item()))

    transformed_pcd = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance, 
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria,
        # callback
    )
    return transformed_pcd.transformation

def pairwise_multistage_registration(source, target):

    device = o3d.core.Device("CUDA:0")
    dtype = o3d.core.float32

    # Create an empty point cloud
    # Use pcd.point to access the points' attributes
    pcd = o3d.t.geometry.PointCloud(device)


    trans_init = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)
    voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])
    #! between 1x-3x of the voxel size
    max_correspondence_distance = o3d.utility.DoubleVector([0.3, 0.14, 0.07])
    criteria_list = [
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001,
                                    relative_rmse=0.0001,
                                    max_iteration=20),
        o3d.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 15),
        o3d.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 10)
    ]

    def callback(output):
        print("Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
            output["iteration_index"].item(),
            output["fitness"].item(),
            output["inlier_rmse"].item()))
    start = time.time()

    transformed_pcd = o3d.t.pipelines.registration.multi_scale_icp(
        source,
        target,
        voxel_sizes,
        criteria_list,
        max_correspondence_distance, 
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        callback,
    )

    ms_icp_time = time.time() - start
    print("Time taken by Multi-Scale ICP: ", ms_icp_time)
    print("Inlier Fitness: ", transformed_pcd.fitness)
    print("Inlier RMSE: ", transformed_pcd.inlier_rmse)
    return transformed_pcd.transformation



def merge_pcd(pcds):
    transformed_pcds = [pcds[0]]
    transformation_current_frame = np.eye(4)
    for idx in range(1,len(pcds),10):
        optimized_T = pairwise_registration(source=pcds[idx], target=pcds[idx-1])
        transformation_current_frame = transformation_current_frame @ optimized_T
        transformed_pcds.append(pcds[idx].transform(transformation_current_frame))

    return transformed_pcds

def get_pcds_np(dir, downsample=False):
    pcds = parse_velo_scans(dir)
    if downsample:
        pcds = downsample_pcds(pcds, 0.1)
    return pcds


def get_pcds(dir, downsample=False):

    absolute_dir = os.path.join(os.getcwd(),dir)
    pcd_paths = sorted(glob.glob(os.path.join(absolute_dir, "*.pcd")))

    pcds = [o3d.io.read_point_cloud(pcd_path) for pcd_path in pcd_paths]

    if downsample:
        pcds = downsample_pcds(pcds, 0.1)
    return pcds


if __name__=='__main__':
    # DATA_DIR = 'KITTI-360/KITTI/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/'
    # PCD_DATA_DIR = 'KITTI-360/KITTI-Small/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/'
    PCD_DATA_DIR = 'results/painted_cloud/'
    pcds = get_pcds(PCD_DATA_DIR,downsample=True)

    o3d.visualization.draw_geometries([pcds[0]])

    transformed_pcds = merge_pcd(pcds)

    #visualize all point clouds
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(transformed_pcds)):
        pcd_combined += transformed_pcds[point_id]

    o3d.visualization.draw_geometries([pcd_combined])