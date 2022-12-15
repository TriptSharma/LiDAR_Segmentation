# RBE549: Sexy Semantics
# Tript Sharma
# Wrapper to perform semantic mapping

import open3d as o3d
import numpy as np
import os
import glob

DATA_DIR = 'KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/'

def parse_velo_scans(data_dir=DATA_DIR):
    '''
    Parse velodyne scan *.bin files to pcd using Open3D
    Input: relative path directory containing the .bin files
    Output: list containing open3d pcd  
    '''
    # parse point cloud
    absolute_dir = os.path.join(os.getcwd(),data_dir)
    pcl_paths = sorted(glob.glob(os.path.join(absolute_dir, "*.bin")))

    # Read field using numpy array
    # TODO: parse all files instead of only first 1000
    pcl_data_np = [np.fromfile(pcl, dtype=np.float32).reshape((-1, 4))[:,:3] for pcl in pcl_paths[:100]]
    # print(pcl_data_np)

    # Convert to Open3D point cloud
    #! o3d takes array of size (N,3)
    o3d_pcd = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)) for points in pcl_data_np]
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
    max_correspondence_distance = 0.5
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=0.000001,
            relative_rmse=0.000001,
            max_iteration=50
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

def merge_pcd(pcds):
    transformed_pcds = [pcds[0]]
    
    for idx in range(1,len(pcds)):
        optimized_T = pairwise_registration(source=pcds[idx], target=pcds[idx-1])
        transformed_pcds.append(pcds[idx].transform(optimized_T))

    return transformed_pcds

pcds = parse_velo_scans()
pcds = downsample_pcds(pcds, 0.1)

# o3d.visualization.draw_geometries(pcds)
transformed_pcds = merge_pcd(pcds)

#visualize all point clouds
pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(transformed_pcds)):
    pcd_combined += transformed_pcds[point_id]

o3d.visualization.draw_geometries(pcd_combined)