# RBE549: Sexy Semantics
# Tript Sharma
# Wrapper to perform semantic mapping

import open3d as o3d
import numpy as np
import os
import glob

DATA_DIR = 'KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/'

def parse_velo_scan(data_dir=DATA_DIR):
    '''
    Parse velodyne scan *.bin files to pcd using Open3D
    Input: relative path directory containing the .bin files
    Output: list containing open3d pcd  
    '''
    # parse point cloud
    absolute_dir = os.path.join(os.getcwd(),data_dir)
    pcl_paths = sorted(glob.glob(os.path.join(absolute_dir, "*.bin")))

    # Read field using numpy array
    pcl_data_np = [np.fromfile(pcl, dtype=np.float32).reshape((-1, 4))[:,:3] for pcl in pcl_paths[:10]]
    # print(pcl_data_np)

    # Convert to Open3D point cloud
    o3d_pcd = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)) for points in pcl_data_np]
    # print(o3d_pcd[0].points)

    return o3d_pcd

