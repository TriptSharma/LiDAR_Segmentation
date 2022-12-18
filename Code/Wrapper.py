from PointPainting import *
from SemanticSegmentation import get_img_paths, parse_img
from ICP import *
import argparse

CAM_2_VELO_CALIB_DATA_DIR = 'KITTI-360/KITTI-Small/2011_09_26/'
CAM_CALIB_DATA_DIR = 'KITTI-360/KITTI-Small/'
PCD_DATA_DIR = 'KITTI-360/KITTI-Small/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/'
RGB_DATA_DIR = 'KITTI-360/KITTI-Small/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/'
SAVE_DIR = 'results/'

argparser = argparse.ArgumentParser()
argparser.add_argument("--calib_data_path", help=".txt files folder", default=CALIB_DATA_DIR)
argparser.add_argument("--pcd_data_path", help=".bin files folder", default=PCD_DATA_DIR)
argparser.add_argument("--save_data_path",help="folder to save all the files i.e. segmented image, point cloud etc.", default=SAVE_DIR)

args = argparser.parse_args()

calib_dir= args.calib_data_path
pcd_dir = args.pcd_data_path
save_dir = args.save_data_path

SEGMENTED_DATA_DIR = save_dir+'segmented_image/'
COLORED_PCD_DATA_DIR = save_dir+'painted_cloud/'

projection_matrix = P_matrix_lidar_to_cam(calib_dir+'calib_velo_to_cam.txt')     # size (3,4)
# print(projection_matrix)

pcds = get_pcds_np(pcd_dir)

rgb_paths = get_img_paths(RGB_DATA_DIR)
seg_paths = get_img_paths(SEGMENTED_DATA_DIR)


for point_cloud, rgb_img_path, seg_img_path in zip(pcds, rgb_paths, seg_paths):
    img_name = rgb_img_path.split('\\')[-1]
    fused_img_filename = save_dir+'projected/'+img_name
    pcd_filename = save_dir+"painted_cloud/"+img_name.split('.')[0]+".pcd"

    rgb_img = parse_img(rgb_img_path)
    segmented_img = parse_img(seg_img_path)
        
    # segmented_img, semantic_labels = semantic_segment_rgb(model, rgb_img)
    pointPainting(projection_matrix, point_cloud, rgb_img, segmented_img, None, fused_img_filename, pcd_filename)

# print('All .pcds have been saved successfully\n')
# print('Performing ICP')



# pcds = get_pcds(COLORED_PCD_DATA_DIR,downsample=True)

# o3d.visualization.draw_geometries([pcds[0]])

# transformed_pcds = merge_pcd(pcds)

# #visualize all point clouds
# pcd_combined = o3d.geometry.PointCloud()
# for point_id in range(len(transformed_pcds)):
#     pcd_combined += transformed_pcds[point_id]

# o3d.visualization.draw_geometries([pcd_combined])