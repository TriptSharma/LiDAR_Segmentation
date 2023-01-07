# RBE549: Sexy Semantics
# Tript Sharma
# Merge Pointclouds using ICP
import numpy as np
import open3d as o3d 
import cv2

def cam_calib(filename):
    pass

def calibration_data(filename):
    '''
    Import and parse calibration data including:
        P: Projection matrix from cam to lidar
        K: camera intrinsic matrix (3x3)
        R0: rectified rotation matrix (3x3)
        Tr_cam_to_lidar: camera to lidar transformation matrix (3x4)
        D: cam distortion values
    Inputs:
        filename: Flie containing data in KITTI dataset calibration file
        '''
    text = ''
    with open(filename,'r') as f:
        text= f.read()
        # if counter==1:
        # elif counter==2:
        #     t= f.readline()

    calibration_data = {}
    for line in text.split('\n'):
        key, val = line.split(':')
        calibration_data[key] = val.lstrip().split()
    
    # t = text.split('\n')[2].split(':')[-1].split()

    # R = np.array(R, dtype=np.float32).reshape((3,3))
    # t = np.array(t, dtype=np.float32).reshape((3,1))

    return calibration_data

def P_matrix_lidar_to_cam(calibration_data):
    '''
    Returns projection matrix for projection of lidar to cam. 
    To convert points from Lidar to cam frame we will first transform the points to cam frame using P^-1 and R^-1
    Then project them in 2D space using P^-1

    NOTE: KITTI has a ref cam (i.e. cam 0), first project lidar to ref cam then ref cam to cam

    i.e. P_lidar2ref = Tr inverse @ Rect matrix (ref cam)
         P_ref2cam = P @ Rect matrix @ P_lidar2ref

    where,
        P = projection matrix from ref cam to current cam (3,4)
        R = rectification matrix for cam 0 (3,3)
        Tr = transformation between ref cam to lidar (3,4)
    Input: 
        calibration_data: dict containing all calib info
    Output:
        P_lidar2cam = (3,4)
    '''
    P =  np.array(calibration_data['P'], dtype=np.float32).reshape(3,4)
    R0 = np.array(calibration_data['R0'], dtype=np.float32).reshape(3,3)    #Rectificattion matrix for cam 0 (ref cam)
    Tr = np.array(calibration_data['Tr_cam_to_lidar'], dtype=np.float32).reshape(3,4)

    R0_expanded = np.eye(4)
    R0_expanded[:3,:3] = R0
    R_l2ref = np.linalg.inv(R0_expanded)

    Tr_ref2lidar = np.vstack([(Tr), [0,0,0,1]])
    Tr_lidar2ref = np.linalg.inv(Tr_ref2lidar)

    return P @ R0_expanded @ Tr_lidar2ref

def project_lid_to_cam(P,lidar_pts):
    """
        Projecting 3D lidar (Velo) points on the image
        Inputs:
            Projection_matrix: lidar to camera projection matrix (Size = [3,4])
            lidar_pts: lidar points array (size = [npoints,3])
        Outputs:
            projected lidar points on image(2D points)[npoints,2] and projected depth [npoints,1]
    """
    
    #homogenize lidar pts
    pts_3d = np.hstack((lidar_pts, np.ones((lidar_pts.shape[0], 1))))
    # project lidar points to image plane
    pts_2d= np.dot(pts_3d,P.T)
    
    #depth values = last col of the the projected points 
    depth = pts_2d[:, 2]
    # handle division by zero cases
    depth[depth==0] = -1e-6

    #get homogenized image point
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    
    pts_2d = pts_2d[:, :2]

    return pts_2d,depth

def remove_lidar_points_beyond_img(pts_2d, xmin, ymin, xmax, ymax):
    """
        @brief      Filter lidar points, keep only those which lie inside image
        @param      P lidar to camera projection matrix[3,4]
        @param      lidar_pts [npoints,3]
        @param      xmin minimum image size width
        @param      ymin minimum image size height
        @param      xmax maximum image size width
        @param      ymax maximum image size height
        @return     points on image(2D points)[npoints,2], list of indices, projected depth [npoints,1]
    """
  
    inside_pts_indices = ((pts_2d[:, 0] >= xmin) & (pts_2d[:, 0] < xmax) & (pts_2d[:, 1] >= ymin) & (pts_2d[:, 1] < ymax))
   
    return  pts_2d, inside_pts_indices


def project_lidar_on_image(P, lidar_pts, size):
    """
        @brief      Projecting 3D lidar points on the image
        @param      P lidar to camera projection matrix[3,4]
        @param      lidar_pts [npoints,3]
        @param      size: image size
        @return     filtered points on image(2D points)[npoints,2] and  projected depth [npoints,1]
    """
    pts_2d,depth = project_lid_to_cam(P,lidar_pts)
    all_pts_2d, fov_inds = remove_lidar_points_beyond_img(pts_2d, 0, 0,size[0], size[1])

    return all_pts_2d[fov_inds],depth[fov_inds], lidar_pts[fov_inds]


def pointPainting(projection_matrix_lidar_to_cam, point_cloud, rgb_img, segmented_img, semantic_labels, fused_img_filename, pcd_filename):
    # print(semantic_labels.shape)

    fused_img = rgb_img.copy()

    #3d points infront of camera will only be projected
    idx = point_cloud[:,0] >= 0
    point_cloud = point_cloud[idx]

    pts_2D, depth, pts_3D_img = project_lidar_on_image(
        projection_matrix_lidar_to_cam, 
        point_cloud, 
        (rgb_img.shape[1], rgb_img.shape[0])
    )

    #Number of lidar points projected on image
    N = pts_3D_img.shape[0]

    #Creating semantic channel for point cloud
    semantic_color = np.zeros((N,3), dtype=np.float64)
    
    for i in range(pts_2D.shape[0]):
        x = np.int32(pts_2D[i, 0])
        y = np.int32(pts_2D[i, 1])

        semantic_color[i] = segmented_img[y,x]/255
        
        pt = (x,y)
        cv2.circle(fused_img, pt, 5, color=tuple(map(int, segmented_img[y,x])), thickness=-1)        
    stacked_img = np.vstack((fused_img, segmented_img, rgb_img))

    # cv2.imshow('fuse', stacked_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite(fused_img_filename,stacked_img)

    painted_pointcloud = np.hstack((pts_3D_img[:,:3], semantic_color))

    visuallize_pointcloud(painted_pointcloud, pcd_filename)


def visuallize_pointcloud(pointcloud, filename):
        """
        @brief      Visualizing colored point cloud
        @param      pointcloud  in lidar coordinate [npoints, 6] in format of [X Y Z R G B]
        @return     None
        """
        
        # Get semantic colors from pointcloud
        colors  = pointcloud[:, 3:]
        #Get xyz values from pointcloud
        xyz = pointcloud[:, 0:3]


        #Initialize Open3D visualizer
        visualizer = o3d.visualization.Visualizer()
        pcd = o3d.geometry.PointCloud()
        visualizer.add_geometry(pcd)


        # Get colors of each point according to cityscapes labels
        # colors = semantics_to_colors(semantics,palette)
    
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(filename,pcd)

