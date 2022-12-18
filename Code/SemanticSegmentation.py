# RBE549: Sexy Semantics
# Tript Sharma
# Perform semantic segmentation on RGB images
import glob
import os
import torch
import cv2
from torchvision import transforms
import numpy as np



def get_color_from_pallete():    
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    
    return colors

def parse_img(path):
    '''
    Parse stereocam's left cam's RGB frames
    Input: relative path directory containing the .png files
    Output: list containing open3d pcd  
    '''
    # read img
    frame = cv2.imread(path)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return rgb

def semantic_segment_rgb(model, rgb_frame):
    '''
    get semantic labels and the corresponding colors for each pixel in the RGB image
    Inputs:
        model: torch model for semantic segmentation
        img: RGB image
    Outputs:
        Semantic class labels for each point, size = (H,W)
        Colored points for each of the points, size = (H, W, 3)
    '''

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    input_tensor = preprocess(rgb_frame)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    output_predictions = output.argmax(0)   # prediction shape = (H,W)

    #color img using semantic labels
    colors = get_color_from_pallete()
    
    # plot the semantic segmentation predictions of 21 classes in each color
    semantic_labels = output_predictions.reshape(-1) # size = (H*W,1)
    semantic_segmented_img = np.array([colors[semantic_labels[i]] for i in range(semantic_labels.shape[0])])    \
                                .reshape(rgb_frame.shape)

    #torch tensor to np array
    semantic_labels = semantic_labels.reshape(rgb_frame.shape[:2]).cpu().detach().numpy()
    return semantic_segmented_img, semantic_labels


def show_img(img):
    cv2.imshow('semantic img', img)

def save_img(filename, img):
    cv2.imwrite(filename, img)

def get_img_paths(rgb_data_dir):
    absolute_dir = os.path.join(os.getcwd(),rgb_data_dir)
    img_paths = sorted(glob.glob(os.path.join(absolute_dir, "*.png")))
    return img_paths

if __name__ == '__main__':
    isVis=False
    isSave = True
    RGB_DATA_DIR = 'KITTI-360/KITTI-Small/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/'
    SAVE_DIR = 'KITTI-360/KITTI-Small/2011_09_26/2011_09_26_drive_0001_sync/image_02/segmented_image_02/'

    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    # model = torchvision.models.segmentation.fcn_resnet101()
    model.eval()

    img_paths = get_img_paths(RGB_DATA_DIR)

    for path in img_paths:
        rgb_frame = parse_img(path)
        semantic_img, _ = semantic_segment_rgb(model, rgb_frame)

        if isVis:
            show_img(semantic_img)
        if isSave:
            save_img(SAVE_DIR+'segmented_'+path.split('\\')[-1], semantic_img)