import glob
import os
import numpy as np
import cv2
import csv

import torch
from torchvision import transforms

from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

import scipy.io

def get_vis_utils():
    names = {}
    with open('Code/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]

def visualize_result(img, pred, filename, index=None):
    colors = scipy.io.loadmat('Code/color150.mat')['colors']

    # filter prediction class if requested
    # if index is not None:
    #     pred = pred.copy()
    #     pred[pred != index] = -1
    #     # print(f'{names[index+1]}:')
        
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    # im_vis = np.concatenate((img, pred_color), axis=1)

    # cv2.imshow('im_vis', im_vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(filename, pred_color)

def build_network():
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights='Code/ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights='Code/ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    segmentation_module.cuda()

    return segmentation_module

def preprocess_img(img):
    img_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])
    img_data = img_to_tensor(img)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]
    print(output_size)

    return singleton_batch, output_size

def eval(model, batch, input_channel_size, input_img):
    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = model(batch, segSize=input_channel_size)
        
    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    return pred

def parse_img(path):
    '''
    Parse stereocam's left cam's RGB frames
    Input: relative path directory containing the .png files
    Output: list containing open3d pcd  
    '''
    # read img
    frame = cv2.imread(path)
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame

def get_img_paths(rgb_data_dir):
    absolute_dir = os.path.join(os.getcwd(),rgb_data_dir)
    img_paths = sorted(glob.glob(os.path.join(absolute_dir, "*.png")))
    return img_paths


def semantic_segmentation():
    pass

if __name__ == '__main__':
    RGB_DATA_DIR = 'KITTI-360/KITTI-Small/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/'
    SAVE_DIR = 'results/segmented_image/'

    model = build_network()

    img_paths = get_img_paths(RGB_DATA_DIR)

    for path in img_paths:
        filename = path.split('\\')[-1]

        rgb_frame = parse_img(path)
        batch, batch_size = preprocess_img(rgb_frame)
        prediction = eval(model, batch, batch_size, rgb_frame)
        visualize_result(rgb_frame, prediction, SAVE_DIR + filename)


    #     semantic_img, _ = semantic_segmentation(model, rgb_frame)

    #     if isVis:
    #         cv2.imshow('semantic img', semantic_img)
    #     if isSave:
    #         cv2.imwrite('sem.png', semantic_img)
