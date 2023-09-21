import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime

sys.path.append('/media/jose/31c0898d-65db-4216-a49b-632493cb9388/shareCode/modelTranspose')

from tools.utils import *
from onnx_.darknet2onnx.tools.darknet2onnx import *



def detect(session,image_src, namesfile=""):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]
    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})

    boxes = post_processing(img_in, 0.4, 0.6, outputs)
    print(boxes)
    return boxes


def getInPutsInfos():
    '''
        定义输入
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='./models/yoloface-gpu.cfg', help="darknet cfg file path")
    parser.add_argument('--weight_file', type=str, default='./models/yoloface-gpu.weights', help="darknet weight file path")
    parser.add_argument('--testImg_path', type=str,  default='./testDatas/test.jpg',help="test img path")

    parser.add_argument('--onnx_file_path', type=str, default='./models/yoloface-gpu.onnx', help="Output onnx file path")
    parser.add_argument('--batch_size', type=int,  default=1, help="Static Batchsize of the model. use batch_size<=0 for dynamic batch size")
    parser.add_argument('--modelInW', type=int,  default=128, help="Static Batchsize of the model. use batch_size<=0 for dynamic batch size")
    parser.add_argument('--modelInH', type=int,  default=96, help="Static Batchsize of the model. use batch_size<=0 for dynamic batch size")
    args = parser.parse_args()
    return args

def exportyolo():
    args = getInPutsInfos()
    if args.batch_size <= 0:
        onnx_path_demo = transform_to_onnx(args.cfg_file, args.weight_file, args.batch_size, args.onnx_file_path)
    else:
        # Transform to onnx as specified batch size
        if args.batch_size != 1:
            transform_to_onnx(args.cfg_file, args.weight_file, args.batch_size, args.onnx_file_path)
        else:
            # Transform to onnx as demo
            onnx_path_demo = transform_to_onnx(args.cfg_file, args.weight_file, args.batch_size, args.onnx_file_path)

    session = onnxruntime.InferenceSession(onnx_path_demo)
    # session = onnx.load(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape, session.get_outputs()[0].shape)
    allresult = session.get_outputs()
    for result in allresult:
        print(type(result))
    print(type(allresult))
    image_src = cv2.imread(args.testImg_path)
    boxes =  detect(session, image_src)
    boxes = boxes[0]
    p1x = boxes[0][0]*image_src.shape[1]
    p1y = boxes[0][1]*image_src.shape[0]
    p2x = boxes[0][2]*image_src.shape[1]
    p2y = boxes[0][3]*image_src.shape[0]
    draw_0 = cv2.rectangle(image_src, (int(p1x), int(p1y)), (int(p2x), int(p2y)), (0, 0, 255))
    cv2.imwrite("./testDatas/testRet.jpg", draw_0)


def exportyoloWithNMS():
    args = getInPutsInfos()
    if args.batch_size <= 0:
        onnx_path_demo = transform_to_onnx(args.cfg_file, args.weight_file, args.batch_size, args.onnx_file_path, True,args.modelInW,args.modelInH)
    else:
        # Transform to onnx as specified batch size
        if args.batch_size != 1:
            transform_to_onnx(args.cfg_file, args.weight_file, args.batch_size, args.onnx_file_path, True,args.modelInW,args.modelInH)
        else:
            # Transform to onnx as demo
            onnx_path_demo = transform_to_onnx(args.cfg_file, args.weight_file, args.batch_size, args.onnx_file_path, True,args.modelInW,args.modelInH)

    session = onnxruntime.InferenceSession(onnx_path_demo)
    # session = onnxruntime.InferenceSession(args.onnx_file_path)
    image_src = cv2.imread(args.testImg_path)
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]
    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})
    print(outputs)
    outputs = outputs[0]
    p1x = outputs[0][0]*image_src.shape[1]
    p1y = outputs[0][1]*image_src.shape[0]
    p2x = outputs[0][2]*image_src.shape[1]
    p2y = outputs[0][3]*image_src.shape[0]
    draw_0 = cv2.rectangle(image_src, (int(p1x), int(p1y)), (int(p2x), int(p2y)), (0, 0, 255))
    cv2.imwrite("./testDatas/testRet22.jpg", draw_0)
if __name__ == '__main__':
    # exportyolo()
    exportyoloWithNMS()
