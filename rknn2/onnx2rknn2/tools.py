# -*- codeing = utf-8 -*-
# @File : 
# @Author : 
# @Version : 
# @Software : 
# @Time : 
# @Purpose : 
import numpy as np
import cv2
from rknn.api import RKNN


class rknnOnnxConfig:
    def __init__(self):
        self.mens = [[0, 0, 0]]
        self.stds = [[255, 255, 255]]
        self.onnxPath = ''
        self.outRknnPath = ''
        self.target_platform = 'rk3588'
        self.quanParams = {'doQuan', False, 'dataPath', '', 'transRGB2BGR', False}

    def printParams(self):
        print(self.__dict__)


class usualGenRK2fromONNX:
    def __init__(self, cfg):
        '''
            onBoard 是否在板子上运行
        '''
        self.mens = cfg.mens
        self.stds = cfg.stds
        self.onnxPath = cfg.onnxPath
        self.outRknnPath = cfg.outRknnPath
        self.target_platform = cfg.target_platform
        self.quanParams = cfg.quanParams

    def printParams(self):
        print(self.__dict__)
    

    def genRKModel(self):
        # Create RKNN object
        rknn = RKNN(verbose=True)

        # Pre-process config
        print('--> Config model')
        rknn.config(mean_values=self.mens, std_values=self.stds, quant_img_RGB2BGR=self.quanParams['transRGB2BGR'],
                     target_platform=self.target_platform)
        print('--> Config model done')


        print('--> Loading model')
        ret = rknn.load_onnx(model=self.onnxPath)
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')

        print('--> Building model')
        ret = rknn.build(do_quantization=self.quanParams['doQuan'], dataset=self.quanParams['dataPath'])
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('--> Building model done')

        
        print('--> Export rknn model')
        ret = rknn.export_rknn(self.outRknnPath)
        if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
        print('done')
        rknn.release()
        return True




tescfgs = rknnOnnxConfig()
aa = usualGenRK2fromONNX(tescfgs)
aa.printParams()