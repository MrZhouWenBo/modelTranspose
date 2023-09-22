import numpy as np
import cv2
from rknn.api import RKNN

class rnkkRUn:
    def __init__(self, rknnModelPath=''):
        '''
            onBoard 是否在板子上运行
        '''
        self.rknnModelPath = rknnModelPath
        self.rknnModel = RKNN(verbose=True)
        # load rknn model
        print('--> Load rknn model')
        ret = self.rknnModel.load_rknn(self.rknnModelPath)
        if ret != 0:
            print('Load rknn model failed!')
            exit(ret)
        print('done')

        print('--> List devices')
        self.rknn.list_devices()
        # 获取输入信息
        input_info = self.rknnModel.get_input()
        print(f"model{rknnModelPath.split('')[-1]}input infos {input_info}")
        output_info = self.rknnModel.get_output()
        print(f"model{rknnModelPath.split('')[-1]}output infos {output_info}")

    def runtest(self, inputs, platform=''):
        if platform is '':
            ret = self.rknnModel.init_runtime()
            if ret != 0:
                print('Init runtime environment failed!')
                exit(ret)
            print('done')

            print('--> Get sdk version')
            sdk_version = self.rknnModel.get_sdk_version()
            print(sdk_version)
            # eval perf
            print('--> Eval perf')
            self.rknnModel.eval_perf(inputs=[inputs])

            # eval perf
            print('--> Eval memory')
            self.rknnModel.eval_memory()

            print('--> Accuracy analysis')
            ret = self.rknnModel.accuracy_analysis(inputs=[inputs], output_dir='./snapshot', target='rk3588')
            if ret != 0:
                print('Accuracy analysis failed!')
                exit(ret)
            print('done')

        else:
            ret = self.rknnModel.init_runtime(target=platform, perf_debug=True, eval_mem=True)
            if ret != 0:
                print('Init runtime environment failed!')
                exit(ret)
            print('done')

            print('--> Accuracy analysis')
            ret = self.rknnModel.accuracy_analysis(inputs=[inputs], output_dir='./snapshot')
            if ret != 0:
                print('Accuracy analysis failed!')
                exit(ret)
            print('done')

        print('--> Running model')
        outputs = self.rknnModel.inference(inputs=[inputs])
        return outputs
    
    def releaseModel(self):
        self.rknnModel.release()