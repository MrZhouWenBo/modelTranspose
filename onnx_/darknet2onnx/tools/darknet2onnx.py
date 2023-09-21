import sys
import torch
from pytorch.darknet2torch.tools.darknet2pytorch import Darknet
from pytorch.darknet2torch.tools.nms import *

class modelWithNMS(nn.Module):
    def __init__(self, model):
        super(modelWithNMS, self).__init__()
        self.model = model #config.CLASS_NUM
        self.nms = NMS()
        # self.conf_threshold = config.NMS_CONF_THRESHOLD
        # self.iou_threshold = config.NMS_IOU_THRESHOLD
        # self.use_class = self.class_num > 1 and config.NMS_USE_CLASS

    def forward(self, x, conf_threshold=0.5, iou_threshold=0.6):
        '''
        prediction[box] shape: [BATCH_SIZE, n_anchors_all, 4], box is xyxy format
        prediction[class] shape: [BATCH_SIZE, n_anchors_all, CLASS_NUM], CLASS_NUM is at least 1

        output result shape: [N, 8]
        dimension 1 order: conf, x1, y1, x2, y2, batch, class
        '''
        x = self.model.forward(x)
        x = self.nms.forward(x)
        return x
def transform_to_onnx(cfgfile, weightfile, batch_size=1, onnx_file_name=None, useNMS=False, modelInW=None, modelInH=None):
    model = Darknet(cfgfile)
    model.print_network()
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    if useNMS:
        newModel = modelWithNMS(model=model)
        input_names = ["input"]
        output_names = ['boxes']
        if modelInW is None:
            x = torch.randn((batch_size, 3, model.height, model.width), requires_grad=True)
        else: 
            x = torch.randn((batch_size, 3, modelInH, modelInW), requires_grad=True)
        torch.onnx.export(newModel,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)
        print("!!!!!12333333")
        return onnx_file_name
    dynamic = False
    if batch_size <= 0:
        dynamic = True

    input_names = ["input"]
    output_names = ['boxes', 'confs']

    if dynamic:
        x = torch.randn((1, 3, model.height, model.width), requires_grad=True)
        if not onnx_file_name:
            onnx_file_name = "yolov4_-1_3_{}_{}_dynamic.onnx".format(model.height, model.width)
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, model.height, model.width), requires_grad=True)
        if onnx_file_name == None:
            onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx".format(batch_size, model.height, model.width)
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        return onnx_file_name


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('weightfile')
    parser.add_argument('--batch_size', type=int, help="Static Batchsize of the model. use batch_size<=0 for dynamic batch size")
    parser.add_argument('--onnx_file_path', help="Output onnx file path")
    args = parser.parse_args()
    transform_to_onnx(args.config, args.weightfile, args.batch_size, args.onnx_file_path)

