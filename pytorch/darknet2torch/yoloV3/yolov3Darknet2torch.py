import sys
import torch
from ..tools.darknet2pytorch import Darknet

def transform_to_onnx(cfgfile, weightfile, batch_size=1, onnx_file_name=None):