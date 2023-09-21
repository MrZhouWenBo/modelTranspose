import torch
import torch.nn as nn
from torchvision.ops import nms

class NMS(nn.Module):
    '''
        note ! 当前NMS为单类 单batch的情况 后续通用型需要修改
    '''

    def __init__(self, classNumber=1):
        super(NMS, self).__init__()
        self.class_num = classNumber #config.CLASS_NUM
        # self.conf_threshold = config.NMS_CONF_THRESHOLD
        # self.iou_threshold = config.NMS_IOU_THRESHOLD
        # self.use_class = self.class_num > 1 and config.NMS_USE_CLASS

    def forward(self, prediction, conf_threshold=0.5, iou_threshold=0.6):
        '''
        prediction[box] shape: [BATCH_SIZE, n_anchors_all, 4], box is xyxy format
        prediction[class] shape: [BATCH_SIZE, n_anchors_all, CLASS_NUM], CLASS_NUM is at least 1

        output result shape: [N, 8]
        dimension 1 order: conf, x1, y1, x2, y2, batch, class
        '''
        box_array = prediction[0]
        # [batch, num, num_classes]
        confs = prediction[1]
        # if type(box_array).__name__ != 'ndarray':
        #     box_array = box_array.cpu().detach().numpy()
        #     confs = confs.cpu().detach().numpy()
        valid_mask = confs >= conf_threshold
        valid_boxes = box_array[valid_mask, :]
        selected_index = nms(valid_boxes, confs[valid_mask], iou_threshold)
        selected_conf = confs[valid_mask][selected_index].unsqueeze(-1)
        selected_boxes = valid_boxes[selected_index]
        return torch.cat([selected_boxes, selected_conf], dim=1)
        # probs, classification = torch.max(prediction['class'].sigmoid(), 2)
        # valid_mask = probs >= conf_threshold
        # valid_mask = confs >= conf_threshold
        # valid_boxes = boxes[valid_mask, :]
        # if not is_exporting() and valid_boxes.shape[0] == 0:
        #     return torch.zeros([0, 8], device=boxes.device)

        # add center offset for different batch & class, so nms consider these boxes non-overlapping
        # batch_offset = torch.arange(boxes.shape[0], device=boxes.device)
        # batch_offset = batch_offset.unsqueeze(1).expand(-1, boxes.shape[1])
        # if self.use_class:
        #     offset = classification + self.class_num * batch_offset
        # else:
        #     offset = batch_offset
        # offset = offset[valid_mask].reshape(-1, 1)
        # selected_index = nms(valid_boxes + offset, probs[valid_mask], self.iou_threshold)

        # selected_conf = probs[valid_mask][selected_index].unsqueeze(-1)
        # selected_boxes = valid_boxes[selected_index]
        # selected_batch = batch_offset[valid_mask][selected_index].unsqueeze(-1)
        # if self.class_num > 0:
        #     selected_class = classification[valid_mask][selected_index].unsqueeze(-1)
        # else:
        #     selected_class = -torch.ones_like(selected_batch)
        # # result: conf, x1, y1, x2, y2, batch, class
        # return torch.cat([selected_conf, selected_boxes, selected_batch, selected_class], dim=1)
