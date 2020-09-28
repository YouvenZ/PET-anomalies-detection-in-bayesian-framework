import numpy as np
import torch
import os
import glob


import skimage
from skimage.measure import regionprops
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


from PIL import Image

#data_dir='/media/hmn-mednuc/InternalDisk_1/datasets/GAINED/resampled_croped/'

def get_path_modality_and_mask(data_dir):
    
    ls_idx_pet = list(map(lambda x:x.split('\\')[-1].split('_')[0] , glob.glob(data_dir +'\\'+ 'PET0\\*00001.nii*')))    
    ls_idx_mask = list(np.unique(np.array(list(map(lambda x:x.split('\\')[-1][:14], glob.glob(data_dir +'\\'+'PET0_mask*/*nii*'))))))
    #print(ls_idx_mask)
    #print(ls_idx_pet)
    ls_ids = sorted(list(set(ls_idx_pet).intersection(set(ls_idx_mask))))
        
    pt_path=[os.path.join(data_dir,'PET0',ids+'_00001.nii') for ids in ls_ids]
    
    mask_path=[os.path.join(data_dir,'PET0_masks',ids+'_mask.nii') for ids in ls_ids]
    
    
    return pt_path,mask_path


def padding(desired_shape,npa,value=0):

    if value==0:
        new_npa=np.zeros(desired_shape)
    else:
        new_npa=np.zeros(desired_shape)+value

    new_npa[:npa.shape[0],:npa.shape[1]] = npa

    return new_npa


def one_hot_encoding(y,n_classes):
    
    dim = len(y.shape)
    if dim == 2:
        one_hot = np.zeros((n_classes,y.shape[0], y.shape[1]))
    if dim == 3:
        one_hot = np.zeros((n_classes,y.shape[0], y.shape[1],y.shape[2]))
    for i,unique_value in enumerate(np.unique(y)):
        one_hot[i,:][y == unique_value] = 1
    return one_hot



def sum_tensor(input, axes, keepdim=False):
    axes = np.unique(axes)
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(int(ax))
    return input



def batch_dice(pred, y, false_positive_weight=1.0, eps=1e-6):
    '''
    compute soft dice over batch. this is a diffrentiable score and can be used as a loss function.
    only dice scores of foreground classes are returned, since training typically
    does not benefit from explicit background optimization. Pixels of the entire batch are considered a pseudo-volume to compute dice scores of.
    This way, single patches with missing foreground classes can not produce faulty gradients.
    :param pred: (b, c, y, x, (z)), softmax probabilities (network output).
    :param y: (b, c, y, x, (z)), one hote encoded segmentation mask.
    :param false_positive_weight: float [0,1]. For weighting of imbalanced classes,
    reduces the penalty for false-positive pixels. Can be beneficial sometimes in data with heavy fg/bg imbalances.
    :return: soft dice score (float).This function discards the background score and returns the mena of foreground scores.
    '''
    if len(pred.size()) == 4:
        axes = (0, 2, 3)
        intersect = sum_tensor(pred * y, axes, keepdim=False)
        denom = sum_tensor(false_positive_weight*pred + y, axes, keepdim=False)
        return torch.mean((2 * intersect / (denom + eps))[1:]) #only fg dice here.

    if len(pred.size()) == 5:
        axes = (0, 2, 3, 4)
        intersect = sum_tensor(pred * y, axes, keepdim=False)
        denom = sum_tensor(false_positive_weight*pred + y, axes, keepdim=False)
        return torch.mean((2 * intersect / (denom + eps))[1:]) #only fg dice here.

    else:
        raise ValueError('wrong input dimension in dice loss')