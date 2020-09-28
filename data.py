
import os
#from medicaltorch import datasets as mt_datasets
#from medicaltorch import transforms as mt_transforms
import matplotlib.pyplot as plt
#from tqdm import tqdm
import numpy as np
import nibabel as nib
import glob
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch._six import string_classes, int_classes
import skimage
from skimage.measure import regionprops
from torchvision.ops.boxes import box_area
from model_utils import *


class SegmentationPair2D(object):
    """This class is used to build 2D segmentation datasets. It represents
    a pair of of two data volumes (the input data and the ground truth data).

    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    :param cache: if the data should be cached in memory or not.
    :param canonical: canonical reordering of the volume axes.
    """
    def __init__(self, input_filename, gt_filename, cache=True,
                 canonical=False):
        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.canonical = canonical
        self.cache = cache

        self.input_handle = nib.load(self.input_filename)

        # Unlabeled data (inference time)
        if self.gt_filename is None:
            self.gt_handle = None
        else:
            self.gt_handle = nib.load(self.gt_filename)

        if len(self.input_handle.shape) > 3:
            raise RuntimeError("4-dimensional volumes not supported.")

        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()

        if self.gt_handle is not None:
            if not np.allclose(input_shape, gt_shape):
                raise RuntimeError('Input and ground truth with different dimensions.')

        if self.canonical:
            self.input_handle = nib.as_closest_canonical(self.input_handle)

            # Unlabeled data
            if self.gt_handle is not None:
                self.gt_handle = nib.as_closest_canonical(self.gt_handle)

    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = self.input_handle.header.get_data_shape()

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_shape = None
        else:
            gt_shape = self.gt_handle.header.get_data_shape()

        return input_shape, gt_shape

    def get_pair_data(self):
        """Return the tuble (input, ground truth) with the data content in
        numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'
        input_data = self.input_handle.get_fdata(cache_mode, dtype=np.float32)

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)
        
 
        return input_data, gt_data

    def get_pair_slice(self, slice_index, slice_axis=0):
        """Return the specified slice from (input, ground truth).

        :param slice_index: the slice number.
        :param slice_axis: axis to make the slicing.
        """
        if self.cache:
            input_dataobj, gt_dataobj = self.get_pair_data()
        else:
            # use dataobj to avoid caching
            input_dataobj = self.input_handle.dataobj

            if self.gt_handle is None:
                gt_dataobj = None
            else:
                gt_dataobj = self.gt_handle.dataobj

        if slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        if slice_axis == 2:
            input_slice = np.asarray(input_dataobj[..., slice_index],
                                     dtype=np.float32)
        elif slice_axis == 1:
            input_slice = np.asarray(input_dataobj[:, slice_index, ...],
                                     dtype=np.float32)
        elif slice_axis == 0:
            input_slice = np.asarray(input_dataobj[slice_index, ...],
                                     dtype=np.float32)

        # Handle the case for unlabeled data
        gt_meta_dict = None
        if self.gt_handle is None:
            gt_slice = None
        else:
            if slice_axis == 2:
                gt_slice = np.asarray(gt_dataobj[..., slice_index],
                                      dtype=np.float32)
            elif slice_axis == 1:
                gt_slice = np.asarray(gt_dataobj[:, slice_index, ...],
                                      dtype=np.float32)
            elif slice_axis == 0:
                gt_slice = np.asarray(gt_dataobj[slice_index, ...],
                                      dtype=np.float32)

        dreturn = {
            "input": input_slice,
            "gt": gt_slice,
        }
        
        return dreturn


class MRI2DSegmentationDataset(Dataset):
    """This is a generic class for 2D (slice-wise) segmentation datasets.

    :param filename_pairs: a list of tuples in the format (input filename,
                           ground truth filename).
    :param slice_axis: axis to make the slicing (default axial).
    :param cache: if the data should be cached in memory or not.
    :param transform: transformations to apply.
    """
    def __init__(self, data_dir, slice_axis=0, cache=False,
                 transform=None, slice_filter_fn=None, canonical=False):

        self.data_dir = data_dir
        self.pt_path,self.mask_path = get_path_modality_and_mask(self.data_dir)
        self.masks_dir = [dir_mask for dir_mask in os.listdir(self.data_dir) if 'masks' in dir_mask] 

        self.filename_pairs = [(p_pt,p_mask) for p_pt,p_mask in zip(self.pt_path,self.mask_path)]

        self.num_classes = 2
        self.num_query = 50
        self.handlers = []
        self.indexes = []
        self.transform = transform
        self.cache = cache
        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.canonical = canonical

        self._load_filenames()
        self._prepare_indexes()

    def _load_filenames(self):
        for input_filename, gt_filename in self.filename_pairs:
            segpair = SegmentationPair2D(input_filename, gt_filename.replace('PET0_masks',str(self.masks_dir[np.random.randint(len(self.masks_dir))])),self.cache, self.canonical)
            self.handlers.append(segpair)
        
    def _prepare_indexes(self):
        for segpair in self.handlers:
            input_data_shape, _ = segpair.get_pair_shapes()
            for segpair_slice in range(input_data_shape[1]):

                # Check if slice pair should be used or not
                if self.slice_filter_fn:
                    slice_pair = segpair.get_pair_slice(segpair_slice,
                                                        self.slice_axis)
                    
                    filter_fn_ret = self.slice_filter_fn(slice_pair)
                    if not filter_fn_ret:
                        continue

                item = (segpair, segpair_slice)
                self.indexes.append(item)

    def set_transform(self, transform):
        """This method will replace the current transformation for the
        dataset.

        :param transform: the new transformation
        """
        self.transform = transform

    def __len__(self):
        """Return the dataset size."""
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific index pair slices (input, ground truth).

        :param index: slice index.
        """
        image_size = (256,256)
        segpair, segpair_slice = self.indexes[index]
        pair_slice = segpair.get_pair_slice(segpair_slice,
                                            self.slice_axis)
# padding input
        

        pair_slice["input"]=padding((256,256),pair_slice["input"])
# normalisation 
        pair_slice["input"]/=100
# multi-label mask to binary mask

        pair_slice["gt"] = (pair_slice['gt']>= 1).astype(np.int)
        #obj_ids = np.unique(pair_slice["gt"])
# padding mask        
        pair_slice["gt"]=padding((256,256),pair_slice["gt"])
        pair_slice["gt"].reshape(256,256)
        
        #boxes = [list(region.bbox) for region in regionprops(skimage.measure.label(pair_slice["gt"])) if region.area>=6]

        #boxes = boxes + [[1,1,1,1] for _ in range(self.num_query-len(boxes))]
        



# one hot encoding mask        
        

        
        
        pair_slice["gt"]=one_hot_encoding(pair_slice["gt"],2)
    
        # if boxes == []:
        #     #labels = [0]
        #     #labels = [self.num_classes-2]
        #     labels = [self.num_classes-2]+[self.num_classes]*(self.num_query-1)
        #     print(len(labels))
        # else:
        #     #labels = [self.num_classes-1]*len(boxes)
        #     labels = [self.num_classes-1]*len(boxes)+[self.num_classes]*(self.num_query-len(boxes))
        #     print(len(labels))

        # boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # boxes[:, 2:] += boxes[:, :2]
        # boxes[:, 0::2].clamp_(min=0, max=256)
        # boxes[:, 1::2].clamp_(min=0, max=256)


        # labels = torch.as_tensor(labels, dtype=torch.int64)
        
        #keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        #print(keep)
        #boxes = boxes[keep]
        #labels = labels[keep]
        # there is only one class

        # Consistency with torchvision, returning PIL Image
        # Using the "Float mode" of PIL, the only mode
        # supporting unbounded float32 values

        input_img = pair_slice["input"]

        # Handle unlabeled data
        if pair_slice["gt"] is None:
            gt_img = None
        else:
            gt_img = pair_slice["gt"]
 
        # target_data_dict = {
        #     'boxes':boxes.float(),
        #     'labels':labels,
        # }

        if self.transform is not None:
            data_dict = self.transform(data_dict)
        pet=torch.from_numpy(input_img[np.newaxis]).float()
        gt_img=torch.from_numpy(gt_img).float()

        #print(pet.shape,gt_img.shape)
        
        
        return pet,gt_img
        #return pet,gt_img,target_data_dict

data_dir='/media/hmn-mednuc/InternalDisk_1/datasets/GAINED/resampled_croped/'
#train_dataset = MRI2DSegmentationDataset(data_dir=data_dir, slice_axis=1,transform=mt_transforms.ToTensor())
# train_dataset = MRI2DSegmentationDataset(data_dir=data_dir, slice_axis=1)

# print(len(train_dataset))

# a,b,c = train_dataset[70]
# print(a)
# print(b)
# print(c)



# #print(data["input"].shape)
# #print(data["gt"].shape)
# # print(data["boxes"])
# # print(data["labels"])

# def prepare_loader(dataset,batch_size=4,shuffle=True):
    
#     train_set,valid_set = random_split(dataset,[int(len(dataset)*0.8),int(len(dataset)*0.2)+1])

#     train_loader = DataLoader(train_set, batch_size=batch_size, num_workers = 4, shuffle=shuffle, pin_memory=torch.cuda.is_available())
#     val_loader = DataLoader(valid_set, batch_size=batch_size, num_workers = 4, shuffle=shuffle, pin_memory=torch.cuda.is_available())
#     return train_loader,val_loader



# train_loader,val_loader=prepare_loader(train_dataset)

# for i in range(0,5):
#     for a,b,c in train_loader:
#         print(a.shape,b.shape,c["boxes"])
#         if i==5:break

#next(iter(train_loader))

# plt.imshow(data['input'][0])
# plt.imshow(data['gt'][0][1],cmap='gray')

# plt.show()


# ax[0].imshow(data[])
# ax[1].imshow(data[])