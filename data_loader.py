import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import SimpleITK as sitk
import glob
import numpy as np
import os
import random
import pickle
import collections

#from medicaltorch import transforms as mt_transforms

from tqdm import tqdm
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset
import torch
from torch._six import string_classes, int_classes

from PIL import Image


data_dir='/media/hmn-mednuc/InternalDisk_1/datasets/GAINED/resampled_croped/'

def get_path_modality_and_mask(data_dir):
    
    ls_idx_pet = list(map(lambda x:x.split('/')[-1].split('_')[0] , glob.glob(data_dir + 'PET0/*00001.nii*')))
    ls_idx_mask = list(np.unique(np.array(list(map(lambda x:x.split('/')[-1][:14], glob.glob(data_dir + 'PET0_mask*/*nii*'))))))
    ls_ids = sorted(list(set(ls_idx_pet).intersection(set(ls_idx_mask))))
    print(ls_ids)
        
    pt_path=[os.path.join(data_dir,'PET0',ids+'_00001.nii') for ids in ls_ids]    
    mask_path=[os.path.join(data_dir,'PET0_masks',ids+'_mask.nii') for ids in ls_ids]
    
    
    return pt_path,mask_path


def padding_with_zero(desired_shape,npa):
    new_npa=np.zeros(desired_shape)
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



# class data_loader_2D(Dataset):
    
#     def __init__(self, data_dir, target, n_classes,desired_shape, transform=None):
#         self.n_classes = n_classes
#         self.desired_shape = desired_shape
#         self.data = data
#         self.target = target
#         self.transform = transform
        
#     def __getitem__(self, index):
#         x = self.data[index]
#         x = padding_with_zero(self.desired_shape,x)[np.newaxis]
#         #x = x[np.newaxis]
#         x /= 100
        
#         #y = self.target[index]
#         y = (self.target[index]>= 1).astype(np.int)
#         y = padding_with_zero(self.desired_shape,y).astype(np.int)
#         y_ohe = one_hot_encoding(y,self.n_classes)
        
        
#         if self.transform:
#             x = self.transform(x)
#         return torch.from_numpy(x).float(), torch.from_numpy(y_ohe).float()
        
#     def __len__(self):
#         return len(self.data)


# dataset = data_loader_2D(all_pt_img, all_mask_img,n_classes=2,desired_shape=(256,256))
# train_set,valid_set =  random_split(dataset,[int(len(dataset)*0.8),int(len(dataset)*0.2)])

# train_loader = DataLoader(train_set, batch_size=2, num_workers = 4, shuffle=True, pin_memory=torch.cuda.is_available())
# val_loader = DataLoader(valid_set, batch_size=2, num_workers = 4, shuffle=True, pin_memory=torch.cuda.is_available())






# class PET_only_dataset(Dataset):
#     images = []
#     labels = []

#     def __init__(self, dataset_location):
#         data = {}
#         for file in os.listdir(dataset_location):
#             filename = os.fsdecode(file)
#             if '.pickle' in filename:
#                 print("Loading file", filename)
#                 file_path = dataset_location + filename
#                 bytes_in = bytearray(0)
#                 input_size = os.path.getsize(file_path)
#                 with open(file_path, 'rb') as f_in:
#                     for _ in range(0, input_size, max_bytes):
#                         bytes_in += f_in.read(max_bytes)
#                 new_data = pickle.loads(bytes_in)
#                 data.update(new_data)
        
#         for key, value in data.items():
#             self.images.append(value['image'].astype(float))
#             self.labels.append(value['masks'])
#             self.series_uid.append(value['series_uid'])

#         assert (len(self.images) == len(self.labels) == len(self.series_uid))

#         for img in self.images:
#             assert np.max(img) <= 1 and np.min(img) >= 0
#         for label in self.labels:
#             assert np.max(label) <= 1 and np.min(label) >= 0

#         del new_data
#         del data

#     def __getitem__(self, index):
#         image = np.expand_dims(self.images[index], axis=0)

#         #Randomly select one of the four labels for this image
#         label = self.labels[index][random.randint(0,3)].astype(float)
#         if self.transform is not None:
#             image = self.transform(image)

#         series_uid = self.series_uid[index]

#         # Convert image and label to torch tensors
#         image = torch.from_numpy(image)
#         label = torch.from_numpy(label)

#         #Convert float32 to float tensors
#         image = image.type(torch.FloatTensor)
#         label = label.type(torch.FloatTensor)

#         return image, label

#     # Override to give PyTorch size of dataset
#     def __len__(self):
#         return len(self.images)






__numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'float32': torch.ByteTensor,
}


class SampleMetadata(object):
    def __init__(self, d=None):
        self.metadata = {} or d

    def __setitem__(self, key, value):
        self.metadata[key] = value

    def __getitem__(self, key):
        return self.metadata[key]

    def __contains__(self, key):
        return key in self.metadata

    def keys(self):
        return self.metadata.keys()


class BatchSplit(object):
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        batch_len = len(self.batch["input"])
        for i in range(batch_len):
            single_sample = {k: v[i] for k, v in self.batch.items()}
            single_sample['index'] = i
            yield single_sample
        raise StopIteration


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

        self.pt_input_handle = nib.load(self.input_filename)
        self.ct_input_handle = nib.load(self.input_filename.replace('_00001.nii','_00000.nii'))


        # Unlabeled data (inference time)
        if self.gt_filename is None:
            self.gt_handle = None
        else:
            self.gt_handle = nib.load(self.gt_filename)

        if len(self.pt_input_handle.shape) > 4:
            raise RuntimeError("4-dimensional volumes not supported.")

        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()

        if self.gt_handle is not None:
            if not np.allclose(input_shape, gt_shape):
                raise RuntimeError('Input and ground truth with different dimensions.')

        if self.canonical:
            self.pt_input_handle = nib.as_closest_canonical(self.pt_input_handle)
            self.ct_input_handle = nib.as_closest_canonical(self.ct_input_handle)


            # Unlabeled data
            if self.gt_handle is not None:
                self.gt_handle = nib.as_closest_canonical(self.input_data)

    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = self.pt_input_handle.header.get_data_shape()

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
        input_data_pt = self.pt_input_handle.get_fdata(cache_mode, dtype=np.float32)[np.newaxis]
        input_data_ct = self.ct_input_handle.get_fdata(cache_mode, dtype=np.float32)[np.newaxis]

        self.inputs_data = np.concatenate((input_data_pt,input_data_ct),axis=0)
        print(self.inputs_data.shape)

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)[np.newaxis]
            print(gt_data.shape)
        return self.inputs_data, gt_data

    def get_pair_slice(self, slice_index, slice_axis=1):
        """Return the specified slice from (input, ground truth).

        :param slice_index: the slice number.
        :param slice_axis: axis to make the slicing.
        """
        if self.cache:
            input_dataobj, gt_dataobj = self.get_pair_data()
        else:
            # use dataobj to avoid caching
            input_dataobj = self.inputs_data.dataobj

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

            gt_meta_dict = SampleMetadata({
                "zooms": self.gt_handle.header.get_zooms()[:2],
                "data_shape": self.gt_handle.header.get_data_shape()[:2],
            })

        input_meta_dict = SampleMetadata({
            "zooms": self.gt_handle.header.get_zooms()[:2],
            "data_shape": self.gt_handle.header.get_data_shape()[:2],
        })

        dreturn = {
            "input": input_slice,
            "gt": gt_slice,
            "input_metadata": input_meta_dict,
            "gt_metadata": gt_meta_dict,
        }
        
        return dreturn


# def get_transforms(cf, is_training=True):
#     if cf.da_kwargs['random_crop']:
#         #crop = RandSpatialCropd(keys=['tep','ct', 'label'],
#         #                        roi_size=(cf.patch_size[0], cf.patch_size[1], cf.patch_size[2]),
#         #                       random_size=False)
#         print('NOT IMPLEMENTED YET')
#     else:
#         #crop = RandCropByPosNegLabeld(keys=['tep', 'ct', 'label'], label_key='label',
#         #                              size=(cf.patch_size[0], cf.patch_size[1], cf.patch_size[2]),
#         #                              pos=cf.da_kwargs['n_pos'], neg=cf.da_kwargs['n_neg'],
#         #                              num_samples=cf.n_samples)
#     if is_training:
#         transforms = Compose([
#             LoadNiftid(keys=['tep', 'ct', 'label']),
#             AddChanneld(keys=['tep', 'ct', 'label']),
#             #ThresholdIntensityd(keys=['label'], threshold=0.5, above=True, cval=1),
#             #Spacingd(keys=['tep','ct', 'label'], pixdim=(4.,4.,4.), interp_order=(3, 0), mode='nearest'),
#             #Orientationd(keys=['tep','ct','label'], axcodes='RAS'),
#             ScaleIntensityRanged(keys=['tep'], a_min=cf.tep_norm[0], a_max=cf.tep_norm[1], b_min=0.0, b_max=1.0,
#                                  clip=True),
#             ScaleIntensityRanged(keys=['ct'], a_min=cf.ct_norm[0], a_max=cf.ct_norm[1], b_min=0.0, b_max=1.0,
#                                  clip=True),
#             CropForegroundd(keys=['tep', 'ct', 'label'], source_key='tep'),
#             # randomly crop out patch samples from big image based on pos / neg ratio
#             # the image centers of negative samples must be in valid image area
#             RandCropByPosNegLabeld(keys=['tep', 'ct', 'label'], label_key='label',
#                                           size=(cf.patch_size[0], cf.patch_size[1], cf.patch_size[2]),
#                                           pos=cf.da_kwargs['n_pos'], neg=cf.da_kwargs['n_neg'],
#                                           num_samples=cf.n_samples),
#             #RandGaussianNoised(keys = ['tep','ct'],prob = cf.da_kwargs['p_gaussian_noise_per_sample']),
#             # user can also add other random transforms
#             #Rand3DElasticd(keys =['ct','tep' , 'label'],
#             #               prob= cf.da_kwargs['p_elastic'],
#             #               spatial_size = (cf.patch_size[0], cf.patch_size[1], cf.patch_size[2]),
#             #               sigma_range = cf.da_kwargs['sigma'],
#             #               magnitude_range =  cf.da_kwargs['magnitude'] ),
#             #RandAffined(keys=['ct','tep' , 'label'], mode=('bilinear','bilinear', 'nearest'),
#             #            prob=1, spatial_size=(cf.patch_size[0], cf.patch_size[1], cf.patch_size[2]),
#             #            rotate_range=(cf.da_kwargs['angle_x'][1], cf.da_kwargs['angle_y'][1], cf.da_kwargs['angle_z'][1]),
#             #            scale_range=(cf.da_kwargs['scale_range_x'], cf.da_kwargs['scale_range_y'], cf.da_kwargs['scale_range_z'])),

#             ToTensord(keys=['tep', 'ct', 'label'])
#         ])
#     else:
#         transforms = Compose([
#             LoadNiftid(keys=['tep', 'ct', 'label']),
#             AddChanneld(keys=['tep', 'ct', 'label']),
#             #ThresholdIntensityd(keys=['label'], threshold=0.5, above=True, cval=1),
#             # Spacingd(keys=['tep','ct', 'label'], pixdim=(4.,4.,4.), interp_order=(3, 0), mode='nearest'),
#             # Orientationd(keys=['tep','ct','label'], axcodes='RAS'),
#             ScaleIntensityRanged(keys=['tep'], a_min=cf.tep_norm[0], a_max=cf.tep_norm[1], b_min=0.0, b_max=1.0,
#                                  clip=True),
#             ScaleIntensityRanged(keys=['ct'], a_min=cf.ct_norm[0], a_max=cf.ct_norm[1], b_min=0.0, b_max=1.0,
#                                  clip=True),
#             CropForegroundd(keys=['tep', 'ct', 'label'], source_key='tep'),
#             # randomly crop out patch samples from big image based on pos / neg ratio
#             # the image centers of negative samples must be in valid image area
#             crop,
#             ToTensord(keys=['tep', 'ct', 'label'])
#         ])
#     return transforms

class MRI2DSegmentationDataset(Dataset):
    """This is a generic class for 2D (slice-wise) segmentation datasets.

    :param filename_pairs: a list of tuples in the format (input filename,
                           ground truth filename).
    :param slice_axis: axis to make the slicing (default axial).
    :param cache: if the data should be cached in memory or not.
    :param transform: transformations to apply.
    """
    def __init__(self, data_dir, slice_axis=1, cache=False,
                 transform=None, slice_filter_fn=None, canonical=False):
        
        self.data_dir = data_dir
        self.pt_path,self.mask_path = get_path_modality_and_mask(self.data_dir)

        self.masks_dir = [dir_mask for dir_mask in os.listdir(self.data_dir) if 'masks' in dir_mask]   

        self.filename_pairs = [(p_pt,p_mask) for p_pt,p_mask in zip(self.pt_path,self.mask_path)]
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
            segpair = SegmentationPair2D(input_filename, gt_filename.replace('PET0_masks',str(self.masks_dir[np.random.randint(len(self.masks_dir))])),
                                         self.cache, self.canonical)
            self.handlers.append(segpair)
        
    def _prepare_indexes(self):
        for segpair in self.handlers:
            input_data_shape, _ = segpair.get_pair_shapes()
            
            #print(input_data_shape)
            for segpair_slice in range(input_data_shape[self.slice_axis]):

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

    # def compute_mean_std(self, verbose=True):
    #     """Compute the mean and standard deviation of the entire dataset.
        
    #     :param verbose: if True, it will show a progress bar.
    #     :returns: tuple (mean, std dev) 
    #     """
    #     sum_intensities = 0.0
    #     numel = 0

    #     with DatasetManager(self,
    #                         override_transform=mt_transforms.ToTensor()) as dset:
    #         pbar = tqdm(dset, desc="Mean calculation", disable=not verbose)
    #         for sample in pbar:
    #             input_data = sample['input']
    #             sum_intensities += input_data.sum()
    #             numel += input_data.numel()
    #             pbar.set_postfix(mean="{:.2f}".format(sum_intensities / numel),
    #                              refresh=False)

    #         training_mean = sum_intensities / numel

    #         sum_var = 0.0
    #         numel = 0

    #         pbar = tqdm(dset, desc="Std Dev calculation", disable=not verbose)
    #         for sample in pbar:
    #             input_data = sample['input']
    #             sum_var += (input_data - training_mean).pow(2).sum()
    #             numel += input_data.numel()
    #             pbar.set_postfix(std="{:.2f}".format(np.sqrt(sum_var / numel)),
    #                              refresh=False)

    #     training_std = np.sqrt(sum_var / numel)
    #     return training_mean.item(), training_std.item()

    def __len__(self):
        """Return the dataset size."""
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific index pair slices (input, ground truth).

        :param index: slice index.
        """
        segpair, segpair_slice = self.indexes[index]
        pair_slice = segpair.get_pair_slice(segpair_slice,
                                            self.slice_axis)

        # Consistency with torchvision, returning PIL Image
        # Using the "Float mode" of PIL, the only mode
        # supporting unbounded float32 values
        input_img = torch.from_numpy(pair_slice["input"])

        # Handle unlabeled data
        if pair_slice["gt"] is None:
            gt_img = None
        else:
            gt_img = torch.from_numpy(pair_slice["gt"])


        data_dict = {
            'input': input_img,
            'gt': gt_img,
            'input_metadata': pair_slice['input_metadata'],
            'gt_metadata': pair_slice['gt_metadata'],
        }

        data_dict_simple = {
            'input': input_img,
            'gt': gt_img,
        }

        if self.transform is not None:
            data_dict['input']=self.transform['inputs'](data_dict['input'])
            data_dict['gt']=self.transform['mask'](data_dict['gt'])
            #data_dict = self.transform(data_dict)

        return data_dict


class DatasetManager(object):
    def __init__(self, dataset, override_transform=None):
        self.dataset = dataset
        self.override_transform = override_transform
        self._transform_state = None

    def __enter__(self):
        if self.override_transform:
            self._transform_state = self.dataset.transform
            self.dataset.transform = self.override_transform
        return self.dataset

    def __exit__(self, *args):
        if self._transform_state:
            self.dataset.transform = self._transform_state





# trans={'inputs':transforms.Compose(
#      [transforms.ToPILImage(mode='LA'), transforms.Pad((0,0,0,128), fill=0, padding_mode='constant'),transforms.ToTensor(),]),
    
#     'mask':transforms.Compose(
#     [transforms.ToPILImage(mode='L'), transforms.Pad((0,0,0,128), fill=0, padding_mode='constant'),transforms.ToTensor(),])}

# train_dataset = MRI2DSegmentationDataset(data_dir,cache=True,transform=trans)
# #train_dataset.__getitem__(55)

# set_data=train_dataset[845]
# print(set_data['input'])
# print(set_data['gt'])
# #print(Image.fromarray(set_data['input']))
# print(len(train_dataset))