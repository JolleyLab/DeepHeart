import os
import logging
import h5py
import preprocessing
from base import BaseDataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from preprocessing.transforms import AffineFromFilename, NumpyFromFilename, ToTensor, OneHotTransform, LabelFromFilename
from preprocessing.transforms import DistanceTransform, Concatenate
from preprocessing.normalization import MinMaxScaling, NormalizeDistanceMap


class TricuspidHDF5Dataset(Dataset):

    def __init__(self, path, input_types, transform=None, mode='train'):
        self.input_types = input_types
        if mode == 'test':
            self.input_types.append('affines')
        self.mode = mode
        self.file_path = path
        self.dataset = None
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_len = len(f[self.mode].keys())

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')[self.mode]
        data = self.dataset[list(self.dataset.keys())[index]]
        data = {input_type: data[input_type][()] for input_type in self.input_types}
        data["cases"] = list(self.dataset.keys())[index]
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return self.dataset_len


class TricuspidDataDirDataset(Dataset):

    def __init__(self, data_dir, input_types, training=True, transform=None):
        sub_dir_names = os.listdir(data_dir)
        pre_splitted = False
        if all(mode in sub_dir_names for mode in ["train", "test"]):
            logging.info("Data directory was pre splitted into train and test datasets")
            data_dir = os.path.join(data_dir, "train" if training else "test")
            pre_splitted = True

        input_paths = {input_type: os.path.join(data_dir, input_type) for input_type in input_types}

        # NB: expecting that same case names are used for all input images/labels
        file_list = [f for f in os.listdir(list(input_paths.values())[0]) if 'nii.gz' in f]

        if not pre_splitted:
            logging.info("Automatically splitting data directory into train and test datasets")
            self.training_subset = file_list[0:int(0.9 * len(file_list))]
            self.testing_subset = file_list[int(0.9 * len(file_list)):]
            file_list = self.training_subset if training else self.testing_subset

        self.data = self.make_dataset(file_list, input_paths)
        self.transform = transform

    def make_dataset(self, file_list, paths):
        dataset = []
        for f in file_list:
            data = {input_type: os.path.join(path, f) for input_type, path in paths.items()}
            data["cases"] = f.replace(".nii.gz", "")
            dataset.append(data)
        return dataset

    def __getitem__(self, index):
        """ transforms should not affect original data """
        data = self.data[index].copy()
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data)


class CommonDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, inputs,
                 input_field, output_field, data_augmentation=None, training=True, pin_memory=False):

        # check input/output type
        inputs_types = [inp["field"] for inp in inputs]

        trsfm = list()

        if os.path.isdir(data_dir):
            self.dataset = TricuspidDataDirDataset(data_dir, inputs_types, training=training)

            # get affine from any input image
            if not training:
                trsfm.append(AffineFromFilename(inputs_types[0], out_field="affines"))

            # loading inputs images
            trsfm.extend([NumpyFromFilename(inp["field"]) for inp in inputs if inp["input_type"] == "image"])

            # loading inputs labels
            trsfm.extend([LabelFromFilename(inp["field"],
                                            merged=inp["n_classes"] == 2) for inp in inputs if inp["input_type"] == "label"])
        else:
            self.dataset = TricuspidHDF5Dataset(data_dir, inputs_types, mode="train" if training else "test")

        # add all data augmentation if training and data_augmentation section is available
        inference_allowed_data_augmentations = ["HistogramEqualization", "HistogramClipping"]
        if data_augmentation:
            aug_trsfms = list()
            for aug in data_augmentation:
                # TODO: add argument for checking if to use data augmentation in inference
                allowed_in_inference = aug['type'] in inference_allowed_data_augmentations
                if allowed_in_inference or training is True:
                    if allowed_in_inference and training is False:
                        aug['args']["execution_probability"] = 1.0
                        logging.info("using {} with execution_probability {}".format(aug['type'],
                                                                                     aug['args']["execution_probability"]))
                    logging.info("Adding {} for {}".format(aug['type'], aug['args']["field" if "field" in aug['args']
                    else "fields"]))
                    aug_trsfms.append(getattr(preprocessing.augmentation, aug['type'])(**aug['args']))
            trsfm.extend(aug_trsfms)

        # one hot transform
        for input_image in inputs:
            if input_image["output_type"] in ["distmap", "onehot"]:
                logging.info("Adding OneHotTransform for {}".format(input_image["field"]))
                trsfm.append(OneHotTransform(input_image["field"]))

        # distance transform
        for input_image in inputs:
            if input_image["output_type"] == "distmap":
                logging.info("Adding DistanceTransform for {}".format(input_image["field"]))
                trsfm.append(DistanceTransform(input_image["field"]))

        # Normalization
        for input_image in inputs:
            if input_image["output_type"] in ["image"]:
                logging.info("Adding MinMaxScaling for {}".format(input_image["field"]))
                trsfm.append(MinMaxScaling(input_image["field"]))

        for input_image in inputs:
            if input_image["output_type"] == "distmap":
                logging.info("Adding NormalizeDistanceMap for {}".format(input_image["field"]))
                trsfm.append(NormalizeDistanceMap(input_image["field"]))

        # NB: concatenate all input fields except the last one
        trsfm.append(Concatenate(in_fields=inputs_types[:-1], out_field=input_field))

        trsfm.extend([ToTensor(input_field),
                      ToTensor(output_field)])

        self.dataset.transform = transforms.Compose(trsfm)
        super(CommonDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                                               pin_memory)
