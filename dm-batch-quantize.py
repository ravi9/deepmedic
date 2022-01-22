#!/usr/bin/env python
import pudb;
import os
import numpy as np
import json
import cv2 as cv
import glob
from addict import Dict
from math import ceil

import sys
from pathlib import Path
import pandas as pd
import pickle

from compression.api import Metric, DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.pipeline.initializer import create_pipeline
from compression.utils.logger import init_logger
import compression

os.environ["KMP_WARNINGS"] = "FALSE"

import argparse

dm_root = "examples/tcga/savedmodels/fp32/"

parser = argparse.ArgumentParser(
    description="Quantizes an OpenVINO model to INT8.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--xml_file", default=dm_root+"deepmedic-4-ov.model.ckpt.xml",
                    help="XML file for OpenVINO to quantize")
parser.add_argument("--bin_file", default=dm_root+"deepmedic-4-ov.model.ckpt.bin",
                help="BIN file for OpenVINO to quantize")
parser.add_argument("--manifest", default="/Share/ravi/upenn/data/tcga-deepmedic-batches-manifest-5rows.csv",
                help="Manifest file (CSV with filenames of images and labels)")
parser.add_argument("--data_dir", default="./data",
                help="Data directory root")
parser.add_argument("--int8_directory", default="./int8_openvino_model",
                help="INT8 directory for calibrated OpenVINO model")
parser.add_argument("--maximum_metric_drop", default=1.0,
                help="AccuracyAwareQuantization: Maximum allowed drop in metric")
parser.add_argument("--accuracy_aware_quantization",
                    help="use accuracy aware quantization",
                    action="store_true", default=False)

args = parser.parse_args()

class bcolors:
    """
    Just gives us some colors for the text
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class MyDataLoader(DataLoader):

    def __init__(self, config):

        super().__init__(config)

        """
        The assumption here is that a manifest file (CSV file) is
        passed to the object. The manifest contains a comma-separated
        list of the image filenames and label filenames for the validation
        dataset. You can modify this code to fit your needs. (For example,
        you could have the image filename and the class if using a
        classification model or the image filename and the object
        bounding boxes if using a localization model).
        """

        self.manifest = config["manifest"]  # Filename for manifest file with image and label filenames
        self.images = []
        self.labels = []

        dataset_df = pd.read_csv(self.manifest, header = None)

        for i, row in dataset_df.iterrows():
            self.images.append(row[0]) #image path
            self.labels.append(row[1]) #mask path

        self.items = np.arange(dataset_df.shape[0])
        self.batch_size = 1

        print(bcolors.UNDERLINE + "\nQuantizing FP32 OpenVINO model to INT8\n" + bcolors.ENDC)

        print(bcolors.OKBLUE + "There are {:,} samples in the test dataset ".format(len(self.items)) + \
            bcolors.OKGREEN + "{}\n".format(self.manifest) + bcolors.ENDC)



    def set_subset(self, indices):
        self._subset = None

    @property
    def batch_num(self):
        return ceil(self.size / self.batch_size)

    @property
    def size(self):
        return self.items.shape[0]

    def __len__(self):
        return self.size

    def myPreprocess(self, image_filename, label_filename):
        """
        Custom code to preprocess input data
        For this example, we show how to process the brain tumor data.
        Change this to preprocess you data as necessary.
        """


        """
        Load the image and label for this item
        """

        # Load feeds_dict_ov
        with open(image_filename, 'rb') as f:
            img = pickle.load(f)

        # Load gt_lbl_of_tiles_per_path
        with open(label_filename, 'rb') as f:
            msk = pickle.load(f)

        return img, msk

    def __getitem__(self, item):
        """
        Iterator to grab the data.
        If the data is too large to fit into memory, then
        you can have the item be a filename to load for the input
        and the label.

        In this example, we use the myPreprocess function above to
        do any custom preprocessing of the input.
        """

        # Load the iage and label files for this item
        image_filename = self.images[self.items[item]]
        label_filename = self.labels[self.items[item]]

        image, label = self.myPreprocess(image_filename, label_filename)

        # IMPORTANT!
        # OpenVINO expects channels first so transpose channels to first dimension
#         image = np.transpose(image, [3,0,1,2]) # Channels first
#         label = np.transpose(label, [3,0,1,2]) # Channels first

        return (item, label), image

class MyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.name = "custom Metric - Dice score"
        self._values = []
        self.round = 1

    @property
    def value(self):
        """ Returns accuracy metric value for the last model output. """
        return {self.name: [self._values[-1]]}

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """
        value = np.ravel(self._values).mean()
        print("Round #{}    Mean {} = {}".format(self.round, self.name, value))

        self.round += 1

        return {self.name: value}

    def calculate_dice(self, pred_seg, gt_lbl):
        union_correct = pred_seg * gt_lbl
        tp_num = np.sum(union_correct)
        gt_pos_num = np.sum(gt_lbl)
        dice = (2.0 * tp_num) / (np.sum(pred_seg) + gt_pos_num) if gt_pos_num != 0 else -1
        return dice

    def update(self, outputs, labels):
        """ Updates prediction matches.

        Args:
            outputs: model output
            labels: annotations

        Put your post-processing code here.
        Put your custom metric code here.
        The metric gets appended to the list of metric values
        """
        try:
            #pudb.set_trace()
            # out_val_of_ops = list(outputs.values())
            prob_maps_batch = outputs[0]

            # prob_maps_batch_0 = prob_maps_batch[:,0,:26,:26,:26]
            # dice0 = calculate_dice(prob_maps_batch_0, labels[0][0])

            prob_maps_batch_1 = prob_maps_batch[:,1,:26,:26,:26]
            dice1 = self.calculate_dice(prob_maps_batch_1, labels[0][1])

            self._values.append(dice1)
        except:
            print (f" Inference Failed. ")

    def reset(self):
        """ Resets collected matches """
        self._values = []

    @property
    def higher_better(self):
        """Attribute whether the metric should be increased"""
        return True

    def get_attributes(self):
        return {self.name: {"direction": "higher-better", "type": ""}}

model_config = Dict({
    "model_name": "resunet_ma",
    "model": args.xml_file,
    "weights": args.bin_file
})

engine_config = Dict({
    "device": "CPU",
    "stat_requests_number": 4,
    "eval_requests_number": 4
})

dataset_config = {
    "manifest": args.manifest,
    "images": "image",
    "labels": "label"
}

default_quantization_algorithm = [
    {
        "name": "DefaultQuantization",
        "params": {
            "target_device": "CPU",
            "preset": "performance",
            #"stat_subset_size": 10
        }
    }
]


accuracy_aware_quantization_algorithm = [
    {
        "name": "AccuracyAwareQuantization", # compression algorithm name
        "params": {
            "target_device": "CPU",
            "preset": "performance",
            "stat_subset_size": 10,
            "metric_subset_ratio": 0.5, # A part of the validation set that is used to compare full-precision and quantized models
            "ranking_subset_size": 300, # A size of a subset which is used to rank layers by their contribution to the accuracy drop
            "max_iter_num": 10,    # Maximum number of iterations of the algorithm (maximum of layers that may be reverted back to full-precision)
            "maximal_drop": args.maximum_metric_drop,      # Maximum metric drop which has to be achieved after the quantization
            "drop_type": "absolute",    # Drop type of the accuracy metric: relative or absolute (default)
            "use_prev_if_drop_increase": True,     # Whether to use NN snapshot from the previous algorithm iteration in case if drop increases
            "base_algorithm": "DefaultQuantization" # Base algorithm that is used to quantize model at the beginning
        }
    }
]

class GraphAttrs(object):
    def __init__(self):
        self.keep_quantize_ops_in_IR = True
        self.keep_shape_ops = False
        self.data_type = "FP32"
        self.progress = False
        self.generate_experimental_IR_V10 = True
        self.blobs_as_inputs = True
        self.generate_deprecated_IR_V7 = False


model = load_model(model_config)

data_loader = MyDataLoader(dataset_config)
metric = MyMetric()


engine = IEEngine(engine_config, data_loader, metric)

if args.accuracy_aware_quantization:
    # https://docs.openvinotoolkit.org/latest/_compression_algorithms_quantization_accuracy_aware_README.html
    print(bcolors.BOLD + "Accuracy-aware quantization method" + bcolors.ENDC)
    pipeline = create_pipeline(accuracy_aware_quantization_algorithm, engine)
else:
    print(bcolors.BOLD + "Default quantization method" + bcolors.ENDC)
    pipeline = create_pipeline(default_quantization_algorithm, engine)


metric_results_FP32 = pipeline.evaluate(model)

compressed_model = pipeline.run(model)

compression.graph.model_utils.save_model(compressed_model, save_path=args.int8_directory, model_name="deepmedic-4-ov.model.ckpt.int8", for_stat_collection=False)

print(bcolors.BOLD + "\nThe INT8 version of the model has been saved to the directory ".format(args.int8_directory) + \
    bcolors.HEADER + "{}\n".format(args.int8_directory) + bcolors.ENDC)

#save_model(compressed_model, "./int8_openvino_model/")

print(bcolors.BOLD + "\Evaluating INT8 Model..." + bcolors.ENDC)

metric_results_INT8 = pipeline.evaluate(compressed_model)

# print metric value
if metric_results_FP32:
    for name, value in metric_results_FP32.items():
        print(bcolors.OKGREEN + "{: <27s} FP32: {}".format(name, value) + bcolors.ENDC)

if metric_results_INT8:
    for name, value in metric_results_INT8.items():
        print(bcolors.OKBLUE + "{: <27s} INT8: {}".format(name, value) + bcolors.ENDC)


