import os
import torch
import random
import pickle
import argparse
import numpy as np
from yolox.exp import get_exp
from yolox.core import launch
from yolox.utils import fuse_model
import torch.backends.cudnn as cudnn
from yolox.evaluators import DetEvaluator
from torch.nn.parallel import DistributedDataParallel as DDP

#Import additional packages for task 3
import supervision as sv
from inference import get_model
from PIL import Image
from io import BytesIO
import requests
import json
# from rfdetr_wrapper import RFDETRWrapper
from rfdetr import RFDETRBase
import warnings
warnings.filterwarnings("ignore", message=".*CoreMLExecutionProvider.*")
from collections import defaultdict
from array import *


def make_parser():
    parser = argparse.ArgumentParser("YOLOX")

    # Can be changed
    parser.add_argument("-f", "--exp_file", default="exps/yolox_x_mot17_test.py",
                        type=str, help="pls input your experiment description file",)
    # parser.add_argument("-c", "--ckpt", default="weights/mot17.pth.tar",
    #                     type=str, help="ckpt for eval")
    parser.add_argument("-n", "--exp_name", type=str,
                        default="../outputs/1. det/mot17_test_0.95_original.pickle")

    # Fixed
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")
    parser.add_argument("--fuse", dest="fuse", default=True, action="store_true", help="Fuse conv and bn",)
    parser.add_argument("--fp16", dest="fp16", default=True, action="store_true",)

    # distributed
    parser.add_argument("-t", "--type", default=None, type=str)
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training",)
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model",)
    parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)
    parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
    parser.add_argument("opts", help="Modify config options", default=None, nargs=argparse.REMAINDER,)

    # det args
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.95, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--min_box_area", default=100, type=int, help="filter out tiny boxes")
    parser.add_argument("--seed", default=10000, type=int, help="eval seed")

    return parser


def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True

    rank = args.local_rank

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = RFDETRBase() #Replaced with rfdetr model
    
    # val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)
    
    #Get image locations
    with open("jsons/mot17_val.json", 'r') as f:
        data = json.load(f)
    
    mot_groups = defaultdict(list)
    for item in data["images"]:
        #Get mot_id
        mot_id = item['file_name'].split("/")[2]

        mot_groups[mot_id].append('../../dataset/' + item['file_name']) #File path
    
    #Debugging mot_groups
    # for item in mot_groups:
    #     print(item)
    #     for a in mot_groups[item]:
    #         print('\t', a)

    #Predict using RFDETRWrapper
    test_path = mot_groups['MOT17-04-FRCNN'][0]
    print('test_id:', test_path)

    # test_img = Image.open(test_path)
    # print('Processing image:', path)
    # predictions = model.predict(test_img, threshold=0.5)
    # print('Predictions for test image', predictions)
    # print('Predictions xyxy:', predictions.xyxy)
    # print('Predictions confidence:', predictions.confidence)
    # print('Predictions class_id:', predictions.class_id)

    #Create dictionary of dictionary to match original output
    output_dict = defaultdict(dict)

    for mot_id, img_paths in mot_groups.items():
        print('Processing MOT ID:', mot_id)
        #Create a list to hold all predictions for this mot_id

        for path in img_paths:
            # Extract frame number from filename
            frame_num = os.path.splitext(os.path.basename(path))[0]  # e.g. '000123'
            frame_num = str(int(frame_num))  # remove leading zeros and make string

            #Predict using RFDETRWrapper
            # print('Processing image:', path)
            img = Image.open(path)
            
            predictions = model.predict(img, threshold=0.5)
            # print('Predictions for test image', predictions)
            # print('Predictions xyxy:', predictions.xyxy)
            # print('Predictions confidence:', predictions.confidence)
            # print('Predictions class_id:', predictions.class_id)
            output_dict[mot_id][frame_num] = np.array(predictions.xyxy, dtype=np.float32)
    
    print('output')
    for i in output_dict:
        # print(i, output_dict[i])
        for j in output_dict[i]:
            print('\t', output_dict[i][j])



    # evaluator = DetEvaluator(args=args, dataloader=val_loader, img_size=exp.test_size, conf_thresh=exp.test_conf,
                            #  nms_thresh=exp.nmsthre, num_classes=exp.num_classes,)

    # start evaluate, x1y1x2y2
    # det_results = evaluator.detect(model, args.fp16)

    with open(args.exp_name, 'wb') as f:
        pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)
    exp.merge(args.opts)

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
