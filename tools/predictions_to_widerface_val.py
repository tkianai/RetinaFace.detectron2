'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-22 10:35:43
@FilePath       : /RetinaFace.detectron2/tools/predictions_to_widerface_val.py
@Description    : Convert detectron2 result to widerface evaluation style
'''

import os
import os.path as osp
import argparse
import pickle
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Translate predictions to widerface eval style")
    parser.add_argument('--res', type=str, help="Output results from detectron2")
    parser.add_argument('--save', type=str, default='./output/widerface/val')
    
    args = parser.parse_args()
    if not osp.exists(args.save):
        os.makedirs(args.save)

    return args


def main(args):
    save_root = args.save
    predictions = pickle.load(open(args.res, 'rb'))
    # parse predictions
    for imgpath, prediction in predictions.items():
        d_name, imgname = imgpath.split('/')[-2:]
        imgname = imgname.split('.')[0]
        save_dir = osp.join(save_root, d_name)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        with open(osp.join(save_dir, imgname + '.txt'), 'w') as w_obj:
            # Save imgname
            w_obj.write(imgname + '\n')
            # Save instances num
            instances = prediction['instances']
            num_instances = len(instances)
            w_obj.write(str(num_instances) + '\n')
            # Save [x,y,w,h,score]
            for i in range(num_instances):
                bbox = instances[i].pred_boxes.tensor.squeeze(0).tolist()
                score = instances[i].scores[0].item()
                item = [
                    bbox[0], bbox[1],
                    bbox[2] - bbox[0], bbox[3] - bbox[1],
                    score
                ]
                w_obj.write(" ".join([str(e) for e in item]) + '\n')

if __name__ == "__main__":
    args = parse_args()
    main(args)