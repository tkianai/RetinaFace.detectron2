'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-25 20:07:26
@FilePath       : /RetinaFace.detectron2/retinaface/data/dataset_mapper.py
@Description    : 
'''

from detectron2.data.dataset_mapper import DatasetMapper as BaseDatasetMapper
from . import detection_utils as d_utils

class DatasetMapper(BaseDatasetMapper):

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        self.tfm_gens = d_utils.build_transform_gen(cfg, is_train)
