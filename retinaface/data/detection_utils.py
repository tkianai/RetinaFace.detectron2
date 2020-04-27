'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-25 19:55:17
@FilePath       : /RetinaFace.detectron2/retinaface/data/detection_utils.py
@Description    : 
'''

import detectron2.data.transforms as T
import logging


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    logger = logging.getLogger("detectron2.data.detection_utils")
    tfm_gens = []
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        tfm_gens.append(T.RandomContrast(0.5, 1.5))
        tfm_gens.append(T.RandomBrightness(0.5, 1.5))
        tfm_gens.append(T.RandomSaturation(0.5, 1.5))
        tfm_gens.append(T.RandomFlip())
        logger.info("TransformGens used in training[Updated]: " + str(tfm_gens))
    return tfm_gens
