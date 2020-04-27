'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-21 12:54:52
@FilePath       : /RetinaFace.detectron2/retinaface/config/config.py
@Description    : 
'''

from detectron2.config import CfgNode


def get_cfg() -> CfgNode:
    """Get a copy of the default config

    Returns:
        CfgNode -- a detectron2 CfgNode instance
    """

    from .defaults import _C
    return _C.clone()
