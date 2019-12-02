########################################################################

# Describe what problem can be solved by the code defined here.
# Inform what functions and classes are included in this file.

# Author : Kuo Chun-Lin 		   / Xu LiangSheng
# E-mail : guojl19@tsinghua.org.cn / xuls18@mails.tsinghua.edu.cn
# Date   : 2019.11.23

########################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from collections import namedtuple
from easydict import EasyDict as edict

########################################################################

__C = edict()
cfg = __C

# To use the configs set here, do 'from "file_name" import cfg'

__C.DATA_NUM = 20000

__C.CLASS_NUM = 20

__C.ITERATION = 10

__C.BATCH_SIZE = 4

__C.PRIOR_VARIANCE = [10, 10, 5, 5]


__C.TRAIN = edict()

########################################################################

__C.TEST = edict()

########################################################################


if __name__ == '__main__':
    pass
