"""Test.py"""

import os
import sys


# insert path

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from confs import logconf
logger = logconf.Logger(__file__).logger


logger.debug('test')
