from typing import Optional
import os

USE_FASTPT_CUDA = os.getenv('USE_FASTPT_CUDA', '0') == '1'
__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']
__version__ = '2.12.0'
debug = False
if USE_FASTPT_CUDA:
  cuda: Optional[str] = '12.6'
  hip: Optional[str] = None
else:
  hip: Optional[str] = '6.3.26105'
  cuda: Optional[str] = None
git_version = 'f6b6f4bcf77b708ae46cf075ea0f0e113c98c130'
__hcu_version__ = '2.12.0+das.opt1.dtk2604'
dtk = None
