"""
This script serves the purpose of TLESS dataset registration.

Two methods to register (in each time running a program...):
    1. Call "from mycode.data.registration import register_TLESS"
       at top of program, then it's registered successfully.
    2. Import "register_TLESS" function below, and call the 
       function in the program.
"""

from mycode.data.dataloader.TLESS_dataloader import load_train as load_train_TLESS
from mycode.data.dataloader.TLESS_dataloader import load_val as load_val_TLESS
from detectron2.data import DatasetCatalog

# Registration for train set
def register_TLESS():
    DatasetCatalog.register("TLESS_train", lambda: load_train_TLESS(root="datasets/T-LESS", re_calculate=False))
    DatasetCatalog.register("TLESS_val", lambda: load_val_TLESS(root="datasets/T-LESS", re_calculate=False))

register_TLESS()
