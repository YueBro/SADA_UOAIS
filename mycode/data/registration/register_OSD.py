"""
This script serves the purpose of OSD dataset registration.

Two methods to register (in each time running a program...):
    1. Call "import mycode.data.registration" at the top of
       program, then it's registered successfully.
    2. Import "register_OSD" function below, and call the 
       function in the program.
"""

from mycode.data.dataloader.OSD_dataloader import load_train as load_train_OSD
from mycode.data.dataloader.OSD_dataloader import load_val as load_val_OSD
from mycode.data.dataloader.OSD_dataloader import load_all as load_all_OSD
from detectron2.data import DatasetCatalog

# Registration for train set
def register_OSD():
    DatasetCatalog.register("OSD_train", lambda: load_train_OSD(root="datasets/OSD-0.2-depth", re_calculate=False))
    DatasetCatalog.register("OSD_val", lambda: load_val_OSD(root="datasets/OSD-0.2-depth", re_calculate=False))
    DatasetCatalog.register("OSD_all", lambda: load_all_OSD(root="datasets/OSD-0.2-depth", re_calculate=False))

register_OSD()
