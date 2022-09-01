from .bop_dataloader import *
from .uoais_dataloader import *
from .ocid_dataloader import *
from .osd_dataloader import *


import os


__all__ = [
    "dataloader_func"
]


def _get_dataloader_func():
    dataloader_func = {}

    if os.path.exists("datasets/UOAIS-Sim"):
        dataloader_func["UOAIS"] = UOAIS_Dataloader().get_path_data
        dataloader_func["UOAIS-Sim"] = dataloader_func["UOAIS"]
    
    if os.path.exists("datasets/T-LESS"):
        dataloader_func["TLESS"] = BOP_Dataloader('datasets/T-LESS').get_path_data
        dataloader_func["T-LESS"] = dataloader_func["TLESS"]

    if os.path.exists("datasets/TYOL/test"):
        dataloader_func["TYOL"] = BOP_Dataloader('datasets/TYOL/test').get_path_data

    if os.path.exists("datasets/HOPE"):
        dataloader_func["HOPE"] = BOP_Dataloader('datasets/HOPE/val').get_path_data
    
    if os.path.exists("datasets/OCID-dataset"):
        dataloader_func["OCID"] = OCID_Dataloader('dataset/OCID-dataset').get_path_data
    
    if os.path.exists("datasets/OSD-0.2-depth"):
        dataloader_func["OSD"] = OSD_Dataloader('dataset/OSD-0.2-depth').get_path_data
    
    return dataloader_func

dataloader_func = _get_dataloader_func()
