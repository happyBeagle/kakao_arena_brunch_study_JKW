import pandas as pd 
import numpy as np
import os 
import sys
from tqdm import tqdm
from collections import defaultdict  


def check_file(path:str, file_type:str, call_func:str)->bool:
    
    if not os.path.exists(path):
        print(f"[Error] {call_func} : file do not exist ==> file name : {path}")
        return False
    
    f_name, f_type = os.path.splitext(path)

    if f_type != file_type:
        print(f"[Error] {call_func} : file type is not {file_type} ==> file type :{f_type}")
        return False
    return True