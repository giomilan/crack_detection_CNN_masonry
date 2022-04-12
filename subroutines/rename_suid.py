import numpy 
import pandas
import shutil
import random 
# 2nd option

import os




my_dir = "C:/Users/Giovanni.Milan/OneDrive - Arup/cracking_ML/dataset_tom_rename/crack_detection_224_masks"
new = "C:/Users/Giovanni.Milan/OneDrive - Arup/cracking_ML/dataset_tom_rename2/crack_detection_224_masks"
for filename in os.listdir(my_dir):
    shutil.copy(os.path.join(my_dir, filename), os.path.join(new, filename)) 
    
    """#add mat in front
    if(filename.split("_")[0][0] == "S" or filename.split("_")[0][0] == "X"):
        os.rename(os.path.join(new, filename), os.path.join(new, "clay_" + filename))"""
        
    #add brickseize
    if(len(filename.split("_")) == 2):
        os.rename(os.path.join(new, filename), os.path.join(new, filename.split("_")[0]+"_"+filename.split("_")[1].split(" ")[0]+"_200"+ filename.split("_")[1].split(" ")[1] ))
        
    
    """if(filename.split("_")[1][0] == "S" or filename.split("_")[1][0] == "X"):
        a = list(filename.split("_")[1])
        random.seed(0)
        random.shuffle(a)
        if(len(filename.split("_")) == 2): 
            os.rename(os.path.join(new, filename), os.path.join(new, filename.split("_")[0] + "_Bldg" + ''.join(a) ))
        else:
            os.rename(os.path.join(new, filename), os.path.join(new, filename.split("_")[0] + "_Bldg" + ''.join(a) + "_" + filename.split("_")[2]))
    print(filename)
    #with open(os.path.join(my_dir, filename), 'r') as f:
    #   text = f.read()
    #   print(text)
    """
    