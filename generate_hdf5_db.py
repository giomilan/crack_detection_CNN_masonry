import numpy as np
from skimage.transform import resize
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
from subroutines.HDF5 import HDF5DatasetWriterMask

#Dimensions of the Images
IMAGE_DIMS=(224 , 224 , 3)
#Main Folder

#main_folder="C:\\Users\\filippo.giunta\\OneDrive - Arup\\Documents\\masonry_crack\\crack_detection_CNN_masonry\\img_to_evaluate\\"

main_folder= os.path.join(os.getcwd(),"img_to_evaluate")

#Folder containing the images that need to be evaluated by the ANN
folder_imgs="imgs"

#Black Mask dir
#mask_img_dir="masks\\black_mask.png"
mask_img_dir = os.path.join("masks","black_mask.png")

#Import black mask
mask_img=cv2.imread(os.path.join(main_folder,mask_img_dir),0)



#Output HDF5 file output
hdf5_output="val.hdf5"

number_of_imgs=len(os.listdir(os.path.join(main_folder,folder_imgs)))

print('Number of imgs in the folder: '+ str(number_of_imgs))


writer = HDF5DatasetWriterMask((number_of_imgs, IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2]), os.path.join(main_folder,hdf5_output))




images=[]
labels=[]

for name_img in os.listdir(os.path.join(main_folder,folder_imgs)):
    img = cv2.imread(os.path.join(main_folder,folder_imgs,name_img))
    if img is not None:
        images.append(img)
        labels.append(mask_img)


for image,mask in zip(images,labels):
    if IMAGE_DIMS != image.shape:
        image = resize(image, (IMAGE_DIMS), mode='constant', preserve_range=True)
    
    image=image / 255
    
    if IMAGE_DIMS[0:2] != mask.shape:
        mask = resize(mask, (IMAGE_DIMS[0], IMAGE_DIMS[1]), mode='constant', preserve_range=True)
        
    # normalize intensity values: [0,1]            
    mask = np.expand_dims(mask, axis=-1)
    mask = mask / 255

    # add the image and label to the HDF5 dataset
    writer.add([image], [mask])
    
try:
    writer.close()    
    print("HDF5 File created correctly!")
except:
    print("Error in creation HDF5 file!")
