import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from keras import backend as K
import albumentations as A


def modify_mask(mask):
    mask = np.expand_dims(mask, axis = 2)
    t_mask = np.zeros(mask.shape)
    np.place(t_mask[:, :, 0], mask[:, :, 0] >=100, 1)
    return t_mask

def map_function(img, mask):
    IMG_SIZE = (256, 256)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Blur(blur_limit = 5, p = 0.85), 
        # A.RandomCrop(height = 512, width = 512, p = 1)
    ])
    img, mask = plt.imread(img.decode()), plt.imread(mask.decode())
    img = cv.resize(img, IMG_SIZE)
    mask = modify_mask(cv.resize(mask, IMG_SIZE))
    
    img = img/255.0
    transformed = transform(image=img, mask=mask)
    img = transformed['image']
    mask = transformed['mask']

#     mask = modify_mask(mask)
    
    return img.astype(np.float64), mask.astype(np.float64)

def placeMaskOnImg(img, mask):
    color = [255, 0, 255]
    color = [i/255.0 for i in color]
    np.place(img[:, :, :], mask[:, :, :] >= 0.5, color)
    return img

def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0) 
 
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def make_pred_good(pred):
#     pred = pred.numpy()
    pred = pred[0][:, :, :]
    pred = np.repeat(pred, 3, 2)
    return pred