import numpy as np

def placeMaskOnImg(img, mask):
    color = [255, 0, 255]
    color = [i/255.0 for i in color]
    np.place(img[:, :, :], mask[:, :, :] >= 0.5, color)
    return img

def make_pred_good(pred):
    pred = pred[0][:, :, :]
    pred = np.repeat(pred, 3, 2)
    return pred