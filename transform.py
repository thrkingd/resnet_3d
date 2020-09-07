from PIL import Image
import cv2
import numpy as np

def group_colorjitter(img_group,value):
    #获取四个扰动参数
    jitter_para=[]
    for i in range(3):
        jitter_para.append(np.random.uniform(max(0, 1 - value), 1 + value))
    jitter_para.append(np.random.uniform(-value,value))
    img_out=[]
    for img in img_group:
        img = np.array(img).astype(np.float32)
        img = BrightnessTransform(img,jitter_para[0])
        img = ContrastnessTransform(img,jitter_para[1])
        img = SaturationTransform(img,jitter_para[2])
        img = HueTransform(img,jitter_para[3])
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img_out.append(img)
    return img_out


def BrightnessTransform(img, alpha):    
    img = img * alpha
    return img.clip(0, 255)
  


def ContrastnessTransform(img, alpha):
    img = img * alpha + cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean() * (
        1 - alpha)
    return img.clip(0, 255)

def  SaturationTransform(img, alpha):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = gray_img[..., np.newaxis]
    img = img * alpha + gray_img * (1 - alpha)
    return img.clip(0, 255)

def HueTransform(img,value):
    dtype = img.dtype
    img = img.astype(np.uint8)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    h, s, v = cv2.split(hsv_img)

    alpha = np.random.uniform(-value,value)
    h = h.astype(np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        h += np.uint8(alpha * 255)
    hsv_img = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR_FULL).astype(dtype)


