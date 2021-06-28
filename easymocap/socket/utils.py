'''
  @ Date: 2021-05-24 20:07:34
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-28 12:05:35
  @ FilePath: /EasyMocapRelease/easymocap/socket/utils.py
'''
import cv2
import numpy as np
from ..mytools.file_utils import write_common_results

def encode_detect(data):
    res = write_common_results(None, data, ['keypoints3d'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')

def encode_smpl(data):
    res = write_common_results(None, data, ['poses', 'shapes', 'expression', 'Rh', 'Th'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')

def encode_image(image):
    fourcc = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    #frame을 binary 형태로 변환 jpg로 decoding
    result, img_encode = cv2.imencode('.jpg', image, fourcc)
    data = np.array(img_encode) # numpy array로 안바꿔주면 ERROR
    stringData = data.tostring()
    return stringData