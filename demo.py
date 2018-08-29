#coding:utf-8
import model
from glob import glob
import numpy as np
from PIL import Image
import time
paths = glob('./test/*.*')

if __name__ =='__main__':
    for i in paths:
        print(i)
        im = Image.open(i)
        img = np.array(im.convert('RGB'))
        t = time.time()
        result,img,angle = model.model(img,model='keras', detectAngle=True) ## if model == crnn ,you should install pytorch
        print("It takes time:{}s".format(time.time()-t))
        print("---------------------------------------")
        print("图像的文字朝向为:{}度\n".format(angle),"识别结果:\n")

        for key in result:
            print(result[key][1])

        print('----------------------------------------')
    
