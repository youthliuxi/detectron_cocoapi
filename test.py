# -*- coding:utf-8 -*-
import skimage.io as io
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

#第一步，读取文件
filePath = "../caffe/bak/annotations/stuff_train2017.json"
coco = COCO(filePath)

def showImage(imgIds):
	#第二步，得到相关ID
	imgIds = coco.getImgIds(imgIds)

	#第三步，通过ID读取图片信息
	Imgs = coco.loadImgs(imgIds)[0]
	
	#第四步，定义图片文件夹位置imageFile，传入图片位置imageUrl，读取图片
	imageFile = "../caffe/bak/train2017/"
	imageUrl = imageFile + Imgs["file_name"]
	#print(Imgs['file_name'])
	I = io.imread(imageUrl)
	
	#第五步，调用plt打印图片
	plt.imshow(I)
	plt.show()

	return Imgs,I

catIds = coco.getCatIds(catNms=["person"])
imgIds = coco.getImgIds(catIds=catIds)
print('imgIds:', imgIds[3])
img,I = showImage(imgIds[3])
img
plt.imshow(I)
annIds = coco.getAnnIds(img['id'])
imgAnns = coco.loadAnns(ids=annIds)
#证明一张图片对应了很多的annations，每个annations都对应不同的类别号
#for ann in imgAnns:
#	print(ann)
coco.showAnns(imgAnns)
plt.show()
