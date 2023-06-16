#import libraries and modules needed
import os
import matplotlib
from matplotlib.figure import Figure
from scipy import integrate, linalg
from matplotlib import pyplot,axis
from os.path import normpath,join
import matplotlib.image as mpimg
import PIL
from PIL import Image
import cv2
import numpy as np

path1 = '/Users/nicolasdhers/Documents/McGill/U3/SURE/CNN PROJECT/Resources'

airfoil_filepath = os.path.join(path1 , 'goe195.dat') # delete title in text file and update path


with open(airfoil_filepath, 'r') as infile:

	x, y = np.loadtxt(infile, dtype=float, unpack=True)
	    

# plot geometry
width = 10
pyplot.figure(figsize=(width, width))
pyplot.grid(False)
pyplot.axis('off')
pyplot.plot(x, y, color='k', linestyle='-', linewidth=2)
#pyplot.axis('scaled', adjustable='box')
xmin, xmax, ymin, ymax = matplotlib.pyplot.axis('image')
#pyplot.xlim(-0.1, 1.1) # to update
#pyplot.ylim(-0.4, 0.4) # to update
pyplot.fill(x,y,'black')


pyplot.savefig('plotgoe195.png') # to update

#print(os.path.abspath('plot.png'))


Mach_max = 0.225
path = '/Users/nicolasdhers/Documents/McGill/U3/SURE/CNN PROJECT/Database'
#path = '/Users/nicolasdhers/Downloads/Test'
img_file = '/Users/nicolasdhers/Downloads/plotgoe195.png' # to update
img = cv2.imread(img_file, 0) # using 0 to force grayscale format and speed up running
img = cv2.resize(img,(500,500)) # resizing image to 500x500 to speed process up
rows,cols = img.shape

imgstd = img

for angle in range(-5,20,1):

	for Machnumber in np.arange(0,Mach_max,0.025):

		img = imgstd
		M = cv2.getRotationMatrix2D((cols/2,rows/2),-angle,1)
		rotated = cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_CONSTANT,borderValue=255)
		img = rotated

		for i in range(rows):
		    for j in range(cols):
		    	
		    	z = img[i,j]
		    	if z == 255:
		    		img[i,j] = Machnumber/Mach_max*255

		
		cv2.imwrite(os.path.join(path , 'goe195_{}_{}.png'.format(str(angle),str(Machnumber))), img) # to update



