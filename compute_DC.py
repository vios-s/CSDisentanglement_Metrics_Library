import numpy as np
from metrics.distance_correlation import distance_correlation
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='', help='Path to the data file.')
parser.add_argument('--save', type=str, default='', help='Path to save the results.')
opts = parser.parse_args()

#Load the data directory and saving path
dir_root = opts.root + '/'
cont_root = 'content_test.npz'
sty_root = 'style_test.npz'
img_root = 'images_test.npz'
result_directory = opts.save+'DC_result.txt'
content = np.load(dir_root+cont_root)['arr_0']
style = np.load(dir_root+sty_root)['arr_0']
image = np.load(dir_root+img_root)['arr_0']

#Distance correlation test
print('Start distance correlation test on content and style...')
dis_correlation = distance_correlation(content.reshape(content.shape[0],-1), style)
print(dis_correlation)
print('Distance correlation test on content and style finished!')

print('Start distance correlation test on image and content...')
dis_correlation_img_cont = distance_correlation(content.reshape(content.shape[0],-1), image.reshape(image.shape[0],-1))
print(dis_correlation_img_cont)
print('Distance correlation test on image and content finished!')

print('Start distance correlation test on images and style...')
dis_correlation_img_sty = distance_correlation(style, image.reshape(image.shape[0],-1))
print(dis_correlation_img_sty)
print('Distance correlation test on image and style finished!')

#Save the results
file = open(result_directory, 'a')
file.write('\nDistance Correlation for ' + dir_root + ':\n')
file.write('content and style: ' + str(dis_correlation) + '\n')
file.write('Image and content: ' + str(dis_correlation_img_cont) + '\n')
file.write('Image and style: ' + str(dis_correlation_img_sty) + '\n')
file.close()