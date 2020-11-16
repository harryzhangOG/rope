import os
import natsort
import sys
import cv2
import numpy as np

directory = 'repeatability_vault/seg'
pic_name = 'repeat_result.png'
filelist = os.listdir(directory)
pics = []
for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
    if not(fichier.endswith('.png')):
        filelist.remove(fichier)
for i, filename in enumerate(natsort.natsorted(filelist), 0):
    if filename.endswith('.png') and 'end' in filename and (filename[-10] == '0' or filename[-10] == '1'):
    #if filename.endswith('.png'):
        print(i)
        pics.append(cv2.imread(os.path.join(directory, filename), 1))

dst = np.zeros_like(pics[0], np.uint8)
for i, p in enumerate(pics):
    if i == 0:
        dst = cv2.addWeighted(dst, 1, p, 0.6, 0)
    else:
        dst = cv2.addWeighted(dst, 1, p, 0.6, 0)
#dst = pics[0]
#for i in range(len(pics)):
#    if i == 0:
#        pass
#    else:
#        alpha = 1/(i + 1)
#        beta = 1*(1.0 - alpha)
#        dst = cv2.addWeighted(pics[i], alpha, dst, beta, 0.0)
 
# Save blended image
dst[:, :150, :]  = 0
cv2.imwrite(os.path.join(directory, pic_name), dst)


