from skimage import io
from skimage.morphology import convex_hull_image
from skimage.util import invert
from skimage.transform import resize
import numpy as np
import glob



def resizer():
    paths = glob.glob('data/real_world_testset/images/*.jpg')
    for i,j in enumerate(paths):

        image = io.imread(j)

        invimg=invert(image)

        chull = convex_hull_image(invimg)


        [rows, columns, channel] = np.where(chull)
        row1 = min(rows)
        row2 = max(rows)
        col1 = min(columns)
        col2 = max(columns)

        newImage = image[row1:row2, col1:col2]


        resized= resize(newImage,(256,256,3), anti_aliasing=True, preserve_range=True,order=3)

        path='data/real_world_testset/images/mod/{:05d}.jpg'.format(i)
        io.imsave(path, resized,quality=100)

