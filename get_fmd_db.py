'''
This is a part of the supplementary material uploaded along with 
the manuscript:
    "Semantic Segmentation of Pathological Lung Tissue with Dilated Fully Convolutional Networks"
    M. Anthimopoulos, S. Christodoulidis, L. Ebner, A. Christe and S. Mougiakakou
    IEEE Journal of Biomedical and Health infomatics (2018)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

For more information please read the README file. The files can also 
be found at: https://github.com/intact-project/LungNet

This script will download the Flickr Material database
https://people.csail.mit.edu/celiu/CVPR2010/FMD/ and
it will generate two npz files with the training and 
validation data.
'''
import glob
import os.path
import urllib2
import zipfile
import numpy as np
import gzip, pickle
from PIL import Image
from ipdb import set_trace as bp

def download_and_unzip_from_url(url,directory):
    # modified version of https://stackoverflow.com/a/22776
    file_name = url.split('/')[-1]
    if not os.path.isfile(file_name):
        u = urllib2.urlopen(url)
        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        print "Downloading: %s Bytes: %s" % (file_name, file_size)

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print status,

        f.close()

    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(directory)
    zip_ref.close()
    
    return file_name


if __name__ == '__main__':
    # get the database and extract it
    filename = download_and_unzip_from_url("https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip", './database')
    
    # list the classes, image dirs and mask dirs
    classes = [c.split('/')[-1] for c in glob.glob('./database/image/*')]
    images = sorted(glob.glob('./database/image/*/*.jpg'))
    masks = sorted(glob.glob('./database/mask/*/*.jpg'))
    cdict = {k:i for i,k in enumerate(classes)}

    # generate the dataset
    X = []
    Y = []
    for i in range(len(images)):
        # loading and resizing the images for a faster demo
        img = Image.open(images[i])
        mask = Image.open(masks[i]).convert('L')
        im = np.asarray(img.resize((192,256)), dtype='float32') / 255
        msk = np.asarray(mask.resize((192,256), resample=Image.NEAREST), dtype='float32') / 255
        cmsk = np.zeros( (im.shape[0], im.shape[1], len(cdict)+1), dtype='bool' ) # +1 for the background
        cmsk[:,:,cdict[images[i].split('/')[-1].split('_')[0]]] = msk
        cmsk[:,:,-1] = np.logical_not(msk)
        if im.ndim != 3:
            im = np.repeat(im[:,:,None], 3, axis=-1) # some images in FMD are b&w
        X.append(im)
        Y.append(cmsk)
        
    # shuffle
    idx = np.random.permutation(len(X))
    X = [X[i] for i in idx]
    Y = [Y[i] for i in idx]
    
    # split and save to pklz
    print('Saving validation set ...')
    valdb = {'X': X[int(len(X)*0.8):], 'Y': Y[int(len(X)*0.8):]}
    np.savez_compressed('fmd-val', db=valdb)
    print('Saving training set ...')
    traindb = {'X': X[:int(len(X)*0.8)], 'Y': Y[:int(len(X)*0.8)]}
    np.savez_compressed('fmd-train', db=traindb)

    # use this in order to load:
    # db = np.load('fmd-val.npz')['db'][()]

