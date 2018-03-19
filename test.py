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
'''
import numpy as np
import custom_metrics as M
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
from keras.models import model_from_json
from custom_layers import BatchNormalization, Softmax4D

from ipdb import set_trace as bp

if __name__ == '__main__':
    modeldir = 'model-4563c5e2-8dc6-4d1f-8366-b5ccf9e027c0' # change this with the correct model directory
    weightsf = 'weights.11-0.16.hdf5' # change this with the corresponding weights file in the modeldir

    # loads the model
    model = model_from_json(open(modeldir+'/architecture.json').read(), 
                            custom_objects={'BatchNormalization':BatchNormalization, 'Softmax4D':Softmax4D})
    model.load_weights(modeldir+'/'+weightsf)

    # loads the data
    db = np.load('fmd-val.npz')['db'][()]

    # Forward pass
    X = db['X'][np.random.choice(len(db['X']))]
    Y_pred = model.predict(X[None,:,:,:])
    Y_lbls = np.argmax(Y_pred,axis=-1)

    all_labels = ['fabric', 'foliage', 'glass', 'leather', 'metal',
                  'paper', 'plastic', 'stone', 'water', 'wood', 'background']
    colors = np.asarray([[141,211,199,255],
                         [255,255,179,255],
                         [190,186,218,255],
                         [251,128,114,255],
                         [128,177,211,255],
                         [253,180,98,255],
                         [179,222,105,255],
                         [252,205,229,255],
                         [217,217,217,255],
                         [188,128,189,255],
                         [204,235,197,255]]) / 255.

    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(X),cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(colors[np.squeeze(Y_lbls)])
    plt.legend(handles=[mpatches.Patch(color=colors[i], label=all_labels[i]) for i in np.unique(Y_lbls)], prop={'size':6})
    plt.show()