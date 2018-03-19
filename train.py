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
import os
import glob
import uuid
import numpy as np
import LungNet as LN
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# debug
from ipdb import set_trace as bp

if __name__ == '__main__':

    # calculated on FMD train dataset np.median(freq)/freq (the extra 0 at the end is the weight for the BG)
    class_weights = [ 0.94058391, 0.92470112, 1.02412472, 0.99289015, 0.95592841, 
                      1.16978173, 1.22464036, 1.23952882, 0.86262530, 1.00721240, 0 ]

    # get model
    model = LN.get_model(11, class_weights, unsuper_weight=0.1)

    # save model image
    modeldir = 'model-'+str(uuid.uuid4())
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    open(modeldir+'/architecture.json', 'w').write(model.to_json())
    plot_model(model, to_file=modeldir+'/model.png', show_shapes=True)

    train_gen = LN.sample_generator('fmd-train.npz')
    val_gen = LN.sample_generator('fmd-val.npz', augment=False)

    checkpoints = ModelCheckpoint(modeldir+'/weights.{epoch:02d}-{val_waccOA:.2f}.hdf5', 
                                  monitor='val_waccOA', verbose=1, mode='max',
                                  save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_waccOA', min_delta=0.01, patience=20, 
                                  verbose=1, mode='max')
    logger = CSVLogger(modeldir+'/training.log')

    # fit model
    model.fit_generator(train_gen, 800, 1000, validation_data=val_gen, validation_steps=200,
                        callbacks=[checkpoints, earlystopping, logger])
