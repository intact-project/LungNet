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
from keras import backend as K

from ipdb import set_trace as bp

class loss(object):
    '''
    Weighted categorical cross entropy over annotation mask plus entropy over non annotated regions.
    '''
    def __init__(self, weights, unsuper_weight=0, unsuper_channel=-1):
        self.config = {'weights': list(weights), 'unsuper_weight':unsuper_weight, 'unsuper_channel':unsuper_channel}
        self.unsuper_weight = K.variable(unsuper_weight)
        self.supervised_loss = wcceOA(weights)
        self.unsupervised_loss = entrONA(unsuper_channel)
        self.__name__ = 'wlossOA'

    def __call__(self, y_true, y_pred):        
        return (1-self.unsuper_weight)*self.supervised_loss(y_true, y_pred) + self.unsuper_weight*self.unsupervised_loss(y_true, y_pred)        

    def get_config(self):
        config = self.config
        return config

class wcceOA(object):
    '''
    Weighted categorical cross entropy over annotation mask
    '''
    def __init__(self, weights):
        self.config = {'weights': list(weights)}
        self.length = len(weights)
        self.weights = K.variable(weights, name='loss_weights')
        self.__name__ = 'wcceOA'

    def __call__(self, y_true, y_pred):
        weights_mask = K.zeros_like(y_true[:,:,:,0])
        for c in range(self.length):
            weights_mask += self.weights[c] * K.cast(y_true[:,:,:,c]>0,K.floatx())
        self.weights_mask = weights_mask
        
        wcce = K.sum(-K.sum(y_true * K.log(y_pred), axis=-1)*self.weights_mask) / K.sum(self.weights_mask+1e-8)

        return wcce

    def get_config(self):
        config = self.config
        return config

class entrONA(object):
    '''
    Entropy over non annotated regions (unsupervised).
    '''
    def __init__(self, unsuper_channel=-1):
        self.config = {'unsuper_channel': unsuper_channel}
        self.__name__ = 'entrONA'
        self.unsuper_channel = K.variable(unsuper_channel, dtype='int')

    def __call__(self, y_true, y_pred):
        unsupermask = y_true[:,:,:,self.unsuper_channel]
        lu = (-K.sum((y_pred ) * K.log(y_pred), axis=-1)) * unsupermask
        return K.sum(lu)/( K.sum( unsupermask )+1e-8 )


class waccOA(object):
    '''
    Weighted accuracy over annotation mask.
    '''
    def __init__(self, weights):
        self.config = {'weights': list(weights)}
        self.length = len(weights)
        self.weights = K.variable(weights, name='acc_weights')
        self.__name__ = 'waccOA'

    def __call__(self, y_true, y_pred):
        weights_mask = K.zeros_like(y_true[:,:,:,0])
        for c in range(self.length):
            weights_mask += self.weights[c] * K.cast(y_true[:,:,:,c]>0,K.floatx())
        self.weights_mask = weights_mask
        
        true_lbls_f = K.argmax(y_true, axis=-1)
        pred_lbls_f = K.argmax(y_pred, axis=-1)
        # count of the correcly classified pixels over mask
        eq = K.cast(K.equal(true_lbls_f,pred_lbls_f),K.floatx())*self.weights_mask
        nom = K.sum(eq)
        den = (K.sum(self.weights_mask)+1e-8)
        return nom/den

    def get_config(self):
        config = self.config
        return config