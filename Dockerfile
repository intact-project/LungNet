# This is a part of the supplementary material uploaded along with 
# the manuscript:
#     "Semantic Segmentation of Pathological Lung Tissue with Dilated Fully Convolutional Networks"
#     M. Anthimopoulos, S. Christodoulidis, L. Ebner, A. Christe and S. Mougiakakou
#     IEEE Journal of Biomedical and Health infomatics (2018)
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# For more information please read the README file. The files can also 
# be found at: https://github.com/intact-project/LungNet
FROM nvidia/cuda:8.0-cudnn5-devel

# Installing system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
            python python-dev python-tk python-pip python-pydot graphviz \
            git g++ gcc gfortran cmake \
            libatlas-base-dev

# Installing general purpose python packages
RUN apt-get install -y --no-install-recommends python-numpy
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade cython
RUN pip install --upgrade glob2
RUN pip install --upgrade ipdb
RUN pip install --upgrade matplotlib
RUN pip install --upgrade pillow
RUN pip install --upgrade h5py

# Installing libgpuarray
RUN git clone https://github.com/Theano/libgpuarray.git && \
    cd libgpuarray && mkdir Build && cd Build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make && \
    make install && \
    cd .. && \
    python setup.py build && \
    python setup.py install
RUN rm -rf libgpuarray
RUN ldconfig

# Setting up project dependencies
RUN pip install --upgrade keras==2.1.5
RUN pip install --upgrade Theano==1.0.1

WORKDIR "/home"

RUN mkdir /root/.keras
RUN echo "{\"backend\": \"theano\", \"image_data_format\": \"channels_last\", \"image_dim_ordering\": \"th\", \"floatx\": \"float32\"}" > /root/.keras/keras.json
RUN echo "[global]\ndevice=cuda\nfloatX=float32\noptimizer_including=cudnn\nmode=FAST_RUN\nallow_gc=true\n[nvcc]\nfastmath=True" > /root/.theanorc
