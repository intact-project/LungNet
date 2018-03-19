# LungNet
This is supplementary material for the manuscript: 

>"Semantic Segmentation of Pathological Lung Tissue with Dilated Fully Convolutional Networks"
M. Anthimopoulos, S. Christodoulidis, L. Ebner, A. Christe and S. Mougiakakou
IEEE Journal of Biomedical and Health infomatics (2018)
https://arxiv.org/abs/1803.06167

In case of any questions, please do not hesitate to contact us.

### Environment:
A `Dockerfile` is provided with all the necessary environment configurations. In order to build and run it you may use the following commands:

```
docker build -t lungnetenv .
docker run --name LungNet -it --rm -v "$PWD":/home lungnetenv /bin/bash
```

Some notes:
- the `--name` sets a name for the container for identification reasons.
- the `-it` flag denotes that the container will be interactive.
- the `--rm` flags means that when the container is stopped the container will also be deleted (i.e. docker start LungNet cannot be used). 
- the `-v` flag mounts the $PWD (current) directory of the host machine in the `/home` of the container (guest).

In order for GPU support the flag `--runtime=nvidia` can be used. For older versions of nvidia drivers the `nvidia-docker` should be used.

After the successful excecution of the `docker build` and `docker run` commands, a bash promt from within the docker container will be available.

### How to use:
There are three scripts with a `__main__` method:

1. `get_fmd_db.py`: This script will download a test database and generate a training and validation datasets. These will be saved in `.npz` format
2. `train.py`: This script will generate a model and train it using the two data files (`fmd-train.npz`, `fmd-val.npz`) generated from the `get_fmd_db.py`
3. `test.py`: This script loads a model and passes a sample through. *(Note: the directory of the model and weights should be defined in the file beforehand)*

Using the bash promt of the container these commands could be used:

```
#/ python get_fmd_db.py
#/ python train.py
#/ python test.py
```

***Important Note:** The demo uses the [Flickr Material database](https://people.csail.mit.edu/celiu/CVPR2010/FMD/) for demontration reasons, no particular effords were made for the optimization of the network for this task.* 

### Output:
The execution generates a folder for each run, which contains a `.png` file with the architecute of the CNN a log file with the metrics that were used along with the best snapshots of the model while training. The training loss and accuracy are also shown during training.

### Disclaimer:
Copyright (C) 2018  Marios Anthimopoulos, Stergios Christodoulidis, Stavroula Mougiakakou / University of Bern  


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
