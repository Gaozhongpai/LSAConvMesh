

# Learning Local Neighboring Structure for Robust 3D Shape Representation
![PaiNeural3DMM architecture](images/architecture.png "PaiNeural3DMM architecture")
This repository is the official implementation of my paper: "Learning Local Neighboring Structure for Robust 3D Shape Representation"
# Project Abstract 
Mesh is a powerful data structure for 3D shapes. Representation learning for 3D meshes is important in many computer vision and graphics applications. The recent success of convolutional neural networks (CNNs) for structured data (e.g., images) suggests the value of adapting insight from CNN for 3D shapes. However, 3D shape data are irregular since each node's neighbors are unordered. Various graph neural networks for 3D shapes have been developed with isotropic filters or predefined local coordinate systems to overcome the node inconsistency on graphs. However, isotropic filters or predefined local coordinate systems limit the representation power. In this paper, we propose a local structure-aware anisotropic convolutional operation (LSA-Conv) that learns adaptive weighting matrices for each node according to the local neighboring structure and performs shared anisotropic filters. In fact, the learnable weighting matrix is similar to the attention matrix in random synthesizer -- a new Transformer model for natural language processing (NLP). Comprehensive experiments demonstrate that our model produces significant improvement in 3D shape reconstruction compared to state-of-the-art methods. 

[Arxiv link](https://arxiv.org/abs/2004.09995)

![Pai-Conv](images/pai-gcn.png "Pai-Conv operation")

![Results](images/results.png "Results")

# Repository Requirements

This code was written in Pytorch 1.4. We use tensorboardX for the visualisation of the training metrics. We recommend setting up a virtual environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html). To install Pytorch in a conda environment, simply run 

```
$ conda install pytorch torchvision -c pytorch
```

Then the rest of the requirements can be installed with 

```
$ pip install -r requirements.txt
```

### Mesh Decimation
For the mesh decimation code we use a function from the [COMA repository](https://github.com/anuragranj/coma) (the files **mesh_sampling.py** and **shape_data.py** - previously **facemesh.py** - were taken from the COMA repo and adapted to our needs). In order to decimate your template mesh, you will need the [MPI-Mesh](https://github.com/MPI-IS/mesh) package (a mesh library similar to Trimesh or Open3D). 


# Data Organization

The following is the organization of the dataset directories expected by the code:

* data **root_dir**/
  * **dataset** name/ (eg DFAUST)
    * template
      * template.obj
      * downsample_method/
        * downsampling_matrices.pkl (created by the code the first time you run it)
    * preprocessed/
      * sliced
        * train.npy (number_meshes, number_vertices, 3) (no Faces because they all share topology)
        * test.npy 
        * points_train/ (created by data_generation.py)
        * points_val/ (created by data_generation.py)
        * points_test/ (created by data_generation.py)
        * paths_train.npy (created by data_generation.py)
        * paths_val.npy (created by data_generation.py)
        * paths_test.npy (created by data_generation.py)

# Usage

#### Data preprocessing 

In order to use a pytorch dataloader for training and testing, we split the data into seperate files by:

```
$ python data_generation.py --root_dir=/path/to/data_root_dir --dataset=DFAUST --num_valid=100
```

#### Training and Testing

```
args['mode'] = 'train' or 'test'

python pai3DMM.py
```

#### Some important notes:
* The code has compatibility with both _mpi-mesh_ and _trimesh_ packages (it can be chosen by setting the _meshpackage_ variable pai3DMM.py).




#### Acknowlegements:

The structure of this codebase is borrowed from [Neural3DMM](https://github.com/gbouritsas/Neural3DMM).

# Cite

Please consider citing our work if you find it useful:

```
@misc{gao2020learning,
    title={Learning Local Neighboring Structure for Robust 3D Shape Representation},
    author={Zhongpai Gao and Guangtao Zhai and Juyong Zhang and Junchi Yan and Yiyan Yang and Xiaokang Yang},
    year={2020},
    eprint={2004.09995},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```