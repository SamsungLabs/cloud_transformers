#/!bin/bash

conda create -n cloud_transformers_env python=3.6 -y
source activate cloud_transformers_env
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
conda install -c open3d-admin -c conda-forge open3d==0.9 -y
conda install -c iopath -c conda-forge iopath -y
pip install -U fvcore
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py36_cu101_pyt160/download.html
pip install h5py==2.10.0 scikit-learn==0.21.3 scipy==1.3.1 transforms3d==0.3.1 opencv-python==3.4.2.17

# install deps

cd chamfer_extension && python setup.py build_ext --inplace && cd ..
cd emd_linear && python setup.py build_ext --inplace && cd ..
cd cpp_wrappers && bash compile_wrappers.sh && cd ..
