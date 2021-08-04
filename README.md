# Cloud Transformers: A Universal Approach To Point Cloud Processing Tasks

This is an official PyTorch code repository of the paper "[Cloud Transformers: A Universal Approach To Point Cloud Processing Tasks
](https://arxiv.org/abs/2007.11679)" (**ICCV**, 2021). 

Here, we present a versatile point cloud processing block that yields state-of-the-art results on many tasks.   
The key idea is to process point clouds with many cheap low-dimensional *different* projections followed by standard convolutions. And we do so both *in parallel* and *sequentially*.

# Datasets

We provide links to the datasets we used to train/evaluate. After unpacking and preparation, please edit the dataset path (`data:path` field) in `configs/*.yaml`

* [ShapeNet Single-View 3D Reconstruction](https://github.com/lmb-freiburg/what3d)
* [ShapeNet Inpainting (GRNet version)](https://drive.google.com/drive/folders/1P_W1tz5Q4ZLapUifuOE4rFAZp6L1XTJz)
* [ScanObjectNN](https://github.com/hkust-vgd/scanobjectnn)
* [S3DIS KPConv protocol](https://github.com/zeliu98/CloserLook3D/tree/master/pytorch)
* [S3DIS 1 x 1 protocol](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip)

# Pre-trained models
We provide our pre-trained models' weights in a single [archive](https://disk.yandex.ru/d/IqL8tMfleVTnyQ).

# Building Dependencies

To install and build all the modules required, please run:
```
bash ./install_deps.sh
```

# Code Structure

In `layers/cloud_transform.py` the core operations are implemented (rasterization `Splat` and de-rasterization `Slice`). 
While in `layers\mutihead_ct_*.py` we provide slightly different versions of Multi-Headed Cloud Transform (MHCT).

The model zoo is situated in `model_zoo`, where the models for corresponding tasks are constructed of Multi-Headed Cloud Transforms.

# Run

We train our models in multi-GPU setting using DistributedDataParallel. To train on `n` GPUs, please run the following commands:

```
python train_${SCRIPT_NAME}.py ${EXP_NAME} -c configs/${CONFIG_NAME}.yaml --master localhost:3315 --rank 0 --num_nodes n
...
python train_${SCRIPT_NAME}.py ${EXP_NAME} -c configs/${CONFIG_NAME}.yaml --master localhost:3315 --rank <n-1> --num_nodes n
```


The semantics for evaluation scripts is almost the same:

```bash
python eval_${SCRIPT_NAME}.py ${EXP_NAME} -c configs/eval/${CONFIG_NAME}.yaml
```

# Cite

If you find our work helpful, please do not hesitate to cite us.


```
@inproceedings{mazur2021cloudtransformers,
  title={Cloud Transformers: A Universal Approach To Point Cloud Processing Tasks},
  author={Mazur, Kirill and Lempitsky, Victor},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2021}
}
```