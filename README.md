## PolyConv
Code for [Object Point Cloud Classification via Poly-Convolutional Architecture Search](https://dl.acm.org/doi/10.1145/3474085.3475252)

### Requirements
```
Python >= 3.7.6, PyTorch >= 1.6.0, torchvision == 0.2.0
```
### Datasets

Modelnet40 will be automatically downloaded and extracted as in `data.py`

### Usage
To train an architecture, run
```
python train.py --arch ${arch_name}
```

For architecture search, run 
```
python search.py --epochs 200 --select_c 1.0 --save PolyConv
```

To derive architectures after MCTS, run 
```
bash derive.sh PolyConv 200 1.0
```

The derived architectures are stored in sample_arch.py, print them out via
```
cat sample_arch.py
```
and train the architecture of interest with
```
python train.py --arch ${arch_name}
```

### Reminder

- PolyConv operations are wrapped in a `module_list` object created by function `creat_ops` as defined in `model.py`. 
It also leverages some functions in `util.py`.
If you wish to employ PolyConv operations in your project, you might directly go to these two files.
- The GPU memory consumption varies when using different sorts of PolyConv ops. With a training batch size of 32, it takes approximately 3 RTX 2080Ti for searching, and usually 1~2 RTX 2080Ti for training. But it might cost up to 4 GPU when the architecture is derived from an immature searching.

### Citation
If your research use any part of this code, please cite:
```
@inproceedings{lin2021object,
  title={Object Point Cloud Classification via Poly-Convolutional Architecture Search},
  author={Lin, Xuanxiang and Chen, Ke and Jia, Kui},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={807--815},
  year={2021}
}
```

### Acknowledgement
Some implementations are from [ENAS](https://github.com/melodyguan/enas), [DARTS](https://github.com/quark0/darts) and [DGCNN](https://github.com/WangYueFt/dgcnn)
