# LiDAR-in-the-loop Hyperparameter Optimization Simulator

*by [Felix Antoine Goudreault](https://scholar.google.com/citations?user=DncgVscAAAAJ), Dominik Scheuble, [Mario Bijelic](http://mariobijelic.de), [Nicolas Robidoux](https://scholar.google.com/citations?user=Rd8f9jYAAAAJ) and [Felix Heide](https://www.cs.princeton.edu/~fheide/) <br>

This is the official code repository for the [LiDAR-in-the-loop Hyperparameter Optimization](https://light.princeton.edu/publication/lidar-in-the-loop-hyperparameter-optimization/), to be presented at [CVPR 2023](https://cvpr2023.thecvf.com/). <br>
This GitHub repository only contains the transient LiDAR simulator source code that was used to generate the realistic point cloud data. It does not contain the optimization algorithm and the object detector sources. The latter is based on the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) library.

<img src="teaser.gif" width="850">

## Requirements

This package requires a customized Carla engine docker image which can be downloaded [here](https://drive.google.com/file/d/1JS9PiQkcHp4Yepi2NTjjDyLXnfiWYBOj/view?usp=drive_link)
and a corresponding carla python interface which can be installed from the [carla wheel file](carla-0.9.13-cp38-cp38-linux_x86_64.whl) located in this repository.
Only python 3.8 is supported with this wheel and other python dependencies are listed in the [setup.cfg](setup.cfg) file (see [Installation](Installation) for more detailed setup instructions).

## Project Structure

    .
    ├── litl_simulator             # contains our LiDAR simulator and viewers
    │   ├── bases.py
    │   ├── data_descriptors.py
    │   ├── exceptions.py
    │   ├── gui.py
    │   ├── settings.py
    │   ├── utils.py
    │   ├── lidargui              # source code for viewer
    │   │   └── ...
    │   └── process_pipelines     # contains source code for raw data processing
    │   │   └── ...
    │   └── simulation            # contains tools for raw data generation
    │       └── ...
    ├── tutorials                 # contains tutorial notebooks
    │   └── ...
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── teaser.gif

## Getting Started

### Setup

1) Install [anaconda](https://docs.anaconda.com/anaconda/install/).

2) Execute the following commands.
```bash
# Create a new conda environment.
conda create --name litl_simulator python=3.8 -y

# Activate the newly created conda environment.
conda activate litl_simulator

# Clone this repository (including submodules!).
git clone git@github.com:princeton-computational-imaging/LITL-Optimization.git
cd LITL-Optimization

# Install packages
pip install .

# Install carla python interface
pip install carla-0.9.13-cp38-cp38-linux_x86_64.whl
```

5) Download custom carla engine docker image [here](https://drive.google.com/file/d/1JS9PiQkcHp4Yepi2NTjjDyLXnfiWYBOj/view?usp=drive_link).

6) Load carla image into docker registry.
```bash
docker load -i path/to/carla_image.tar.gz
``

7) Look at [tutorial part 1](tutorials/part1_generate_raw_data.ipynb) to run the custom carla engine.


### Tutorials

There are 3 notebooks in the [tutorials](tutorials) directory which explains in details how to use this repository. To run them, execute the following commands:

```bash
# install jupyter notebooks
pip install notebook
# run jupyter in the tutorials directory
# this will open a browser window from which you can select the desired notebook
jupyter notebook tutorials
```

1) [part 1](tutorials/part1_generate_raw_data.ipynb) explains how to generate raw data from the custom carla engine which can be downloaded using the link above. It also details how to run the docker image in order to visualize the scene from which the point clouds are generated in real time.
2) [part 2](tutorials/part2_process_raw_data.ipynb) shows how to use the various models implemented in this repo in order to generate realistic point clouds.
3) [part 3](tutorials/part3_gui.ipynb) discribes the special GUI developed to visualize the point clouds generated with this project.


### Disclaimer

The code has been successfully tested on
- Ubuntu 18.04.6 and 20.04 LTS + CUDA 11.3 + python 3.8


## License

TODO
This software is made available for non-commercial use under a Creative Commons [License](LICENSE).<br>
A summary of the license can be found [here](https://creativecommons.org/licenses/by-nc/4.0/).


## Citation(s)

If you find this work useful, please consider citing our paper.
```bibtex
@inproceeding{goudreault2023lidarhyperoptim,
title={LiDAR-in-the-loop Hyperparameter Optimization},
author={Felix Goudreault and Dominik Scheuble and Mario Bijelic and Nicolas Robidoux and Felix Heide},
journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2023}
}
```
You may also want to check out our earlier works: <br>
[*Differentiable Compound Optics and Processing Pipeline Optimization for End-to-end Camera Design*](https://light.princeton.edu/publication/deep_compound_optics/).

```bibtex
@article{Tseng2021DeepCompoundOptics,
title={Differentiable Compound Optics and Processing Pipeline Optimization for End-to-end Camera Design},
author={Tseng, Ethan and Mosleh, Ali and Mannan, Fahim and St-Arnaud, Karl and Sharma, Avinash and Peng, Yifan and Braun, Alexander and Nowrouzezahrai, Derek and Lalonde, Jean-Francois and Heide, Felix},
journal={ACM Transactions on Graphics (TOG)},
volume={40},
number={2},
articleno = {18}
year={2021},
publisher={ACM}
}
```

[*Hardware-in-the-loop End-to-end Optimization of Camera Image Processing Pipelines*](https://light.princeton.edu/publication/hil_image_optimization/).

```bibtex
@InProceedings{isp_opt_cvpr20,
author={Mosleh, Ali and Sharma, Avinash and Onzon, Emmanuel and Mannan, Fahim and Robidoux, Nicolas and Heide, Felix},
title={Hardware-in-the-loop End-to-end Optimization of Camera Image Processing Pipelines},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

[*End-to-end High Dynamic Range Camera Pipeline Optimization*](https://light.princeton.edu/publication/hdr_isp_opt/).

```bibtex
@InProceedings{robidoux2021hdr_isp_opt,
author = {Nicolas Robidoux, Luis Eduardo García Capel, Dong-eun Seo, Avinash Sharma, Federico Ariza, Felix Heide},
title = {End-to-end High Dynamic Range Camera Pipeline Optimization},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year = {2021}
}
```

## Contributions
Please feel free to suggest improvements to this repository.<br>
We are always open to merge useful pull request.

## Acknowledgments

This work was supported by the AISEE project with funding from the FFG, BMBF, and NRCIRA. We thank the Federal Ministry for Economic Affairs and Energy for support through the PEGASUS-family project VVM-Verification and Validation Methods for Automated Vehicles Level 4 and 5. Felix Heide was supported by an NSF CAREER Award (2047359), a Packard Foundation Fellowship, a Sloan Research Fellowship, a Sony Young Faculty Award, a Project X Innovation Award, and an Amazon Science Research Award.
