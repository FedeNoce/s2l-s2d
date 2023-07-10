# s2l-s2d
This repository contains the code of the [paper](https://arxiv.org/abs/2306.01415) "Learning Landmarks motion from Speech for Speaker-Agnostic 3D Talking Heads Generation".
In this paper we propose a novel approach, based on landmarks motion, for generating 3D Talking Heads from speech. In this repo we insert the code for training the 2 models proposed in the paper S2L and S2D.

## Installation
To run the code you will need to install:
* python=3.8
* pytorch-gpu=1.13.0
* trimesh=3.9.2
* librosa=3.9.2
* transformers=4.6.1
* You will need to setup [MPI-IS](https://github.com/MPI-IS/mesh) for meshes rendering
* If you want to run the demo you will need: pysimplegui==4.60.5, sounddevice==0.4.6, soundfile==0.12.1

## Training Setup
*  Clone the repo:
```sh
git clone https://github.com/FedeNoce/s2l-s2d.git
```
*  Download the vocaset dataset from [here](https://voca.is.tue.mpg.de/download.php), get Training Data (8GB)
*  Put the donwloaded file into S2L/vocaset and S2D/vocaset
*  To train S2L preprocess the datas running ```preprocess_voca_data.py``` in S2L/vocaset
*  Ther run ```train_S2L.py```
*  To train S2D preprocess the datas running ```Data_processing.py``` in S2D
*  Ther run ```train_S2D.py```
  
## Inference
*  Download the pretrained models from [here](https://drive.google.com/drive/folders/1h0l8cMUh_7GVedJykYH8zSEqNhj3BVeJ?usp=sharing), put them in S2L/Results and in S2D/Results
*  Run the gui-demo with ```new_gui_demo.py```

* [**Federico Nocentini**](https://github.com/FedeNoce)
