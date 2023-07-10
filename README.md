# S2L-S2D: Learning Landmarks Motion from Speech for Speaker-Agnostic 3D Talking Heads Generation

This repository contains the code for the paper "Learning Landmarks Motion from Speech for Speaker-Agnostic 3D Talking Heads Generation" ([link](https://arxiv.org/abs/2306.01415)). The paper presents a novel approach based on landmarks motion for generating 3D Talking Heads from speech. The code includes the implementation of two models proposed in the paper: S2L and S2D.

![alt text](https://github.com/FedeNoce/s2l-s2d/images/s2l.png)
![alt text](https://github.com/FedeNoce/s2l-s2d/images/s2d.png)

## Installation
To run the code, you need to install the following dependencies:
- Python 3.8
- PyTorch-GPU 1.13.0
- Trimesh 3.9.2
- Librosa 3.9.2
- Transformers 4.6.1 from Hugging Face
- MPI-IS for mesh rendering ([link](https://github.com/MPI-IS/mesh))
- Additional dependencies for running the demo: pysimplegui==4.60.5, sounddevice==0.4.6, soundfile==0.12.1

## Training Setup
1. Clone the repository:
```sh
git clone https://github.com/FedeNoce/s2l-s2d.git
```
2. Download the vocaset dataset from [here](https://voca.is.tue.mpg.de/download.php) (Training Data, 8GB).
3. Put the downloaded file into the "S2L/vocaset" and "S2D/vocaset" directories.
4. To train S2L, preprocess the data by running "preprocess_voca_data.py" in the "S2L/vocaset" directory. Then, run "train_S2L.py".
5. To train S2D, preprocess the data by running "Data_processing.py" in the "S2D" directory. Then, run "train_S2D.py".

## Inference
1. Download the pretrained models from [here](https://drive.google.com/drive/folders/1h0l8cMUh_7GVedJykYH8zSEqNhj3BVeJ?usp=sharing) and place them in the "S2L/Results" and "S2D/Results" directories.
2. Run the GUI demo using "new_gui_demo.py".

For more information, you can visit the GitHub profile of [Federico Nocentini](https://github.com/FedeNoce).
