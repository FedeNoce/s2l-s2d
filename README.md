# s2l-s2d
This repository contains the code of the paper "Learning Landmarks motion from Speech for Speaker-Agnostic 3D Talking Heads Generation"[Arxiv](https://arxiv.org/abs/2306.01415)
## About The Project
We implemented the 2D Kmeans algorithm in three different ways:
1) A sequential mode in Python
2) A Parallel mode in OpenMP
3) A Parallel mode in Cuda

### Built With

* [Python](https://www.python.org/)
* [OpenMP](https://www.openmp.org/)
* [CUDA](https://developer.nvidia.com/cuda-zone)

For more information, read the [report](reports/Kmeans_Report.pdf) or the [presentation](reports/KMeans_Presentation.pdf)
## Getting Started

In order to get a local copy and run some tests, follow these simple steps.

1. Clone the repo
```sh
git clone https://github.com/FedeNoce/Kmeans.git
```
2. Chose the implementation:  ```kmeans_seq.py``` for sequential, ```Kmeans_openMp.cpp``` for parallel with OpenMP, ```2D_kmeans_cuda.cu``` for parallel with CUDA.
3. Choose the dataset and copy the file path in the code
4. Set the parameters with your settings
5. Run the tests
6. Evaluate the clustering of the tests running ```evaluate_kmeans.py``` 
## Authors

* [**Corso Vignoli**](https://github.com/CVignoli)
* [**Federico Nocentini**](https://github.com/FedeNoce)


## Acknowledgments
Parallel Computing © Course held by Professor [Marco Bertini](https://www.unifi.it/p-doc2-2020-0-A-2b333d2d3529-1.html) - Computer Engineering Master Degree @[University of Florence](https://www.unifi.it/changelang-eng.html)

