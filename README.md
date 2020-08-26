# Overview
This repository constists of the implementations of the Distance Correlation (DC) and Information Over Bias (IOB) metrics proposed in [link]. The two metrics can be used to assess the level of disentanglement between spatial content and vector style representations. Both metrics are ready to use with PyTorch and TensorFlow implementations.

# System Requirements
* Pytorch 1.5.1 or higher with GPU support
* Python 3.7.2 or higher
* TensorFlow r2.0 or higher with GPU support
* CUDA toolkit 10 or newer

# Metric 1: Distance Correlation (DC) for measuring "independence"


## Evaluating independence with DC

# Metric 2: Information Over Bias (IOB) for measuring "informativeness"

## Evaluating informativeness with IOB

# Citation
If you find our metrics useful please cite the following paper:
```
@inproceedings{liu2020metrics,
  author       = "Xiao Liu and Spyridon Thermos and Gabriele Valvano and Agisilaos Chartsias and Alison O'Neil and Sotirios A. Tsaftaris",
  title        = "Metrics for Exposing the Biases of Content-Style Disentanglement",
  booktitle    = "arxiv",
  year         = "2020"
}
```

# License
All scripts are released under the MIT License.
