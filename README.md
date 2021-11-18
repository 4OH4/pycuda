# PyCUDA
Supporting files for "Accelerate computation with PyCUDA" talk

[Accompanying blog post on Medium](https://medium.com/@rupertt/accelerate-computation-with-pycuda-2c12a6555cc6?sk=822d3578b25cbcd0b0cdd85c9bcfb80f)

## Installation and setup
Requirements:
 - [NumPy](https://numpy.org/install/)
 - [PyCUDA](https://documen.tician.de/pycuda/install.html)

A CUDA-compatible GPU is required, as well as the [NVIDIA Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit).

## Running the examples:

To run the toy example (multiplying numbers from two arrays):
```
$ python toy_example.py
```

To run the median filter comparison (CUDA vs NumPY):
```
$ python median_filter.py
Median_filter - using CUDA: True
2212.39 cycles/sec
2183.41 cycles/sec
2193.94 cycles/sec
2164.50 cycles/sec
2190.10 cycles/sec
Median_filter - without CUDA
51.98 cycles/sec
51.11 cycles/sec
50.96 cycles/sec
51.17 cycles/sec
51.13 cycles/sec
```