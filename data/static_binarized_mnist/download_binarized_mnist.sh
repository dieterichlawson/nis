#!/bin/bash
wget -O train.amat http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat
wget -O test.amat http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat
wget -O valid.amat http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat
python convert_to_npy.py 
rm train.amat
rm test.amat
rm valid.amat
