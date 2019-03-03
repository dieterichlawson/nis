wget -O train-ims.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -O train-lbs.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -O test-ims.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -O test-lbs.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
python3 convert_to_npy.py
rm *.gz
