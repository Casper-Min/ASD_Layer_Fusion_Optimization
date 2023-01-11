from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer
from fusion.scheduling.batch_size import get_batch_size

"""
VGGNet-16

Simonyan and Zisserman, 2014
"""
batch_size = get_batch_size()
NN = Network('VGG_padded')

NN.set_input_layer(InputLayer(3, 256, nimg=batch_size))

NN.add('conv1', ConvLayer(3, 64, 256, 3, nimg=batch_size))
NN.add('conv2', ConvLayer(64, 64, 256, 3, nimg=batch_size))
NN.add('pool1', PoolingLayer(64, 128, 2, nimg=batch_size))

NN.add('conv3', ConvLayer(64, 128, 128, 3, nimg=batch_size))
NN.add('conv4', ConvLayer(128, 128, 128, 3, nimg=batch_size))
NN.add('pool2', PoolingLayer(128, 64, 2, nimg=batch_size))

NN.add('conv5', ConvLayer(128, 256, 64, 3, nimg=batch_size))
NN.add('conv6', ConvLayer(256, 256, 64, 3, nimg=batch_size))
NN.add('conv7', ConvLayer(256, 256, 64, 3, nimg=batch_size))
NN.add('pool3', PoolingLayer(256, 32, 2, nimg=batch_size))

NN.add('conv8', ConvLayer(256, 512, 32, 3, nimg=batch_size))
NN.add('conv9', ConvLayer(512, 512, 32, 3, nimg=batch_size))
NN.add('conv10', ConvLayer(512, 512, 32, 3, nimg=batch_size))
NN.add('pool4', PoolingLayer(512, 16, 2, nimg=batch_size))

NN.add('conv11', ConvLayer(512, 512, 16, 3, nimg=batch_size))
NN.add('conv12', ConvLayer(512, 512, 16, 3, nimg=batch_size))
NN.add('conv13', ConvLayer(512, 512, 16, 3, nimg=batch_size))
NN.add('pool5', PoolingLayer(512, 8, 2, nimg=batch_size))

# NN.add('fc1', FCLayer(512, 4096, 7, nimg=batch_size))
# NN.add('fc2', FCLayer(4096, 4096, nimg=batch_size))
# NN.add('fc3', FCLayer(4096, 1000, nimg=batch_size))

