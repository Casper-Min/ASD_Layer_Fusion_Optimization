from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, EltwiseLayer
from fusion.scheduling.batch_size import get_batch_size

"""
ResNet-18

He, Zhang, Ren, and Sun, 2015
"""
batch_size = get_batch_size()
NN = Network('ResNet18_B3')


####################################################################

# Block2
# Full Block(Not Padded)

# NN.set_input_layer(InputLayer(64, 56, nimg=batch_size))

# NN.add('conv2_1_1', ConvLayer(64, 64, 56, 3, nimg=batch_size))
# NN.add('conv2_1_2', ConvLayer(64, 64, 56, 3, nimg=batch_size))
# NN.add('conv2_1_res', EltwiseLayer(64, 56, 2, nimg=batch_size),prevs=('__INPUT__','conv2_1_2'))

# NN.add('conv2_2_1', ConvLayer(64, 64, 56, 3, nimg=batch_size))
# NN.add('conv2_2_2', ConvLayer(64, 64, 56, 3, nimg=batch_size))
# NN.add('conv2_2_res', EltwiseLayer(64, 56, 2, nimg=batch_size),prevs=('conv2_1_res','conv2_2_2'))

# Full Block(Padded)

# NN.set_input_layer(InputLayer(64, 64, nimg=batch_size))

# NN.add('conv2_1_1', ConvLayer(64, 64, 64, 3, nimg=batch_size))
# NN.add('conv2_1_2', ConvLayer(64, 64, 64, 3, nimg=batch_size))
# NN.add('conv2_1_res', EltwiseLayer(64, 64, 2, nimg=batch_size),prevs=('__INPUT__','conv2_1_2'))

# NN.add('conv2_2_1', ConvLayer(64, 64, 64, 3, nimg=batch_size))
# NN.add('conv2_2_2', ConvLayer(64, 64, 64, 3, nimg=batch_size))
# NN.add('conv2_2_res', EltwiseLayer(64, 64, 2, nimg=batch_size),prevs=('conv2_1_res','conv2_2_2'))

# Downsample

# NN.set_input_layer(InputLayer(64, 64, nimg=batch_size))

# NN.add('conv2_1_1', ConvLayer(64, 64, 64, 3, nimg=batch_size))
# NN.add('conv2_1_2', ConvLayer(64, 64, 64, 3, nimg=batch_size))
# NN.add('conv2_1_res', EltwiseLayer(64, 64, 2, nimg=batch_size),prevs=('__INPUT__','conv2_1_2'))

# No_Downsample

# NN.set_input_layer(InputLayer(64, 64, nimg=batch_size))

# NN.add('conv2_2_1', ConvLayer(64, 64, 64, 3, nimg=batch_size))
# NN.add('conv2_2_2', ConvLayer(64, 64, 64, 3, nimg=batch_size))
# NN.add('conv2_2_res', EltwiseLayer(64, 64, 2, nimg=batch_size),prevs=('__INPUT__','conv2_2_2'))

####################################################################

# Block3
# Full Block(Not Padded)

# NN.set_input_layer(InputLayer(64, 56, nimg=batch_size))

# NN.add('conv3_1_1', ConvLayer(64, 128, 28, 3, 2, nimg=batch_size))
# NN.add('conv3_1_2', ConvLayer(128, 128, 28, 3, nimg=batch_size))
# NN.add('conv3_1_br', ConvLayer(64, 128, 28, 1, 2, nimg=batch_size),prevs=('__INPUT__',))
# NN.add('conv3_1_res', EltwiseLayer(128, 28, 2, nimg=batch_size),prevs=('conv3_1_2','conv3_1_br'))

# NN.add('conv3_2_1', ConvLayer(128, 128, 28, 3, nimg=batch_size))
# NN.add('conv3_2_2', ConvLayer(128, 128, 28, 3, nimg=batch_size))
# NN.add('conv3_2_res', EltwiseLayer(128, 28, 2, nimg=batch_size),prevs=('conv3_1_res','conv3_2_2'))

# Full Block(Padded)

# NN.set_input_layer(InputLayer(64, 64, nimg=batch_size))

# NN.add('conv3_1_1', ConvLayer(64, 128, 32, 3, 2, nimg=batch_size))
# NN.add('conv3_1_2', ConvLayer(128, 128, 32, 3, nimg=batch_size))
# NN.add('conv3_1_br', ConvLayer(64, 128, 32, 1, 2, nimg=batch_size),prevs=('__INPUT__',))
# NN.add('conv3_1_res', EltwiseLayer(128, 32, 2, nimg=batch_size),prevs=('conv3_1_2','conv3_1_br'))

# NN.add('conv3_2_1', ConvLayer(128, 128, 32, 3, nimg=batch_size))
# NN.add('conv3_2_2', ConvLayer(128, 128, 32, 3, nimg=batch_size))
# NN.add('conv3_2_res', EltwiseLayer(128, 32, 2, nimg=batch_size),prevs=('conv3_1_res','conv3_2_2'))

# Downsample

# NN.set_input_layer(InputLayer(64, 64, nimg=batch_size))

# NN.add('conv3_1_1', ConvLayer(64, 128, 32, 3, 2, nimg=batch_size))
# NN.add('conv3_1_2', ConvLayer(128, 128, 32, 3, nimg=batch_size))
# NN.add('conv3_1_br', ConvLayer(64, 128, 32, 1, 2, nimg=batch_size),prevs=('__INPUT__',))
# NN.add('conv3_1_res', EltwiseLayer(128, 32, 2, nimg=batch_size),prevs=('conv3_1_2','conv3_1_br'))

# No_Downsample

# NN.set_input_layer(InputLayer(128, 32, nimg=batch_size))

# NN.add('conv3_2_1', ConvLayer(128, 128, 32, 3, nimg=batch_size))
# NN.add('conv3_2_2', ConvLayer(128, 128, 32, 3, nimg=batch_size))
# NN.add('conv3_2_res', EltwiseLayer(128, 32, 2, nimg=batch_size),prevs=('__INPUT__','conv3_2_2'))

####################################################################

# Block4
# Full Block(Not Padded)

# NN.set_input_layer(InputLayer(128, 28, nimg=batch_size))

# NN.add('conv4_1_1', ConvLayer(128, 256, 14, 3, 2, nimg=batch_size))
# NN.add('conv4_1_2', ConvLayer(256, 256, 14, 3, nimg=batch_size))
# NN.add('conv4_1_br', ConvLayer(128, 256, 14, 1, 2, nimg=batch_size),prevs=('__INPUT__',))
# NN.add('conv4_1_res', EltwiseLayer(256, 14, 2, nimg=batch_size),prevs=('conv4_1_2','conv4_1_br'))

# NN.add('conv4_2_1', ConvLayer(256, 256, 14, 3, nimg=batch_size))
# NN.add('conv4_2_2', ConvLayer(256, 256, 14, 3, nimg=batch_size))
# NN.add('conv4_2_res', EltwiseLayer(256, 14, 2, nimg=batch_size),prevs=('conv4_1_res','conv4_2_2'))

# Full Block(Padded)

# NN.set_input_layer(InputLayer(128, 32, nimg=batch_size))

# NN.add('conv4_1_1', ConvLayer(128, 256, 16, 3, 2, nimg=batch_size))
# NN.add('conv4_1_2', ConvLayer(256, 256, 16, 3, nimg=batch_size))
# NN.add('conv4_1_br', ConvLayer(128, 256, 16, 1, 2, nimg=batch_size),prevs=('__INPUT__',))
# NN.add('conv4_1_res', EltwiseLayer(256, 16, 2, nimg=batch_size),prevs=('conv4_1_2','conv4_1_br'))

# NN.add('conv4_2_1', ConvLayer(256, 256, 16, 3, nimg=batch_size))
# NN.add('conv4_2_2', ConvLayer(256, 256, 16, 3, nimg=batch_size))
# NN.add('conv4_2_res', EltwiseLayer(256, 16, 2, nimg=batch_size),prevs=('conv4_1_res','conv4_2_2'))

# Downsample

# NN.set_input_layer(InputLayer(128, 32, nimg=batch_size))

# NN.add('conv4_1_1', ConvLayer(128, 256, 16, 3, 2, nimg=batch_size))
# NN.add('conv4_1_2', ConvLayer(256, 256, 16, 3, nimg=batch_size))
# NN.add('conv4_1_br', ConvLayer(128, 256, 16, 1, 2, nimg=batch_size),prevs=('__INPUT__',))
# NN.add('conv4_1_res', EltwiseLayer(256, 16, 2, nimg=batch_size),prevs=('conv4_1_2','conv4_1_br'))

# No_Downsample

# NN.set_input_layer(InputLayer(256, 16, nimg=batch_size))

# NN.add('conv4_2_1', ConvLayer(256, 256, 16, 3, nimg=batch_size))
# NN.add('conv4_2_2', ConvLayer(256, 256, 16, 3, nimg=batch_size))
# NN.add('conv4_2_res', EltwiseLayer(256, 16, 2, nimg=batch_size),prevs=('__INPUT__','conv4_2_2'))

####################################################################

# Block5
# Full Block(Not Padded)

# NN.set_input_layer(InputLayer(256, 14, nimg=batch_size))

# NN.add('conv5_1_1', ConvLayer(256, 512, 7, 3, 2, nimg=batch_size))
# NN.add('conv5_1_2', ConvLayer(512, 512, 7, 3, nimg=batch_size))
# NN.add('conv5_1_br', ConvLayer(256, 512, 7, 1, 2, nimg=batch_size),prevs=('__INPUT__',))
# NN.add('conv5_1_res', EltwiseLayer(512, 7, 2, nimg=batch_size),prevs=('conv5_1_2','conv5_1_br'))

# NN.add('conv5_2_1', ConvLayer(512, 512, 7, 3, nimg=batch_size))
# NN.add('conv5_2_2', ConvLayer(512, 512, 7, 3, nimg=batch_size))
# NN.add('conv5_2_res', EltwiseLayer(512, 7, 2, nimg=batch_size),prevs=('conv5_1_res','conv5_2_2'))

# Full Block(Padded)

NN.set_input_layer(InputLayer(256, 16, nimg=batch_size))

NN.add('conv5_1_1', ConvLayer(256, 512, 8, 3, 2, nimg=batch_size))
NN.add('conv5_1_2', ConvLayer(512, 512, 8, 3, nimg=batch_size))
NN.add('conv5_1_br', ConvLayer(256, 512, 8, 1, 2, nimg=batch_size),prevs=('__INPUT__',))
NN.add('conv5_1_res', EltwiseLayer(512, 8, 2, nimg=batch_size),prevs=('conv5_1_2','conv5_1_br'))

NN.add('conv5_2_1', ConvLayer(512, 512, 8, 3, nimg=batch_size))
NN.add('conv5_2_2', ConvLayer(512, 512, 8, 3, nimg=batch_size))
NN.add('conv5_2_res', EltwiseLayer(512, 8, 2, nimg=batch_size),prevs=('conv5_1_res','conv5_2_2'))

# Downsample

# NN.set_input_layer(InputLayer(256, 16, nimg=batch_size))

# NN.add('conv5_1_1', ConvLayer(256, 512, 8, 3, 2, nimg=batch_size))
# NN.add('conv5_1_2', ConvLayer(512, 512, 8, 3, nimg=batch_size))
# NN.add('conv5_1_br', ConvLayer(256, 512, 8, 1, 2, nimg=batch_size),prevs=('__INPUT__',))
# NN.add('conv5_1_res', EltwiseLayer(512, 8, 2, nimg=batch_size),prevs=('conv5_1_2','conv5_1_br'))

# No_Downsample

# NN.set_input_layer(InputLayer(512, 8, nimg=batch_size))

# NN.add('conv5_2_1', ConvLayer(512, 512, 8, 3, nimg=batch_size))
# NN.add('conv5_2_2', ConvLayer(512, 512, 8, 3, nimg=batch_size))
# NN.add('conv5_2_res', EltwiseLayer(512, 8, 2, nimg=batch_size),prevs=('__INPUT__','conv5_2_2'))

####################################################################

#Total Network

# NN.set_input_layer(InputLayer(3, 224, nimg=batch_size))

# NN.add('conv1', ConvLayer(3, 64, 112, 7, 2, nimg=batch_size))
# NN.add('pool1', PoolingLayer(64, 56, 3, 2, nimg=batch_size))


# NN.set_input_layer(InputLayer(64, 112, nimg=batch_size))
# NN.add('conv1', ConvLayer(64, 64, 56, 3, 2, nimg=batch_size),prevs=('__INPUT__',))



# NN.set_input_layer(InputLayer(64, 56, nimg=batch_size))

# NN.add('conv2_1_1', ConvLayer(64, 64, 56, 3, nimg=batch_size))
# NN.add('conv2_1_2', ConvLayer(64, 64, 56, 3, nimg=batch_size))
# NN.add('conv2_1_res', EltwiseLayer(64, 56, 2, nimg=batch_size),prevs=('__INPUT__','conv2_1_2'))

# NN.add('conv2_2_1', ConvLayer(64, 64, 56, 3, nimg=batch_size))
# NN.add('conv2_2_2', ConvLayer(64, 64, 56, 3, nimg=batch_size))
# NN.add('conv2_2_res', EltwiseLayer(64, 56, 2, nimg=batch_size),prevs=('conv2_1_res','conv2_2_2'))


# NN.add('conv3_1_1', ConvLayer(64, 128, 28, 3, 2, nimg=batch_size))
# NN.add('conv3_1_2', ConvLayer(128, 128, 28, 3, nimg=batch_size))
# NN.add('conv3_1_br', ConvLayer(64, 128, 28, 1, 2, nimg=batch_size),prevs=('conv2_2_res',))
# NN.add('conv3_1_res', EltwiseLayer(128, 28, 2, nimg=batch_size),prevs=('conv3_1_2','conv3_1_br'))

# NN.add('conv3_2_1', ConvLayer(128, 128, 28, 3, nimg=batch_size))
# NN.add('conv3_2_2', ConvLayer(128, 128, 28, 3, nimg=batch_size))
# NN.add('conv3_2_res', EltwiseLayer(128, 28, 2, nimg=batch_size),prevs=('conv3_1_res','conv3_2_2'))


# NN.add('conv4_1_1', ConvLayer(128, 256, 14, 3, 2, nimg=batch_size))
# NN.add('conv4_1_2', ConvLayer(256, 256, 14, 3, nimg=batch_size))
# NN.add('conv4_1_br', ConvLayer(128, 256, 14, 1, 2, nimg=batch_size),prevs=('conv3_2_res',))
# NN.add('conv4_1_res', EltwiseLayer(256, 14, 2, nimg=batch_size),prevs=('conv4_1_2','conv4_1_br'))

# NN.add('conv4_2_1', ConvLayer(256, 256, 14, 3, nimg=batch_size))
# NN.add('conv4_2_2', ConvLayer(256, 256, 14, 3, nimg=batch_size))
# NN.add('conv4_2_res', EltwiseLayer(256, 14, 2, nimg=batch_size),prevs=('conv4_1_res','conv4_2_2'))


# NN.add('conv5_1_1', ConvLayer(256, 512, 7, 3, 2, nimg=batch_size))
# NN.add('conv5_1_2', ConvLayer(512, 512, 7, 3, nimg=batch_size))
# NN.add('conv5_1_br', ConvLayer(256, 512, 7, 1, 2, nimg=batch_size),prevs=('conv4_2_res',))
# NN.add('conv5_1_res', EltwiseLayer(512, 7, 2, nimg=batch_size),prevs=('conv5_1_2','conv5_1_br'))

# NN.add('conv5_2_1', ConvLayer(512, 512, 7, 3, nimg=batch_size))
# NN.add('conv5_2_2', ConvLayer(512, 512, 7, 3, nimg=batch_size))
# NN.add('conv5_2_res', EltwiseLayer(512, 7, 2, nimg=batch_size),prevs=('conv5_1_res','conv5_2_2'))

# NN.add('pool5', PoolingLayer(512, 1, 7, nimg=batch_size))

# NN.add('fc', FCLayer(512, 1000, nimg=batch_size))


####################################################################


# NN.set_input_layer(InputLayer(128, 56, nimg=batch_size))

# NN.add('conv2_1_1', ConvLayer(128, 128, 56, 3, nimg=batch_size))
# NN.add('conv2_1_2', ConvLayer(128, 128, 56, 3, nimg=batch_size))
# NN.add('conv2_1_3', ConvLayer(128, 128, 56, 3, nimg=batch_size))


