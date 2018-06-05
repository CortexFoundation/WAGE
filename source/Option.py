import time
import tensorflow as tf

debug = False
Time = time.strftime('%Y-%m-%d', time.localtime())
# Notes = 'vgg7 2888'

resmode = 0.25 # 4.0, 2.0, 1.0, 0.5, 0.25

bitsW = 8  # bit width of weights
bitsA = 8  # bit width of activations
bitsG = 8  # bit width of gradients
bitsE = 8  # bit width of errors
bitsR = 16  # bit width of randomizer

Notes = 'ResNet29_'+str(resmode)+'_'+''.join([str(item) for item in [bitsW, bitsA, bitsG, bitsE]])
print Notes
#Notes = 'temp'

GPU = [0]
batchSize = 128

dataSet = 'CIFAR100'

loadModel = None
# loadModel = '../model/' + '2017-12-06' + '(' + 'vgg7 2888' + ')' + '.tf'
# saveModel = None
saveModel = '../model/' + Time + '(' + Notes + ')' + '.tf'


lr = tf.Variable(initial_value=0., trainable=False, name='lr', dtype=tf.float32)
# bitsW = 2 => lr_mult = 1
# bitsW = 4 => lr_mult = 0.25
lr_mult_dict = {2:1, 4:0.25, 8:0.015625}
lr_mult = lr_mult_dict[bitsW]
print lr_mult
lr_schedule = [0, 8*lr_mult, 200, 1*lr_mult, 250,1./8*lr_mult,300, 0]



L2 = 0 #1.0/1024

lossFunc = 'SSE'
# lossFunc = tf.losses.softmax_cross_entropy
# optimizer = tf.train.GradientDescentOptimizer(1)  # lr is controlled in Quantize.G
# optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

# shared variables, defined by other files
seed = None
sess = None
W_scale = []
