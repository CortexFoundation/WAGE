import numpy as np
import time
import tensorflow as tf
import NN
import Option
import Log
import getData
import Quantize
import re
import os 
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
# for single GPU quanzation
def quantizeGrads(Grad_and_vars):
  if Quantize.bitsG <= 16:
    grads = []
    for grad_and_vars in Grad_and_vars:
      grads.append([Quantize.G(grad_and_vars[0]), grad_and_vars[1]])
    return grads
  return Grad_and_vars

def showVariable(keywords=None):
  Vars = tf.global_variables()
  Vars_key = []
  for var in Vars:
    print var.device,var.name,var.shape,var.dtype
    if keywords is not None:
      if var.name.lower().find(keywords) > -1:
        Vars_key.append(var)
    else:
      Vars_key.append(var)
  return Vars_key

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def main():
  # get Option
  GPU = Option.GPU
  batchSize = Option.batchSize
  pathLog = '../log/' + Option.Time + '(' + Option.Notes + ')' + '.txt'
  Log.Log(pathLog, 'w+', 1) # set log file
  print time.strftime('%Y-%m-%d %X', time.localtime()), '\n'
  print open('Option.py').read()

  # get data
  numThread = 4*len(GPU)
  assert batchSize % len(GPU) == 0, ('batchSize must be divisible by number of GPUs')

  with tf.device('/cpu:0'):
    batchTrainX,batchTrainY,batchTestX,batchTestY,numTrain,numTest,label =\
        getData.loadData(Option.dataSet,batchSize,numThread)

  batchNumTrain = numTrain / batchSize
  batchNumTest = numTest / 100

  optimizer = Option.optimizer
  global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
  Net = []
  splitTrainX = tf.split(batchTrainX,len(GPU))
  splitTrainY = tf.split(batchTrainY,len(GPU))
  # splitTestX = tf.split(batchTestX,len(GPU))
  # splitTestY = tf.split(batchTestY,len(GPU))
  tower_grads = []
  # on my machine, alexnet does not fit multi-GPU training
  # for single GPU
  with tf.variable_scope(tf.get_variable_scope()):
    for i in GPU:
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (Option.TOWER_NAME, i)) as scope:
          Net.append(NN.NN(splitTrainX[i], splitTrainY[i], training=True, global_step=global_step,GPU=i))
          lossTrainBatch, errorTrainBatch = Net[-1].build_graph()
          update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # batchnorm moving average update ops (not used now)

          # since we quantize W at the beginning and the update delta_W is quantized,
          # there is no need to quantize W every iteration
          # we just clip W after each iteration for speed
          update_op += Net[i].W_clip_op
          total_loss = tf.add_n([lossTrainBatch], name='total_loss')

          # Attach a scalar summary to all individual losses and the total loss; do the
          # same for the averaged version of the losses.
          for l in [lossTrainBatch] + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            loss_name = re.sub('%s_[0-9]*/' % Option.TOWER_NAME, '', l.op.name)
            tf.summary.scalar(loss_name, l)
          gradTrainBatch = optimizer.compute_gradients(total_loss)
          gradTrainBatch_quantize = quantizeGrads(gradTrainBatch)
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          tf.get_variable_scope().reuse_variables()
          Net.append(NN.NN(batchTestX, batchTestY, training=False))
          _, errorTestBatch = Net[-1].build_graph()
          tower_grads.append(gradTrainBatch_quantize)


  grads = average_gradients(tower_grads)
  showVariable()
  summaries.append(tf.summary.scalar('learning_rate', Option.lr))
  for grad, var in grads:
    if grad is not None:
      summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
  
  with tf.control_dependencies(update_op):
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
  for var in tf.trainable_variables():
    summaries.append(tf.summary.histogram(var.op.name, var))
  variable_averages = tf.train.ExponentialMovingAverage(
        Option.MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  train_op = tf.group(train_op, variables_averages_op)

  # Build an initialization operation to run below.
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  config.log_device_placement = False
  saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
  summary_op = tf.summary.merge(summaries)
  sess = Option.sess = tf.InteractiveSession(config=config)
  sess.run(tf.global_variables_initializer())
  # Start the queue runners.
  tf.train.start_queue_runners(sess=sess)



  def getErrorTest():
    errorTest = 0.
    for i in tqdm(xrange(batchNumTest),desc = 'Test', leave=False):
      errorTest += sess.run([errorTestBatch])[0]
    errorTest /= batchNumTest
    return errorTest

  if Option.loadModel is not None:
    print 'Loading model from %s ...' % Option.loadModel,
    saver.restore(sess, Option.loadModel)
    print 'Finished',
    errorTestBest = getErrorTest()
    print 'Test:', errorTestBest

  else:
    # at the beginning, we discrete W
    sess.run([Net[0].W_q_op])

  print "\nOptimization Start!\n"
  for epoch in xrange(1000):
    # check lr_schedule
    if len(Option.lr_schedule) / 2:
      if epoch == Option.lr_schedule[0]:
        Option.lr_schedule.pop(0)
        lr_new = Option.lr_schedule.pop(0)
        if lr_new == 0:
          print 'Optimization Ended!'
          exit(0)
        lr_old = sess.run(Option.lr)
        sess.run(Option.lr.assign(lr_new))
        print 'lr: %f -> %f' % (lr_old, lr_new)

    print 'Epoch: %03d ' % (epoch),
    summary_writer = tf.summary.FileWriter(Option.trainDir, sess.graph)


    lossTotal = 0.
    errorTotal = 0
    t0 = time.time()
    for batchNum in tqdm(xrange(batchNumTrain), desc='Epoch: %03d' % epoch, leave=False, smoothing=0.1):
      if Option.debug is False:
        _, loss_delta, error_delta = sess.run([train_op, lossTrainBatch, errorTrainBatch])
      else:
        _, loss_delta, error_delta, H, W, W_q, gradH, gradW, gradW_q=\
        sess.run([train_op, lossTrainBatch, errorTrainBatch, Net[0].H, Net[0].W, Net[0].W_q, Net[0].gradsH, Net[0].gradsW, gradTrainBatch_quantize])
      assert not np.isnan(loss_delta), 'Model diverged with loss = NaN'

      # if step % 10 == 0:
      #   num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
      #   examples_per_sec = num_examples_per_step / duration
      #   sec_per_batch = duration / FLAGS.num_gpus

      #   format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
      #                 'sec/batch)')
      #   print (format_str % (datetime.now(), step, loss_value,
      #                        examples_per_sec, sec_per_batch))

      if batchNum % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, batchNum)

      # Save the model checkpoint periodically.
      # if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
      #   checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      #   saver.save(sess, checkpoint_path, global_step=step)
      lossTotal += loss_delta
      errorTotal += error_delta

    lossTotal /= batchNumTrain
    errorTotal /= batchNumTrain

    print 'Loss: %.4f Train: %.4f' % (lossTotal, errorTotal),

    # get test error
    errorTest = getErrorTest()
    print 'Test: %.4f FPS: %d' % (errorTest, numTrain / (time.time() - t0)),

    if epoch == 0:
      errorTestBest = errorTest
    if errorTest < errorTestBest:
      if Option.saveModel is not None:
        saver.save(sess, Option.saveModel)
        print 'S',
    if errorTest < errorTestBest:
      errorTestBest = errorTest
      print 'BEST',

    print ''


if __name__ == '__main__':
  main()

