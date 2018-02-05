import os
import time
import numpy as np


num_of_classes = 7
num_examples = 46000
leaf_example_multiplier = 10
lr = 0.1
bits = 29
Alpha = 0.1
passes = 1
learn_at_leaf = 1
random = 0
loss = "squared"

tree_node = int(passes*(num_examples/(np.log(num_examples)/np.log(2)*leaf_example_multiplier)));

train_data = "shuttle.preprocess.data"
if os.path.exists(train_data) is not True:
    os.system("wget http://kalman.ml.cmu.edu/wen_datasets/shuttle.preprocess.data")
saved_model = "shuttle_tree.vw"

start = time.time()
os.system(".././vw --memory_tree {} --learn_at_leaf {} --leaf_example_multiplier {} --Alpha {} --loss_function {} --random_weights {} -l {} -b {} -q {} -c --passes {} --holdout_off {} -f {}".format(
                tree_node, learn_at_leaf,  leaf_example_multiplier, Alpha, loss, random, lr, bits, 'ab', passes, train_data, saved_model))
train_time = time.time() - start

#test:
print "train time {}".format(train_time)





