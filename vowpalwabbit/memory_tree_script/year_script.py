import os
import time
import numpy as np


num_of_classes = 90
num_examples = 464000
leaf_example_multiplier = 2
lr = 1
bits = 29
passes = 1
learn_at_leaf = 0
router_error_feature = 0
random = 0
loss = "squared"

tree_node = int(passes*(num_examples/(np.log(num_examples)/np.log(2)*leaf_example_multiplier)));

train_data = "year.preprocess.data.vw"
if os.path.exists(train_data) is not True:
    os.system("wget http://kalman.ml.cmu.edu/wen_datasets/{}".format(train_data))
saved_model = "{}_tree.vw".format(train_data)

start = time.time()
os.system(".././vw --memory_tree {} --learn_at_leaf {} --leaf_example_multiplier {} --loss_function {} --random_weights {} -l {} -b {} -q {} -c --passes {} --holdout_off {} -f {}".format(
                tree_node, learn_at_leaf, leaf_example_multiplier, loss, random, lr, bits, 'ab', passes, train_data, saved_model))
train_time = time.time() - start

#test:
print "train time {}".format(train_time)





