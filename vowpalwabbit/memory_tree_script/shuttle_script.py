import os
import time
import numpy as np


num_of_classes = 7
num_examples = 46000
leaf_example_multiplier = 2
lr = 0.1
bits = 28
passes = 1
learn_at_leaf = 0
router_error_feature = 0
random = 0
loss = "squared"

tree_node = int(passes*(num_examples/(np.log(num_examples)/np.log(2)*leaf_example_multiplier)));

train_data = "shuttle.preprocess.data"
if os.path.exists(train_data) is not True:
    os.system("wget http://kalman.ml.cmu.edu/wen_datasets/shuttle.preprocess.data")
saved_model = "shuttle_tree.vw"

start = time.time()
os.system(".././vw --memory_tree {} --learn_at_leaf {} --router_error_feature {} --leaf_example_multiplier {} --loss_function {} --random_weights {} -l {} -b {} -c --passes {} --holdout_off {} -f {}".format(
                tree_node, learn_at_leaf, router_error_feature, leaf_example_multiplier, loss, random, lr, bits, passes, train_data, saved_model))
train_time = time.time() - start

#test:
print "train time {}".format(train_time)





