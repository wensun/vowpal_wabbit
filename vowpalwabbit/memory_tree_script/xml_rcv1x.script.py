import os
import time
import numpy as np
from IPython import embed

print("perform experiments on rcv1x (multilabel)")
leaf_example_multiplier = 2
lr = 0.1
bits = 30
alpha = 0.1
passes = 6 #4
learn_at_leaf = 1
use_oas = 0
dream_at_update = 0# 1
num_queries = 1  #does not really use
hal_version = 1 #does not really use
loss = "squared"
dream_repeats = 3
Precision_at_K = 5

num_examples = 630000
max_num_labels = 2456

tree_node = int(num_examples/(np.log(num_examples)/np.log(2)*leaf_example_multiplier))
train_data = "rcv1x_train.mat.mult_label.vw.txt"
test_data = "rcv1x_test.mat.mult_label.vw.txt"
if os.path.exists(train_data) is not True:
        os.system("wget http://kalman.ml.cmu.edu/wen_datasets/{}".format(train_data))
if os.path.exists(test_data) is not True:
        os.system("wget http://kalman.ml.cmu.edu/wen_datasets/{}".format(test_data))

saved_model = "{}.vw".format(train_data)

print "## Training..."
start = time.time()
#train_data = 'tmp_rcv1x.vw.txt'
os.system(".././vw --memory_tree_xml {} --learn_at_leaf {} --dream_at_update {}\
                --max_number_of_labels {} --dream_repeats {} \
                --Precision_at_K {} --hal_version {} --oas {} \
                --num_queries {} --leaf_example_multiplier {} --Alpha {} -l {} -b {} -c --passes {} --loss_function {} {} -f {}".format(
                tree_node, learn_at_leaf, dream_at_update,
                max_num_labels, dream_repeats, 
                Precision_at_K, hal_version, use_oas, 
                num_queries, leaf_example_multiplier, 
                alpha, lr, bits, 
                passes, loss, 
                train_data, saved_model))
train_time = time.time() - start

print "## Testing..."
start = time.time()
os.system(".././vw {} -i {}".format(test_data, saved_model))
test_time = time.time() - start
print "## train time {}, and test time {}".format(train_time, test_time)

