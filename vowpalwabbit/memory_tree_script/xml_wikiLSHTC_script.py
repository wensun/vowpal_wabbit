import os
import time
import numpy as np
from IPython import embed

print "perform experiments on WikiLSHTC (multilabel)"
leaf_example_multiplier = 4
lr = 0.1
bits = 28
alpha = 0.3
passes = 4
learn_at_leaf = 0
num_queries = 1  #does not really use
hal_version = 1 #does not really use
loss = "squared"
dream_repeats = 3
Precision_at_K = 5

num_examples = 1778351
tree_node = int(num_examples/(np.log(num_examples)/np.log(2)*leaf_example_multiplier))
train_data = "wikiLSHTC_train.mat.mult_label.vw.txt"
test_data = "wikiLSHTC_test.mat.mult_label.vw.txt"
if os.path.exists(train_data) is not True:
        os.system("wget http://kalman.ml.cmu.edu/wen_datasets/{}".format(train_data))
if os.path.exists(test_data) is not True:
        os.system("wget http://kalman.ml.cmu.edu/wen_datasets/{}".format(test_data))

saved_model = "{}.vw".format(train_data)

print "## Training..."
start = time.time()
#train_data = 'tmp_rcv1x.vw.txt'
os.system(".././vw --memory_tree_xml {} --learn_at_leaf {} --dream_repeats {} --Precision_at_K {} --hal_version {} --num_queries {} --leaf_example_multiplier {} --Alpha {} -l {} -b {} -c --passes {} --loss_function {} --holdout_off {} -f {}".format(
                tree_node, learn_at_leaf, dream_repeats, 
                Precision_at_K, hal_version, num_queries, 
                leaf_example_multiplier, 
                alpha, lr, bits, 
                passes, loss, 
                train_data, saved_model))
train_time = time.time() - start

print "## Testing..."
start = time.time()
os.system(".././vw {} -i {}".format(test_data, saved_model))
test_time = time.time() - start
print "## train time {}, and test time {}".format(train_time, test_time)

