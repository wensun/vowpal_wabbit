import os
import time
import numpy as np
from IPython import embed


#for shot in available_shots.iterkeys():
print "## perform experiments on aloi ##"
num_of_classes = 1000
leaf_example_multiplier = 2
shots = 100
lr = 0.001
bits = 30
passes = 3
learn_at_leaf = 1
hal_version = 0
loss = "squared"

tree_node = int(2*passes*(num_of_classes*shots/(np.log(num_of_classes*shots)/np.log(2)*leaf_example_multiplier)));

train_data = "aloi_train.vw"
test_data = "aloi_test.vw"
  
saved_model = "{}.vw".format(train_data)

print "## Training..."
start = time.time()
os.system(".././vw --memory_tree {} --learn_at_leaf {} --hal_version {} --leaf_example_multiplier {} -l {} -b {} -c --passes {} --loss_function {} --holdout_off {} -f {}".format(
                tree_node, learn_at_leaf, hal_version, leaf_example_multiplier, lr, bits, passes, loss, train_data, saved_model))
train_time = time.time() - start

    #test:
print "## Testing..."
start = time.time();
os.system(".././vw {} -i {}".format(test_data, saved_model))

test_time = time.time() - start


print "## train time {}, and test time {}".format(train_time, test_time)





