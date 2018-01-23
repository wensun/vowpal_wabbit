import os
import time
import numpy as np

num_of_examples = 82783
leaf_example_multiplier = 10
lr = 0.1
bits = 30
passes = 2
learn_at_leaf = 0
task = 2
loss_function = "squared"
alpha = 0.1
lam = 0.0

train_N = passes * num_of_examples

print "lr {}, bits {}, passes {}, learn_at_leaf {}, task {}, loss {}, alpha {}, lam {}".format(lr,
bits, passes, learn_at_leaf, task, loss_function, alpha, lam)

tree_node = int(passes*(num_of_examples/(np.log(num_of_examples)/np.log(2)*leaf_example_multiplier)))

train_data = "mscoco_train2_test_normalized_small.vw"
if os.path.exists(train_data) is not True:
    os.system("wget http://kalman.ml.cmu.edu/wen_datasets/mscoco/{}".format(train_data))

saved_model = "mscoco_{}.vw".format(learn_at_leaf)

start = time.time()
os.system(".././vw --memory_tree {} --learn_at_leaf {} --leaf_example_multiplier {} --task {} --train_N {} --loss_function {} -l {} -b {} {}").format(tree_node, learn_at_leaf, leaf_example_multiplier, task, train_N, loss_function, lr, bits, train_data)

train_time = time.time() - start

print train_time
