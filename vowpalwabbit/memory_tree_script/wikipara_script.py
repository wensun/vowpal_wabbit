import os
import time
import numpy as np


num_of_classes = 50000
leaf_example_multiplier = 2
lr = 0.1
bits = 30
shots = 3
passes = 3
learn_at_leaf = 1
#task = 1
loss = "squared"


tree_node = int(2*passes*(num_of_classes*shots/(np.log(num_of_classes*shots)/np.log(2)*leaf_example_multiplier)));


if shots == 3 and (num_of_classes != 10000 and num_of_classes != 50000):
    train_data = "/data2/wen_vw_datasets/paradata{}_three_shot.vw.train.gz".format(num_of_classes)
    test_data = "/data2/wen_vw_datasets/paradata{}_three_shot.vw.test.gz".format(num_of_classes);
elif shots == 1:
    train_data = "/data2/wen_vw_datasets/paradata{}_one_shot.vw.train.gz".format(num_of_classes)
    test_data = "/data2/wen_vw_datasets/paradata{}_one_shot.vw.test.gz".format(num_of_classes)

elif shots == 10:
    train_data = "/data2/wen_vw_datasets/paradata{}_ten_shot.vw.train.gz".format(num_of_classes)
    test_data = "/data2/wen_vw_datasets/paradata{}_ten_shot.vw.test.gz".format(num_of_classes)
elif shots == 3 and (num_of_classes == 10000 or num_of_classes == 50000):
    train_data = "/data2/wen_vw_datasets/paradata{}.vw.train".format(num_of_classes)
    test_data = "/data2/wen_vw_datasets/paradata{}.vw.test".format(num_of_classes)

saved_model = "wikipara_tree_{}.vw".format(num_of_classes)


start = time.time()
os.system(".././vw --memory_tree {} --learn_at_leaf {} --leaf_example_multiplier {} -l {} -b {} -c --passes {} --loss_function {} --holdout_off {} -f {}".format(
                tree_node, learn_at_leaf, leaf_example_multiplier, lr, bits, passes, loss, train_data, saved_model))
train_time = time.time() - start

#test:
start = time.time();
os.system(".././vw {} -i {}".format(test_data, saved_model))

test_time = time.time() - start


print train_time, test_time





