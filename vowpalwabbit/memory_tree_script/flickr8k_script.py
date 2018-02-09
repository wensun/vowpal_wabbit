import os
import time
import numpy as np

seeds = [100,200,300,400,500]

num_of_examples = 7000
leaf_example_multiplier = 10
lr = 0.1
bits = 30
passes = 2
num_passes =2
learn_at_leaf = 1
task = 2
loss_function = "squared"
alpha = 0.1
lam = 0.0

train_N = passes*num_of_examples

print "lr {}, bits {}, passes {}, learn_at_leaf {}, task {}, loss {}, alpha {}, lam {}".format(
    lr, bits, passes, learn_at_leaf, task, loss_function, alpha, lam)


tree_node = int(passes*(num_of_examples/(np.log(num_of_examples)/np.log(2)*leaf_example_multiplier)));


for seed in seeds:
    print "################# at seed {} #########".format(seed)
    train_data = "flickr8k_train2_test_normalized_{}.vw".format(seed)
    if os.path.exists(train_data) is not True:
        os.system("wget http://kalman.ml.cmu.edu/wen_datasets/flickr8k/{}".format(train_data))
        #train_data = "/data2/wen_vw_datasets/pascal_train_normalized.vw"
        #train_data = "/data2/wen_vw_datasets/flickr8k_train2_test_normalized_400.vw"
        #test_data = "/data2/wen_vw_datasets/pascal_test_normalized.vw"
        saved_model = "flickr8k_tree_{}.vw".format(learn_at_leaf)

    start = time.time()
    os.system(".././vw --memory_tree {} --learn_at_leaf {} --leaf_example_multiplier {} --task {} --num_passes {}\
              --train_N {} --loss_function {} -l {} -b {} {}".format(
                tree_node, learn_at_leaf, leaf_example_multiplier, task, num_passes, train_N, loss_function, lr, bits, train_data))
    train_time = time.time() - start

    #test:
    #start = time.time();
    #os.system(".././vw {} -i {}".format(test_data, saved_model))

    #test_time = time.time() - start
    print train_time#, test_time





