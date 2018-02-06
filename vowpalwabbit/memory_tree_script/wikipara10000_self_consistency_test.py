import os
import time
import numpy as np
from IPython import embed


available_shots = {'three':3, "one":1}
available_shots = {'one':1}

for shot in available_shots.iterkeys():
    print "## perform experiments on {}-shot wikipara-10K ##".format(shot)
    shots = available_shots[shot]
    num_of_classes = 10000
    leaf_example_multiplier = 4
    lr = 0.1
    bits = 28#30
    passes = 3
    hal_version = 1
    dream_repeats = 0
    num_queries = int(np.log(passes*shots*num_of_classes)/np.log(2.))
    alpha = 0.5
    learn_at_leaf = 0
    #task = 1
    loss = "squared"

    tree_node = int(2*passes*(num_of_classes*shots/(np.log(num_of_classes*shots)/np.log(2)*leaf_example_multiplier)));

    train_data = "paradata10000_{}_shot.vw.train".format(shot)
    test_data = "paradata10000_{}_shot.vw.train".format(shot)
    if os.path.exists(train_data) is not True:
        os.system("wget http://kalman.ml.cmu.edu/wen_datasets/{}".format(train_data))
    if os.path.exists(test_data) is not True:
        os.system("wget http://kalman.ml.cmu.edu/wen_datasets/{}".format(test_data))

    saved_model = "{}.vw".format(train_data)

    print "## Training..."
    start = time.time()
    os.system(".././vw --memory_tree {} --learn_at_leaf {} --hal_version {} \
            --leaf_example_multiplier {} --dream_repeats {} --num_queries {} \
            --Alpha {} -l {} -b {} -c --passes {} \
            --loss_function {} --holdout_off {} -f {}".format(
                tree_node, learn_at_leaf, hal_version, 
                leaf_example_multiplier, dream_repeats, num_queries, 
                alpha, lr, bits, passes, 
                loss, train_data, saved_model))

    train_time = time.time() - start

    #test:
    print "## Testing..."
    start = time.time();
    os.system(".././vw {} -i {}".format(test_data, saved_model))

    test_time = time.time() - start


    print "## train time {}, and test time {}".format(train_time, test_time)





