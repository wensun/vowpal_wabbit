import os
import time
import numpy as np

available_shots = {"five":5, "three":3}
available_shots = {"three":3}

for key in available_shots.iterkeys():
    shots = available_shots[key]
    num_of_classes = 20000
    leaf_example_multiplier = 4#2#10
    lr = 0.1
    bits = 30
    passes = 2
    learn_at_leaf = 1 # 0
    hal_version = 1
    bandit = 1
    top_K = 1
    dream_repeats = 3
    num_queries = 1
    alpha = 0.1
    loss = "squared"#"logistic"

    tree_node = int(passes*(num_of_classes*shots/(np.log(num_of_classes*shots)/np.log(2)*leaf_example_multiplier)));
    train_data = "imagenet_{}_shots_training.txt".format(shots)
    test_data = "imagenet_{}_shots_testing.txt".format(shots)

    if os.path.exists(train_data) is not True:
        os.system("wget http://kalman.ml.cmu.edu/wen_datasets/{}".format(train_data))
    if os.path.exists(test_data) is not True:
        os.system("wget http://kalman.ml.cmu.edu/wen_datasets/{}".format(test_data))

    saved_model = "{}.vw".format(train_data)

    print "## Training.."
    start = time.time()
    os.system(".././vw --memory_tree {} --learn_at_leaf {} --hal_version {} --bandit {} --top_K {}\
              --leaf_example_multiplier {} --dream_repeats {} --num_queries {}\
              --Alpha {} --loss_function {}  -l {} -b {} -c --passes {} \
              --holdout_off {} -f {}".format(
                tree_node, learn_at_leaf, hal_version, bandit, top_K,
                  leaf_example_multiplier, dream_repeats, num_queries,
                  alpha, loss, lr, bits, passes,
                  train_data, saved_model))
    train_time = time.time() - start

    #test:
    print "## testing ..."
    start = time.time();
    os.system(".././vw  {} -i {}".format(test_data, saved_model))

    test_time = time.time() - start

    print "num_of_classes {}, multiplier {}, lr {}, bits {}, shots {}, passes {}, learn_at_leaf {}, loss {}".format(num_of_classes, leaf_example_multiplier, lr, bits, shots, passes, learn_at_leaf, loss)
    print "train time {}, test time {}".format(train_time, test_time)





