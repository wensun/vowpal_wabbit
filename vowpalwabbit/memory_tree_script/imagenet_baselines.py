import os
import time
import numpy as np


available_shots = {"one":1}

for shot in available_shots.iterkeys():
    print "{}-shot wikipara-10k ".format(shot)
    shots = available_shots[shot]

    num_of_classes = 22000
    lr = 0.1
    bits = 30
    passes = 3
    loss = "squared"

    train_data = "imagenet_{}_shots_training.txt".format(shots)
    test_data = "imagenet_{}_shots_testing.txt".format(shots)
    if os.path.exists(train_data) is not True:
        os.system("wget http://kalman.ml.cmu.edu/wen_datasets/{}".format(train_data))
    if os.path.exists(test_data) is not True:
        os.system("wget http://kalman.ml.cmu.edu/wen_datasets/{}".format(test_data))

    saved_model = "{}.vw".format(train_data)

    saved_model_rt = "recall_tree.vw"
    command_recall_tree = ".././vw --recall_tree {} -b {} \
    --loss_function {} -l {} -c --passes {} --holdout_off {} -f {}".format(
        num_of_classes, bits, loss, lr, passes, train_data, saved_model_rt 
    )
    

    saved_model_lm = "log_multi.vw"
    command_log_multi = ".././vw --log_multi {} -b {} \
    --loss_function {} -l {} -c --passes {} --holdout_off {} -f {}".format(
        num_of_classes, bits, loss, lr, passes, train_data, saved_model_lm
    )

    saved_model_oaa = "oaa.vw"
    command_oaa = ".././vw --oaa {} -b {} \
    --loss_function {} -l {} -c --passes {} --holdout_off {} -f {}".format(
        num_of_classes, bits, loss, lr, passes, train_data, saved_model_oaa
    )


    print "#################### Recall Tree ################"
    os.system(command_recall_tree)
    start = time.time()
    os.system(".././vw {} -t -i {}".format(test_data, saved_model_rt))
    rt_test_time = time.time() - start
    print "Recall Tree inference time is {}".format(rt_test_time/10000)


    print "################ Log Multi ##################"
    os.system(command_log_multi)
    start = time.time()
    os.system(".././vw {} -t -i {}".format(test_data, saved_model_lm))
    lm_test_time = time.time() - start
    print "log multi inference time is {}".format(lm_test_time / 10000)

    print "############### OAA #######################"
    os.system(command_oaa)
    start = time.time()
    os.system(".././vw {} -t -i {}".format(test_data, saved_model_oaa))
    oaa_test_time = time.time() - start
    print "oaa inference time is {}".format(oaa_test_time / 10000)
    



