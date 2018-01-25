import os
import time
import numpy as np


available_shots = {"three":3}

for shot in available_shots.iterkeys():
    print "{}-shot wikipara-10k ".format(shot)
    shots = available_shots[shot]

    num_of_classes = 10000
    lr = 0.1
    bits = 30
    passes = 3
    loss = "squared"

    train_data = "paradata10000_{}_shot.vw.train".format(shot)
    test_data = "paradata10000_{}_shot.vw.test".format(shot)
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
    start = time.time()
    os.system(command_recall_tree)
    rt_construct_time = time.time() - start

    start = time.time()
    os.system(".././vw {} -t -i {}".format(test_data, saved_model_rt))
    rt_test_time = time.time() - start
    print "Recall Tree construct time {}, and inference time is {}".format(rt_construct_time, rt_test_time/10000)


    print "################ Log Multi ##################"
    start = time.time()
    os.system(command_log_multi)
    lm_construct_time = time.time() - start

    start = time.time()
    os.system(".././vw {} -t -i {}".format(test_data, saved_model_lm))
    lm_test_time = time.time() - start
    print "log multi construct time {}, and inference time is {}".format(lm_construct_time, lm_test_time / 10000)

    print "############### OAA #######################"
    start = time.time()
    os.system(command_oaa)
    oaa_construct_time = time.time() - start

    start = time.time()
    os.system(".././vw {} -t -i {}".format(test_data, saved_model_oaa))
    oaa_test_time = time.time() - start
    print "oaa construct time {}, and inference time is {}".format(oaa_construct_time, oaa_test_time / 10000)
    



