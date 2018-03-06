import os
import time
import numpy as np


print "Aloi baselines.."

num_of_classes = 1000
shots = 100
lr = 0.1
bits = 29
passes = 3 # 5
loss = "squared"
train_data = "aloi_train.vw"
test_data = "aloi_test.vw"
if os.path.exists(train_data) is not True:
    os.system("wget http://kalman.ml.cmu.edu/wen_datasets/{}".format(train_data))
if os.path.exists(test_data) is not True:
    os.system("wget http://kalman.ml.cmu.edu/wen_datasets/{}".format(test_data))



saved_model_oaa = "oaa.vw"
command_oaa = ".././vw --oaa {} -b {} \
--loss_function {} -l {} -c --passes {} --holdout_off {} -f {}".format(
    num_of_classes, bits, loss, lr, passes, train_data, saved_model_oaa
)

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


print "#################### Recall Tree ################"
start = time.time()
os.system(command_recall_tree)
rt_construct_time = time.time() - start
start = time.time()
os.system(".././vw {} -t -i {}".format(test_data, saved_model_rt))
rt_test_time = time.time() - start
print "Recall Tree construct time {}, and inference time is {}".format(rt_construct_time, rt_test_time/10800)


print "################ Log Multi ##################"
start = time.time()
os.system(command_log_multi)
lm_construct_time = time.time() - start
start = time.time()
os.system(".././vw {} -t -i {}".format(test_data, saved_model_lm))
lm_test_time = time.time() - start
print "log multi construct time {}, and inference time is {}".format(lm_construct_time, lm_test_time / 10800)


print "############### OAA #######################"
start = time.time()
os.system(command_oaa)
oaa_construct_time = time.time() - start
start = time.time()
os.system(".././vw {} -t -i {}".format(test_data, saved_model_oaa))
oaa_test_time = time.time() - start
print "oaa construct time {}, and inference time is {}".format(oaa_construct_time, oaa_test_time / 10800)

