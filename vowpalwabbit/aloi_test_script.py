import time 
import os

max_nodes = 4000
leaf_example_multi = 5
learn_at_leaf = 1
rand_weight = 1
lr = 0.0005
bits = 27
num_passes = 3
loss_function = 'logistic'
data_file = '../demo/recall_tree/aloi_train.vw'
model_name= 'aloi_mt_learn_leaf.vw'

command = './vw --memory_tree {} --leaf_example_multiplier {} --learn_at_leaf {} --random_weights {} -l {} -b {} --passes {} -c --holdout_off --loss_function {} {} -f {}'.format(
            max_nodes,
            leaf_example_multi,
            learn_at_leaf,
            rand_weight,
            lr,
            bits,
            num_passes,
            loss_function,
            data_file,
            model_name
    )

#train
train_start = time.time()
os.system(command)
train_end = time.time()
train_time = train_end - train_start;
print "training time: {}".format(train_time);

#test:
test_file = '../demo/recall_tree/aloi_test.vw.gz'
test_command = './vw {} -i {}'.format(test_file, model_name);
test_start = time.time();
os.system(test_command);
test_end = time.time();
test_time = test_end - test_start;
print "test time: {}".format(test_time);






