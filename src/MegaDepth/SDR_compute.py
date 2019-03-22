import time
import torch
import sys

from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader_TEST
from models.models import create_model

dataset_root = "/phoenix/S6/zl548/"
test_list_dir_l = dataset_root + '/MegaDpeth_code/test_list/landscape/'
input_height = 240
input_width = 320
test_data_loader_l = CreateDataLoader_TEST(dataset_root, test_list_dir_l, input_height, input_width)
test_dataset_l = test_data_loader_l.load_data()
test_dataset_size_l = len(test_data_loader_l)
print('========================= test L images = %d' % test_dataset_size_l)

test_list_dir_p = dataset_root + '/MegaDpeth_code/test_list/portrait/'
input_height = 320
input_width = 240
test_data_loader_p = CreateDataLoader_TEST(dataset_root, test_list_dir_p, input_height, input_width)
test_dataset_p = test_data_loader_p.load_data()
test_dataset_size_p = len(test_data_loader_p)
print('========================= test P images = %d' % test_dataset_size_p)

model = create_model(opt)

batch_size = 32
diw_index = 0 
total_steps = 0
best_loss = 100

error_list = [0 , 0, 0]
total_list = [0 , 0, 0]

list_l = range(test_dataset_size_l)
list_p = range(test_dataset_size_p)


def test_SDR(model):
    total_loss =0 
    # count = 0
    print("============================= TEST SDR============================")
    model.switch_to_eval()
    diw_index = 0

    for i, data in enumerate(test_dataset_l):
        stacked_img = data['img_1']
        targets = data['target_1']    
        error, samples = model.evaluate_SDR(stacked_img, targets)

        for j in range(0,3):
            error_list[j] += error[j]
            total_list[j] += samples[j]

        print("EQUAL  ", error_list[0]/float(total_list[0]))
        print("INEQUAL    ", error_list[1]/float(total_list[1]))
        print("TOTAL    ",error_list[2]/float(total_list[2]))

    for i, data in enumerate(test_dataset_p):
        stacked_img = data['img_1']
        targets = data['target_1']    

        error, samples = model.evaluate_SDR(stacked_img, targets)

        for j in range(0,3):
            error_list[j] += error[j]
            total_list[j] += samples[j]

        print("EQUAL  ", error_list[0]/float(total_list[0]))
        print("INEQUAL    ", error_list[1]/float(total_list[1]))
        print("TOTAL    ",error_list[2]/float(total_list[2]))


    print("=========================================================SDR Summary =====================")
    print("Equal SDR:\t" , float(error_list[0])/ float(total_list[0]))
    print("Unequal SDR:\t" , float(error_list[1])/ float(total_list[1]))
    print("SDR:\t" , float(error_list[2])/ float(total_list[2]))


print("WE ARE TESTING SDR!!!!")
test_SDR(model)
