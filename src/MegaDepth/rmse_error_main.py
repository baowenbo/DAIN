import time
import torch
import sys

from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model

dataset_root = "/phoenix/S6/zl548/"
test_list_dir_l = '/phoenix/S6/zl548/MegaDpeth_code/test_list/landscape/'
input_height = 240
input_width = 320
is_flipped = False
shuffle = False

test_data_loader_l = CreateDataLoader(dataset_root, test_list_dir_l, input_height, input_width, is_flipped, shuffle)
test_dataset_l = test_data_loader_l.load_data()
test_dataset_size_l = len(test_data_loader_l)
print('========================= test images = %d' % test_dataset_size_l)
test_list_dir_p = '/phoenix/S6/zl548/MegaDpeth_code/test_list/portrait/'
input_height = 320
input_width = 240
test_data_loader_p = CreateDataLoader(dataset_root, test_list_dir_p, input_height, input_width, is_flipped, shuffle)
test_dataset_p = test_data_loader_p.load_data()
test_dataset_size_p = len(test_data_loader_p)
print('========================= test images = %d' % test_dataset_size_p)


model = create_model(opt)


def test(model):
    total_loss =0 
    toal_count = 0
    print("============================= TEST ============================")
    model.switch_to_eval()
    for i, data in enumerate(test_dataset_l):
        stacked_img = data['img_1']
        targets = data['target_1']    

        rmse_loss , count = model.evaluate_sc_inv(stacked_img, targets)

        total_loss += rmse_loss
        toal_count += count

        print('RMSE loss is', total_loss/float(toal_count))

    for i, data in enumerate(test_dataset_p):
        stacked_img = data['img_1']
        targets = data['target_1']    
        rmse_loss , count = model.evaluate_sc_inv(stacked_img, targets)

        total_loss += rmse_loss
        toal_count += count

        print('RMSE loss is', total_loss/float(toal_count))


    print('average RMSE loss is', total_loss/float(toal_count))

print("WE ARE IN TESTING RMSE!!!!")
test(model)
print("WE ARE DONE TESTING!!!")


print("We are done")
