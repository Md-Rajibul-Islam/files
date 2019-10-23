import os
import shutil
import random


# copying data to train_0 and train_1 according to class_0 and class_1 respectively
#root_dir = '/Users/mdrajibulislam/Desktop/CM2003/Labs/data_extra'

root_dir = '/Users/mdrajibulislam/Desktop/CM2003/Labs/project_data_breast_cancer_classification'
output_dir_0 = '/Users/mdrajibulislam/Desktop/CM2003/Labs/CM2003_project_data/train/0'
output_dir_1 = '/Users/mdrajibulislam/Desktop/CM2003/Labs/CM2003_project_data/train/1'
ref = 4
i = 0
for root, dirs, files in os.walk(root_dir):
    i = i+1
    if i == 1:
        continue
    else:
        number_of_files = len(os.listdir(root))
        if number_of_files > ref:
            name_list = os.listdir(root)
            for filename in name_list:
                file_to_copy = root + '/' + filename
                if file_to_copy[-5] == '1':
                    shutil.copy(file_to_copy, output_dir_1)
                else:
                    shutil.copy(file_to_copy, output_dir_0)
print('Finished for copying!')


# Moving 20 % data files from train to test
root_dir = '/Users/mdrajibulislam/Desktop/CM2003/Labs/CM2003_project_data/train/0'
output_dir = '/Users/mdrajibulislam/Desktop/CM2003/Labs/CM2003_project_data/test/0'
ref = 1

for root, dirs, files in os.walk(root_dir):
    number_of_files = len(os.listdir(root))
    random.shuffle(os.listdir(root))
    if number_of_files > ref:
        ref_move = int(round(0.2 * number_of_files))   # randomly copying 20%
        for i in range(ref_move):
            file_list = random.choice(os.listdir(root))
            file_to_move = root + '/' + file_list
            if os.path.isfile(file_to_move) == True:
                shutil.move(file_to_move, output_dir)
    else:
        continue
print('Finished for testing_0 !')

root_dir = '/Users/mdrajibulislam/Desktop/CM2003/Labs/CM2003_project_data/train/1'
output_dir = '/Users/mdrajibulislam/Desktop/CM2003/Labs/CM2003_project_data/test/1'
ref = 1

for root, dirs, files in os.walk(root_dir):
    number_of_files = len(os.listdir(root))
    random.shuffle(os.listdir(root))
    if number_of_files > ref:
        ref_move = int(round(0.2 * number_of_files))   # randomly copying 20%
        for i in range(ref_move):
            file_list = random.choice(os.listdir(root))
            file_to_move = root + '/' + file_list
            if os.path.isfile(file_to_move) == True:
                shutil.move(file_to_move, output_dir)
    else:
        continue
print('Finished for testing_1 !')


# Moving 10 % of remaining data files from train to validation
root_dir = '/Users/mdrajibulislam/Desktop/CM2003/Labs/CM2003_project_data/train/0'
output_dir = '/Users/mdrajibulislam/Desktop/CM2003/Labs/CM2003_project_data/validation/0'
ref = 1

for root, dirs, files in os.walk(root_dir):
    number_of_files = len(os.listdir(root))
    random.shuffle(os.listdir(root))
    if number_of_files > ref:
        ref_move = int(round(0.1 * number_of_files))   # randomly copying 10%
        for i in range(ref_move):
            file_list = random.choice(os.listdir(root))
            file_in_track = root
            file_to_move = root + '/' + file_list
            if os.path.isfile(file_to_move) == True:
                shutil.move(file_to_move, output_dir)
    else:
        continue
print('Finished for validation_0  !')

root_dir = '/Users/mdrajibulislam/Desktop/CM2003/Labs/CM2003_project_data/train/1'
output_dir = '/Users/mdrajibulislam/Desktop/CM2003/Labs/CM2003_project_data/validation/1'
ref = 1

for root, dirs, files in os.walk(root_dir):
    number_of_files = len(os.listdir(root))
    random.shuffle(os.listdir(root))
    if number_of_files > ref:
        ref_move = int(round(0.1 * number_of_files))   # randomly copying 10%
        for i in range(ref_move):
            file_list = random.choice(os.listdir(root))
            file_in_track = root
            file_to_move = root + '/' + file_list
            if os.path.isfile(file_to_move) == True:
                shutil.move(file_to_move, output_dir)
    else:
        continue
print('Finished for validation_1 !')








