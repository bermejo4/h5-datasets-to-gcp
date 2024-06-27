import sys
import os
from dotenv import load_dotenv
from utils.telegram_notification import telegram_notification


load_dotenv()

aisy_path = os.getenv("AISY_PATH")
sys.path.append(os.path.abspath(aisy_path))

import aisy_sca
from app import *
from custom.custom_models.neural_networks import *
import json


datasets_root_folder = os.getenv("DATASETS_ROOT_FOLDER")
resources_root_folder = os.getenv("RESOURCES_ROOT_FOLDER")
databases_root_folder = os.getenv("DATABASES_ROOT_FOLDER")

config_file_folder = os.getenv("CONFIG_FILE_FOLDER")


max_dataset_num = 5

batch_sizes = [400, 200]

telegram_notification(f"The program has started")
for i in range(0, max_dataset_num):
    # Reading the txt file content: 
    with open(config_file_folder+'/config_dataset_file_'+str(i)+'_FPGA.txt', 'r') as file:
        data = file.read()

    # Transform the content from a string to a dictionary:
    dataset_configuration = json.loads(data)
    print(dataset_configuration)

    # AISY SCA configuration
    telegram_notification(f"The program is in dataset number {i}")
    for batchsize in batch_sizes:
        for byte in range(0,15):#range of bits 0-16:
            try:
                aisy = aisy_sca.Aisy()
                aisy.set_resources_root_folder(resources_root_folder)
                aisy.set_database_root_folder(databases_root_folder)
                aisy.set_datasets_root_folder(datasets_root_folder)
                aisy.set_database_name("database_ecg_hyper_selec_grid_search_cnn.sqlite")
                aisy.set_dataset(dataset_configuration)
                aisy.set_aes_leakage_model(leakage_model='ID', byte=byte)
                aisy.set_batch_size(batchsize)
                aisy.set_epochs(25)
                grid_search = {
                    "neural_network": "cnn",
                    "hyper_parameters_search": {
                        'conv_layers': [2, 3],
                        'kernel_1': [10],
                        'kernel_2': [10],
                        'kernel_3': [10],
                        'stride_1': [5],
                        'stride_2': [5],
                        'stride_3': [5],
                        'filters_1': [8],
                        'filters_2': "double_from_previous_convolution",
                        'filters_3': "double_from_previous_convolution",
                        'pooling_type_1': ["Average"],
                        'pooling_type_2': ["Average"],
                        'pooling_type_3': ["Average"],
                        'pooling_size_1': [2],
                        'pooling_size_2': [2],
                        'pooling_size_3': [2],
                        'pooling_stride_1': [2],
                        'pooling_stride_2': [2],
                        'pooling_stride_3': [2],
                        'neurons': [100],
                        'layers': [1, 2],
                        'dropout_rate': [0.50],
                        'learning_rate': [0.001],
                        'activation': ["selu"],
                        'optimizer': ["Adam"]
                    },
                    "structure": {
                        "use_pooling_after_convolution": True,  # only for CNNs
                        "use_pooling_before_first_convolution": False,
                        "use_pooling_before_first_dense": False,  # only for MLPs
                        "use_batch_norm_after_pooling": True,
                        "use_batch_norm_before_pooling": False,
                        "use_batch_norm_after_convolution": False,
                        "use_dropout_after_dense_layer": True,
                        "use_dropout_before_dense_layer": False,
                    },
                    "metric": "guessing_entropy",
                    "stop_condition": False,
                    "stop_value": 1.0,
                    "train_after_search": True
                }

                aisy.run(grid_search=grid_search)

            except Exception as e:
                telegram_notification(f"Error in batch: {batchsize} and bit: {byte}")

telegram_notification(f"The program has finished")

