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
    # Leer el contenido del archivo txt
    with open(config_file_folder+'/config_dataset_file_'+str(i)+'_FPGA.txt', 'r') as file:
        data = file.read()

    # Convertir el contenido leído de string a diccionario
    dataset_configuration = json.loads(data)

    # Verificar el contenido de la variable
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
                aisy.set_database_name("database_ecg_hyper_selec_rand_search_cnn.sqlite")
                aisy.set_dataset(dataset_configuration)
                aisy.set_aes_leakage_model(leakage_model='ID', byte=byte)
                
                random_search = {
                    "neural_network": "cnn",
                    "hyper_parameters_search": {
                        'conv_layers': {"min": 1, "max": 3, "step": 1},
                        'kernel_1': {"min": 4, "max": 20, "step": 1},
                        'kernel_2': {"min": 4, "max": 20, "step": 1},
                        'kernel_3': {"min": 4, "max": 20, "step": 1},
                        'stride_1': {"min": 1, "max": 4, "step": 1},
                        'stride_2': {"min": 1, "max": 4, "step": 1},
                        'stride_3': {"min": 1, "max": 4, "step": 1},
                        'filters_1': {"min": 8, "max": 32, "step": 4},
                        'filters_2': "double_from_previous_convolution",
                        'filters_3': "double_from_previous_convolution",
                        'pooling_type_1': ["Average", "Max"],
                        'pooling_type_2': "equal_from_previous_pooling",
                        'pooling_type_3': "equal_from_previous_pooling",
                        'pooling_size_1': {"min": 1, "max": 1, "step": 1},
                        'pooling_size_2': "equal_from_previous_pooling",
                        'pooling_size_3': "equal_from_previous_pooling",
                        'pooling_stride_1': {"min": 2, "max": 2, "step": 1},
                        'pooling_stride_2': {"min": 2, "max": 2, "step": 1},
                        'pooling_stride_3': {"min": 2, "max": 2, "step": 1},
                        'neurons': {"min": 100, "max": 400, "step": 100},
                        'layers': {"min": 2, "max": 3, "step": 1},
                        'learning_rate': [0.005, 0.001, 0.0005, 0.0001],
                        'activation': ["relu", "selu"],
                        'epochs': {"min": 5, "max": 5, "step": 1},
                        'batch_size': {"min": 100, "max": 1000, "step": 100},
                        'optimizer': ["Adam", "RMSprop"]
                    },
                    "structure": {
                        "use_pooling_after_convolution": True,  # only for CNNs
                        "use_pooling_before_first_convolution": False,
                        "use_pooling_before_first_dense": False,  # only for MLPs
                        "use_batch_norm_after_pooling": True,
                        "use_batch_norm_before_pooling": False,
                        "use_batch_norm_after_convolution": False,
                        "use_dropout_after_dense_layer": False,
                        "use_dropout_before_dense_layer": False,
                    },
                    "metric": "guessing_entropy",
                    "stop_condition": False,
                    "stop_value": 1.0,
                    "max_trials": 10,
                    "train_after_search": True
                }

                aisy.run(random_search=random_search)

            except Exception as e:
                telegram_notification(f"Error in byte: {byte}")

telegram_notification(f"The program has finished")
