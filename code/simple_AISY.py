import sys
import os
from dotenv import load_dotenv
from telegram_notification import telegram_notification


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

max_dataset_num = 256

nn_types = [mlp, cnn]
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
    for nn in nn_types:
        telegram_notification(f"The program is in dataset number {i} and nn: {str(nn.__name__)}")
        for batchsize in batch_sizes:
            for bit in range(0,15):#range of bits 0-16:
                try:
                    aisy = aisy_sca.Aisy()
                    aisy.set_resources_root_folder(resources_root_folder)
                    aisy.set_database_root_folder(databases_root_folder)
                    aisy.set_datasets_root_folder(datasets_root_folder)
                    aisy.set_database_name("database_ecg_simple.sqlite")
                    aisy.set_dataset(dataset_configuration)
                    aisy.set_aes_leakage_model(leakage_model='ID', byte=bit)
                    aisy.set_batch_size(batchsize)
                    aisy.set_epochs(25)
                    aisy.set_neural_network(nn)
                    aisy.run()
                except Exception as e:
                    telegram_notification(f"Error in nn: {str(nn.__name__)} batch: {batchsize} and bit: {bit}")

telegram_notification(f"The program has finished")
