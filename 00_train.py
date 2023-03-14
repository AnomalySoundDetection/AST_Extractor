"""
 @file   00_train.py
 @brief  Script for training
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
import time
import gc
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
# from import
from tqdm import tqdm
# original lib
import common as com
import random
from AST_Model import load_extractor
from Dataset import AudioDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from RealNVP import BuildFlow
########################################################################

########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)

########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    param = com.load_yaml()
        
    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # training device
    device_0 = torch.device('cuda:0')
    device_1 = torch.device('cuda:1')

    # training info
    epochs = int(param["fit"]["epochs"])
    batch_size = int(param["fit"]["batch_size"])

    # 11/30/22: I decouple the dataset and the following hyper-parameters to make it easier to adapt to new datasets
    # dataset spectrogram mean and std, used to normalize the input
    # norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
    # target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128}
    # # if add noise for data augmentation, only use for speech commands
    # noise = {'audioset': False, 'esc50': False, 'speechcommands':True}

    # audio config
    audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'dcase', 
                    'mode': 'train', 'mean': -4.2677393, 'std': 4.5689974, 'noise': False}
    
    val_audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'dcase', 
                        'mode': 'test', 'mean': -4.2677393, 'std': 4.5689974, 'noise': False}

    # load base_directory list
    machine_list = com.get_machine_list(param["dev_directory"])
    print("=====================================")
    print("Train Machine List: ", machine_list)
    print("=====================================")

    # loop of the base directory
    for idx, machine in enumerate(machine_list):
        print("\n===========================")
        print("[{idx}/{total}] {machine}".format(machine=machine, idx=idx+1, total=len(machine_list)))
        
        root_path = param["dev_directory"] + "/" + machine

        data_list = com.select_dirs(param=param, machine=machine)
        id_list = com.get_machine_id_list(target_dir=root_path, dir_type="train")

        print("Current Machine: ", machine)
        print("Machine ID List: ", id_list)

        train_list = []
        val_list = []

        for path in data_list:
            if random.random() < 0.85:
                train_list.append(path)
            else:
                val_list.append(path)
        
        for _id in id_list:
            # generate dataset
            print("\n----------------")
            print("Generating Dataset of Current ID: ", _id)

            train_dataset = AudioDataset(data=train_list, _id=_id, root=root_path, audio_conf=audio_conf)
            val_dataset = AudioDataset(data=val_list, _id=_id, root=root_path, audio_conf=val_audio_conf)
            
            train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_dl = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

            print("------ DONE -------")
            
            model_file_path = "{model}/model_{machine}_{_id}.pt".format(model=param["model_directory"], machine=machine, _id=_id)
            history_img = "{model}/history_{machine}_{_id}.png".format(model=param["model_directory"], machine=machine, _id=_id)

            if os.path.exists(model_file_path):
                com.logger.info("model exists")
                continue

            # train model
            print("\n----------------")
            print("Start Model Training...")

            train_loss_list = []
            val_loss_list = []
 
            
            extractor = load_extractor()
            extractor = nn.DataParallel(extractor, output_device=device_1)
            extractor = extractor.to(device=device_0)
            extractor.eval()

            flow_model = BuildFlow(int(param['latent_size']), int(param['NF_layers']))
            flow_model = nn.DataParallel(flow_model, output_device=device_0)
            flow_model = flow_model.to(device=device_1)

            # set up training info
            optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-6)

            for epoch in range(1, epochs+1):
                train_loss = 0.0
                val_loss = 0.0
                print("Epoch: {}".format(epoch))

                flow_model.train()

                for batch in tqdm(train_dl):

                    optimizer.zero_grad()
                    feature = extractor(batch)
                    loss = flow_model(feature)

                    # Do backprop and optimizer step
                    if ~(torch.isnan(loss) | torch.isinf(loss)):
                        loss.backward()
                        optimizer.step()

                    train_loss += loss.item()

                    del batch
                
                train_loss /= len(train_dl)
                train_loss_list.append(train_loss)

                flow_model.eval()
                
                with torch.no_grad():
                    for batch in tqdm(val_dl):
                        
                        feature = extractor(batch)
                        loss = flow_model(feature)

                        #loss = loss_function(batch)
                        #val_loss += loss.item()
                        del batch

                val_loss /= len(val_dl)
                val_loss_list.append(val_loss)

                print("Train Loss: {train_loss}, Validation Loss: {val_loss}".format(train_loss=train_loss, val_loss=val_loss))
            
            visualizer.loss_plot(train_loss_list, val_loss_list)
            visualizer.save_figure(history_img)
            
            torch.save(flow_model.state_dict(), model_file_path)
            com.logger.info("save_model -> {}".format(model_file_path))

            del train_dataset, val_dataset, train_dl, val_dl
            
            gc.collect()

            time.sleep(5)