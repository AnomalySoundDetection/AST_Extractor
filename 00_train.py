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
import numpy as np
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
from Flow_Model import FastFlow
from sklearn import metrics
import csv
########################################################################

def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

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
    mode, machine_type = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    param = com.load_yaml()
        
    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # training device
    #device_0 = torch.device('cuda:0')
    device_1 = torch.device('cuda:1')

    # training info
    epochs = int(param["fit"]["epochs"])
    batch_size = int(param["fit"]["batch_size"])

    channel = int(param["channel"])
    latent_size = int(param["latent_size"])
    layer_num = int(param["NF_layers"])

    # 11/30/22: I decouple the dataset and the following hyper-parameters to make it easier to adapt to new datasets
    # dataset spectrogram mean and std, used to normalize the input
    # norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
    # target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128}
    # # if add noise for data augmentation, only use for speech commands
    # noise = {'audioset': False, 'esc50': False, 'speechcommands':True}

    # audio config
    audio_conf = {'num_mel_bins': 128, 'target_length': int(param['tdim']), 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'dcase', 
                    'mode': 'train', 'mean': -4.2677393, 'std': 4.5689974, 'noise': False, 'sample_rate': 16000}
    
    val_audio_conf = {'num_mel_bins': 128, 'target_length': int(param['tdim']), 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'dcase', 
                        'mode': 'test', 'mean': -4.2677393, 'std': 4.5689974, 'noise': False, 'sample_rate': 16000}

    test_audio_conf = {'num_mel_bins': 128, 'target_length': int(param['tdim']), 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'dcase', 
                    'mode': 'train', 'mean': -4.2677393, 'std': 4.5689974, 'noise': False, 'sample_rate': 16000}

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

        if machine_type != machine:
            print("Skipping machine {}".format(machine))
            continue

        # data_list = com.select_dirs(param=param, machine=machine)
        # id_list = com.get_machine_id_list(target_dir=root_path, dir_type="train")

        train_list = com.select_dirs(param=param, machine=machine)
        test_list = com.select_dirs(param=param, machine=machine, dir_type="test")

        id_list = com.get_machine_id_list(target_dir=root_path, dir_type="train")

        print("Current Machine: ", machine)
        print("Machine ID List: ", id_list)

        # train_list = []
        # val_list = []

        # for path in data_list:
        #     if random.random() < 1.1:
        #         train_list.append(path)
        #     else:
        #         val_list.append(path)
        
        for _id in id_list:
            # generate dataset
            print("\n----------------")
            print("Generating Dataset of Current ID: ", _id)

            train_dataset = AudioDataset(data=train_list, _id=_id, root=root_path, frame_length=int(param['frame_length']), 
                                         shift_length=int(param['shift_length']), audio_conf=audio_conf)
            # val_dataset = AudioDataset(data=val_list, _id=_id, root=root_path, frame_length=int(param['frame_length']), 
                                        #  shift_length=int(param['shift_length']),audio_conf=val_audio_conf)
            test_dataset = AudioDataset(data=test_list, _id=_id, root=root_path, frame_length=int(param['frame_length']), 
                                         shift_length=int(param['shift_length']), audio_conf=test_audio_conf, train=False)
            
            train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            # val_dl = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
            test_dl = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

            print("------ DONE -------")
            
            model_file_path = "{model}/model_{machine}_{_id}.pt".format(model=param["model_directory"], machine=machine, _id=_id)
            checkpoint_path = "{model}/checkpoint_{machine}_{_id}.pt".format(model=param["model_directory"], machine=machine, _id=_id)
            history_img = "{model}/history_{machine}_{_id}.png".format(model=param["model_directory"], machine=machine, _id=_id)

            if os.path.exists(model_file_path):
                com.logger.info("model exists")
                continue

            # train model
            print("\n----------------")
            print("Start Model Training...")

            train_loss_list = []
            val_loss_list = []
            
            extractor = load_extractor(int(param['tdim']))
            #extractor = nn.DataParallel(extractor, device_ids=[0, 1])
            extractor = extractor.to(device=device_1)
            extractor.eval()

            flow_model = FastFlow(flow_steps=int(param['NF_layers']))
            #flow_model = nn.DataParallel(flow_model, device_ids=[1, 0])
            flow_model = flow_model.to(device=device_1)

            # set up training info
            optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-4)
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)

            original_auc = 0
            original_pauc = 0

            file_list = test_dataset.data
            normal_num = test_dataset.normal_num
            anomaly_num = test_dataset.anomaly_num

            for epoch in range(1, epochs+1):
                try:
                    train_loss = 0.0

                    print("Epoch: {}".format(epoch))

                    flow_model.train()

                    '''
                    Train
                    '''
                    print("Running train dataset...")

                    for batch in tqdm(train_dl):

                        optimizer.zero_grad()

                        with torch.no_grad():
                            batch = batch.to(device=device_1)
                            feature = extractor(batch)
                    
                        output, log_jac_dets = flow_model(feature)
                        
                        loss = 0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets

                        loss = torch.mean(loss)
                    
                        if ~(torch.isnan(loss) | torch.isinf(loss)):
                            loss.backward()
                            optimizer.step()
                        
                        train_loss += loss.item()

                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    train_loss /= len(train_dl)
                    train_loss_list.append(train_loss)

                    print("Train Loss: {train_loss}".format(train_loss=train_loss))

                    '''
                    Test AUROC
                    '''
                    anomaly_score_list = [0. for file in file_list]
                    ground_truth_list = [0 for file in file_list]
                    weight = [0. for file in file_list]

                    flow_model.eval()

                    print("Running test dataset...")
                    print("----------------- Evaluating -------------------")
                    
                    with torch.no_grad():
                        for idx, batch_info in enumerate(tqdm(test_dl)):
                    
                            batch, ground_truth = batch_info

                            batch = batch.to(device=device_1)
                            feature = extractor(batch)

                            output, log_jac_dets = flow_model(feature)

                            log_prob = -torch.mean(output**2, dim=1) * 0.5
                    
                            log_prob = log_prob.reshape(shape=(1, -1)) 
                            sorted_log_prob, _ = torch.sort(log_prob)
                    
                            anomaly_score = -torch.mean(sorted_log_prob[:100])
                            #anomaly_score = -log_prob
                    
                            ground_truth_list[idx] = ground_truth.item()
                            anomaly_score_list[idx] = anomaly_score.item()

                            if ground_truth.item() == 1:
                                weight[idx] = anomaly_num / len(file_list)
                            else:
                                weight[idx] = normal_num / len(file_list)

                        auc = metrics.roc_auc_score(ground_truth_list, anomaly_score_list, sample_weight=weight)
                        p_auc = metrics.roc_auc_score(ground_truth_list, anomaly_score_list, max_fpr=param["max_fpr"], sample_weight=weight)

                        com.logger.info("AUC of {machine} {_id}: {auc}".format(machine=machine, _id=_id, auc=auc))
                        com.logger.info("pAUC of {machine} {_id}: {p_auc}".format(machine=machine, _id=_id, p_auc=p_auc))

                    if auc >= original_auc and p_auc >= original_pauc:
                        torch.save({
                            'model_stat_dict': flow_model.state_dict(),
                            'optimizer_stat_dict': optimizer.state_dict(),
                            'epoch': epoch, 
                            'loss': train_loss
                            }, checkpoint_path
                        )

                        original_auc = auc
                        original_pauc = p_auc

                    else:
                        checkpoint = torch.load(checkpoint_path)

                        flow_model = FastFlow(flow_steps=int(param['NF_layers']))
                        flow_model.load_state_dict(checkpoint['model_stat_dict'])
                        flow_model = flow_model.to(device=device_1)

                        optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-4)
                        optimizer.load_state_dict(checkpoint['optimizer_stat_dict'])

                        train_loss = checkpoint['loss']
                        train_loss_list[-1] = train_loss

                        print("Reloading Model from epoch {ep}".format(ep=checkpoint['epoch']))

                except:
                    print("Error during training, stop at epoch {ep}".format(ep=epoch))
                    torch.save({
                        'model_stat_dict': flow_model.state_dict()
                        }, model_file_path
                    )
                    com.logger.info("save model -> {}".format(model_file_path))
                    exit(1)
            
            visualizer.loss_plot(train_loss_list, val_loss_list)
            visualizer.save_figure(history_img)

            print("----------------- Recording Score -------------------")
            anomaly_score_list = [0. for file in file_list]
            ground_truth_list = [0 for file in file_list]
            weight = [0. for file in file_list]

            anomaly_score_csv = "{result}/anomaly_score_{machine}_{_id}.csv".format(result=param["result_directory"],
                                                                                     machine=machine,
                                                                                     _id=_id)
            anomaly_score_record = []

            with torch.no_grad():                   
                for idx, batch_info in enumerate(tqdm(test_dl)):
                
                    batch, ground_truth = batch_info

                    batch = batch.to(device=device_1)
                    feature = extractor(batch)

                    output, log_jac_dets = flow_model(feature)

                    log_prob = -torch.mean(output**2, dim=1) * 0.5

            
                    log_prob = log_prob.reshape(shape=(1, -1)) 
                    sorted_log_prob, _ = torch.sort(log_prob)
            
                    anomaly_score = -torch.mean(sorted_log_prob[:100])
                    #anomaly_score = -log_prob
            
                    ground_truth_list[idx] = ground_truth.item()
                    anomaly_score_list[idx] = anomaly_score.item()

                    if ground_truth.item() == 1:
                        weight[idx] = anomaly_num / len(file_list)
                    else:
                        weight[idx] = normal_num / len(file_list)

                    anomaly_score_record.append([os.path.basename(file_list[idx]), anomaly_score.item()])

                # append AUC and pAUC to lists
                auc = metrics.roc_auc_score(ground_truth_list, anomaly_score_list, sample_weight=weight)
                p_auc = metrics.roc_auc_score(ground_truth_list, anomaly_score_list, max_fpr=param["max_fpr"], sample_weight=weight)

                anomaly_score_record.append([_id, auc, p_auc])
                #performance.append([auc, p_auc])
                com.logger.info("AUC of {machine} {_id}: {auc}".format(machine=machine, _id=_id, auc=auc))
                com.logger.info("pAUC of {machine} {_id}: {p_auc}".format(machine=machine, _id=_id, p_auc=p_auc))

                # save anomaly score
                save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_record)
                com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))
            
            torch.save({
                'model_stat_dict': flow_model.state_dict()
                }, model_file_path
            )

            com.logger.info("save_model -> {}".format(model_file_path))

            #del train_dataset, val_dataset, train_dl, val_dl, flow_model, extractor
            del train_dataset, test_dataset, train_dl, test_dl, flow_model, extractor

            gc.collect()
            torch.cuda.empty_cache()
            

            time.sleep(5)