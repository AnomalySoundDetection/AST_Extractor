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
from AST_Model import load_extractor, ASTModel
from Dataset import AudioDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Flow_Model import FastFlow
from sklearn import metrics
import csv
import torch.nn.functional as F
########################################################################

########################################################################
# Save CSV file
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

########################################################################
# Plot
########################################################################
import matplotlib.pyplot as plt

def plot_anomaly_map(anomaly_map, img_path):

    #anomaly_map = anomaly_map.transpose(1, 2)
    anomaly_map = F.interpolate(anomaly_map.unsqueeze(0), size=(256, 512), mode='bilinear', align_corners=False)

    anomaly_map = anomaly_map.squeeze(0).permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(15, 10))
    img = ax.imshow(anomaly_map, vmin=0, vmax=2)
    cbar = ax.figure.colorbar(img, ax=ax)

    #print(resized_tensor.shape)
    #plt.imshow(anomaly_map)
    #plt.colorbar()
    plt.savefig(img_path) 
    plt.close()

    del anomaly_map

def plot_stat(anomaly_map, img_path):
    
    anomaly_map = anomaly_map.reshape(shape=(-1, 1))
    anomaly_map_stat = anomaly_map.cpu().numpy()
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.hist(anomaly_map_stat, bins=256, range=(0, 2))
    plt.savefig(img_path)
    plt.close()

    del anomaly_map

########################################################################
# Visualize
########################################################################
def visualize_one_epoch(extractor, flow_model, test_dl, file_list, fpr, folder):
    
    extractor.eval()
    flow_model.eval()

    # print("file list length: ", len(file_list))
    anomaly_score_list = [0. for file in file_list]
    ground_truth_list = [0 for file in file_list]
    # weight = [0. for file in file_list]

    with torch.no_grad():
        for idx, batch_info in enumerate(tqdm(test_dl)):
    
            batch, ground_truth = batch_info

            batch = batch.to(device=device_1)
            feature = extractor(batch)

            output, log_jac_dets = flow_model(feature)

            log_prob = -torch.mean(output**2, dim=1) * 0.5
            
            plot_log_prob = -log_prob
            #plot_log_prob = plot_log_prob.reshape(shape=(plot_log_prob.shape[1], plot_log_prob.shape[2]))
            #plot_log_prob = plot_log_prob.transpose(1, 0)
            #print(plot_log_prob.shape)

            img_path = folder + '/' + file_list[idx].split("/")[-1][:-4] + '.png'
            # print("img_path: {img_path}".format(img_path=img_path))
            
            #plot_log_prob = plot_log_prob.numpy()
            #plot_anomaly_map(anomaly_map=plot_log_prob, img_path=img_path)
            plot_stat(anomaly_map=plot_log_prob, img_path=img_path)

            log_prob = log_prob.reshape(shape=(1, -1)) 
            sorted_log_prob, _ = torch.sort(log_prob)
    
            anomaly_score = -torch.mean(sorted_log_prob[:100])
            #anomaly_score = -log_prob
    
            ground_truth_list[idx] = ground_truth.item()
            anomaly_score_list[idx] = anomaly_score.item()

            # if ground_truth.item() == 1:
            #     weight[idx] = anomaly_num / len(file_list)
            # else:
            #     weight[idx] = normal_num / len(file_list)

    auc = metrics.roc_auc_score(ground_truth_list, anomaly_score_list)
    p_auc = metrics.roc_auc_score(ground_truth_list, anomaly_score_list, max_fpr=fpr)

    return auc, p_auc


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
    #visual_folder = "./visual"
    visual_folder = "./visual_stat"

    os.makedirs(visual_folder, exist_ok=True)

    # training device
    device_1 = torch.device('cuda:0')

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
    test_audio_conf = {'num_mel_bins': 128, 'target_length': int(param['tdim']), 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'dcase', 
                    'mode': 'test', 'mean': -4.2677393, 'std': 4.5689974, 'noise': False, 'sample_rate': 16000}

    # load base_directory list
    machine_list = com.get_machine_list(param["dev_directory"])
    print("=====================================")
    print("Machine List: ", machine_list)
    print("=====================================")

    # loop of the base directory
    for idx, machine in enumerate(machine_list):
        print("\n===========================")
        print("[{idx}/{total}] {machine}".format(machine=machine, idx=idx+1, total=len(machine_list)))
        
        root_path = param["dev_directory"] + "/" + machine

        if machine_type != machine:
            print("Skipping machine {}".format(machine))
            continue
        
        machine_train_list = com.select_dirs(param=param, machine=machine)
        machine_test_list = com.select_dirs(param=param, machine=machine, dir_type="test")

        id_list = com.get_machine_id_list(target_dir=root_path)

        print("Current Machine: ", machine)
        print("Machine ID List: ", id_list)

        visual_machine_folder = visual_folder + '/' + machine
        os.makedirs(visual_machine_folder, exist_ok=True)
        
        for _id in id_list:

            test_list = [sample for sample in machine_test_list if _id in sample]
            visual_machine_id_folder = visual_machine_folder + '/' + _id
            os.makedirs(visual_machine_id_folder, exist_ok=True)

            # generate dataset
            print("----------------")
            print("Generating Dataset of Current ID: ", _id)

            test_dataset = AudioDataset(data=test_list, _id=_id, root=root_path, frame_length=int(param['frame_length']), 
                                         shift_length=int(param['shift_length']), audio_conf=test_audio_conf, train=False)

            test_dl = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

            
            model_file_path = "{model}/model_{machine}_{_id}.pt".format(model="./model", machine=machine, _id=_id)

            if not os.path.exists(model_file_path):
                com.logger.error("{machine} {_id} model not found ".format(machine=machine, _id=_id))
                continue

            test_model = torch.load(model_file_path)

            test_extractor = ASTModel(input_tdim=int(param['tdim']), audioset_pretrain=True)
            test_extractor.load_state_dict(test_model['ast_state_dict'])
            test_extractor = test_extractor.to(device=device_1)
            test_extractor = test_extractor.float()
            test_extractor.eval()

            test_flow_model = FastFlow(flow_steps=int(param['NF_layers']))
            test_flow_model.load_state_dict(test_model['flow_state_dict'])
            test_flow_model = test_flow_model.to(device=device_1)
            test_flow_model = test_flow_model.float()
            test_flow_model.eval()

            visualize_one_epoch(extractor=test_extractor, flow_model=test_flow_model, test_dl=test_dl, \
                                    file_list=test_list, fpr=param['max_fpr'], folder=visual_machine_id_folder)

            del test_dataset, test_dl, test_flow_model, test_extractor
            # del extractor, flow_model, test_extractor, test_model, train_dataset, train_dl, \
            #     val_dataset, test_dl, test_dataset, test_dl

            gc.collect()
            torch.cuda.empty_cache()

            time.sleep(5)