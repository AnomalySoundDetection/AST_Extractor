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
# Train
########################################################################
def train_one_epoch(extractor, flow_model, optimizer, train_dl):

    train_loss = 0.0
    
    extractor.eval()
    flow_model.train()

    # print("########## AST MODEL ##########")
    # print(extractor)
    # print("###############################")

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

    print("Train Loss: {train_loss}".format(train_loss=train_loss))

def test_one_epoch(extractor, flow_model, val_dl, file_list, fpr):
    
    extractor.eval()
    flow_model.eval()

    # print("file list length: ", len(file_list))
    anomaly_score_list = [0. for file in file_list]
    ground_truth_list = [0 for file in file_list]
    # weight = [0. for file in file_list]

    with torch.no_grad():
        for idx, batch_info in enumerate(tqdm(val_dl)):
    
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

            # if ground_truth.item() == 1:
            #     weight[idx] = anomaly_num / len(file_list)
            # else:
            #     weight[idx] = normal_num / len(file_list)

    auc = metrics.roc_auc_score(ground_truth_list, anomaly_score_list)
    p_auc = metrics.roc_auc_score(ground_truth_list, anomaly_score_list, max_fpr=fpr)

    return auc, p_auc

def record_csv(extractor, flow_model, test_dl, file_list, anomaly_score_csv, anomaly_num, normal_num, fpr):

    extractor.eval()
    flow_model.eval()

    anomaly_score_record = []

    anomaly_score_list = [0. for file in file_list]
    ground_truth_list = [0 for file in file_list]
    weight = [0. for file in file_list]

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


    auc = metrics.roc_auc_score(ground_truth_list, anomaly_score_list, sample_weight=weight)
    p_auc = metrics.roc_auc_score(ground_truth_list, anomaly_score_list, max_fpr=fpr, sample_weight=weight)

    anomaly_score_record.append([_id, auc, p_auc])

    save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_record)

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
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["checkpoint_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)

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
    audio_conf = {'num_mel_bins': 128, 'target_length': int(param['tdim']), 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'dcase', 
                    'mode': 'train', 'mean': -4.2677393, 'std': 4.5689974, 'noise': False, 'sample_rate': 16000}
    
    val_audio_conf = {'num_mel_bins': 128, 'target_length': int(param['tdim']), 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'dcase', 
                    'mode': 'train', 'mean': -4.2677393, 'std': 4.5689974, 'noise': False, 'sample_rate': 16000}

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

        
        for _id in id_list:
            
            train_list, val_list, test_list = [], [], []

            tmp_train_list = [sample for sample in machine_train_list if _id in sample]

            for sample in tmp_train_list:
                if random.random() < 0.9:
                    train_list.append(sample)
                else:
                    val_list.append(sample)

            tmp_train_list = [sample for sample in machine_train_list if _id not in sample]
            random.shuffle(tmp_train_list)
            random.shuffle(tmp_train_list)
            random.shuffle(tmp_train_list)

            val_list += tmp_train_list[:len(val_list)]

            test_list = [sample for sample in machine_test_list if _id in sample]

            # generate dataset
            print("----------------")
            print("Generating Dataset of Current ID: ", _id)

            train_dataset = AudioDataset(data=train_list, _id=_id, root=root_path, frame_length=int(param['frame_length']), 
                                         shift_length=int(param['shift_length']), audio_conf=audio_conf)

            val_dataset = AudioDataset(data=val_list, _id=_id, root=root_path, frame_length=int(param['frame_length']), 
                                         shift_length=int(param['shift_length']), audio_conf=val_audio_conf, train=False)

            test_dataset = AudioDataset(data=test_list, _id=_id, root=root_path, frame_length=int(param['frame_length']), 
                                         shift_length=int(param['shift_length']), audio_conf=test_audio_conf, train=False)
            
            train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

            val_dl = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

            test_dl = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

            
            model_file_path = "{model}/model_{machine}_{_id}.pt".format(model=param["model_directory"], machine=machine, _id=_id)
            checkpoint_path = "{checkpoint}/checkpoint_{machine}_{_id}.pt".format(checkpoint=param["checkpoint_directory"], machine=machine, _id=_id)

            if mode:
                if os.path.exists(model_file_path):
                    com.logger.info("model exists")
                    continue
                
                # train model
                print("------------ Model Loading ------------\n")

                #train_loss_list = []
                
                extractor = ASTModel(input_tdim=int(param['tdim']), audioset_pretrain=True)
                extractor = extractor.to(device=device_1)

                flow_model = FastFlow(flow_steps=int(param['NF_layers']))
                flow_model = flow_model.to(device=device_1)
                flow_model = flow_model.float()

                # set up training info
                optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-4)
                #scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)

                original_auc, original_pauc = 0, 0

                for epoch in range(1, epochs+1):                  
                    print("Epoch: {}".format(epoch))
                    
                    print("----------------- Training -------------------")
                    train_one_epoch(extractor=extractor, flow_model=flow_model, optimizer=optimizer, train_dl=train_dl)

                    if epoch % 5 == 0:    
                        print("----------------- Evaluating -------------------")
                        auc, p_auc = test_one_epoch(extractor=extractor, flow_model=flow_model, val_dl=val_dl, file_list=val_list, fpr=param['max_fpr'])
                            
                        com.logger.info("AUC of {machine} {_id}: {auc}".format(machine=machine, _id=_id, auc=auc))
                        com.logger.info("pAUC of {machine} {_id}: {p_auc}".format(machine=machine, _id=_id, p_auc=p_auc))

                        # auc, p_auc = test_one_epoch(extractor=extractor, flow_model=flow_model, val_dl=test_dl, file_list=test_list, fpr=param['max_fpr'])
                            
                        # com.logger.info("AUC of {machine} {_id}: {auc}".format(machine=machine, _id=_id, auc=auc))
                        # com.logger.info("pAUC of {machine} {_id}: {p_auc}".format(machine=machine, _id=_id, p_auc=p_auc))

                        if auc >= original_auc and p_auc >= original_pauc:
                        # if p_auc >= original_pauc:
                            torch.save({
                                'ast_state_dict': extractor.state_dict(),
                                'flow_state_dict': flow_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch
                                }, checkpoint_path
                            )

                            original_auc, original_pauc = auc, p_auc
                            print()
                        else:
                            checkpoint = torch.load(checkpoint_path)

                            extractor = ASTModel(input_tdim=int(param['tdim']), audioset_pretrain=True)
                            extractor.load_state_dict(checkpoint['ast_state_dict'])
                            extractor = extractor.to(device=device_1)

                            flow_model = FastFlow(flow_steps=int(param['NF_layers']))
                            flow_model.load_state_dict(checkpoint['flow_state_dict'])
                            flow_model = flow_model.to(device=device_1)

                            optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-4)
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                            print("Reloading Model from epoch {ep}\n".format(ep=checkpoint['epoch']))

                torch.save({
                    'ast_state_dict': extractor.state_dict(),
                    'flow_state_dict': flow_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epochs
                    }, model_file_path
                )

                com.logger.info("save_model -> {}".format(model_file_path))

                del train_dataset, val_dataset, train_dl, val_dl, flow_model, extractor

                gc.collect()
                torch.cuda.empty_cache()


            print("---------------- Recording Score ------------------")

            if not os.path.exists(model_file_path):
                com.logger.error("{machine} {_id} model not found ".format(machine=machine, _id=_id))
                continue

            anomaly_score_csv = "{result}/anomaly_score_{machine}_{_id}.csv".format(result=param["result_directory"],
                                                                                     machine=machine,
                                                                                     _id=_id)

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

            auc, p_auc = record_csv(extractor=test_extractor, flow_model=test_flow_model, test_dl=test_dl, file_list=test_list, anomaly_score_csv=anomaly_score_csv, 
                                    anomaly_num=test_dataset.anomaly_num, normal_num=test_dataset.normal_num, fpr=param['max_fpr'])
                
            com.logger.info("Final AUC of {machine} {_id}: {auc}".format(machine=machine, _id=_id, auc=auc))
            com.logger.info("Final pAUC of {machine} {_id}: {p_auc}".format(machine=machine, _id=_id, p_auc=p_auc)) 
            com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

            del test_dataset, test_dl, test_flow_model, test_extractor
            # del extractor, flow_model, test_extractor, test_model, train_dataset, train_dl, \
            #     val_dataset, val_dl, test_dataset, test_dl

            gc.collect()
            torch.cuda.empty_cache()

            time.sleep(5)