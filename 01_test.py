"""
 @file   01_test.py
 @brief  Script for test
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import csv
import re
import itertools
import sys
import torch
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
from Flow_Model import BuildFlow
from sklearn import metrics
import copy
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.load_yaml()
#######################################################################


########################################################################
# def
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode, _ = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(param["result_directory"], exist_ok=True)

    # load base directory
    # training device
    device = torch.device('cuda')

    # batch size
    batch_size = int(param["fit"]["batch_size"])

    channel = int(param["channel"])
    latent_size = int(param["latent_size"])
    layer_num = int(param["NF_layers"])

    # audio config
    test_audio_conf = {'num_mel_bins': 128, 'target_length': int(param['tdim']), 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'dcase', 
                    'mode': 'train', 'mean': -4.2677393, 'std': 4.5689974, 'noise': False, 'sample_rate': 16000}

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    machine_list = com.get_machine_list(param["dev_directory"])
    print("=====================================")
    print("Test Machine List: ", machine_list)
    print("=====================================")

    # loop of the base directory
    for idx, machine in enumerate(machine_list):
        print("\n===========================")
        print("[{idx}/{total}] {machine}".format(machine=machine, idx=idx+1, total=len(machine_list)))
        
        root_path = param["dev_directory"] + "/" + machine

        data_list = com.select_dirs(param=param, machine=machine, dir_type="test")
        id_list = com.get_machine_id_list(target_dir=root_path, dir_type="test")

        print("Current Machine: ", machine)
        print("Machine ID List: ", id_list)
        

        # inputDim = param["feature"]["baseline"]["n_mels"] * param["feature"]["baseline"]["frames"]
        # model = Net(inputDim=inputDim)
        # model.load_state_dict(torch.load(model_file))
        
        # device = torch.device('cuda')
        # model = model.to(device)
        #summary(model.float(), input_size=(inputDim, ))

        for _id in id_list:
            
            model_file_path = "{model}/model_{machine}_{_id}.pt".format(model=param["model_directory"], machine=machine, _id=_id)

            if not os.path.exists(model_file_path):
                com.logger.error("{machine} {_id} model not found ".format(machine=machine, _id=_id))
                continue

            print("\n----------------")
            print("Generating Dataset of Current ID: ", _id)

            test_dataset = AudioDataset(data=data_list, _id=_id, root=root_path, frame_length=int(param['frame_length']), 
                                         shift_length=int(param['shift_length']), audio_conf=test_audio_conf, train=False)
            
            test_dl = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

            file_list = test_dataset.data

            print("------ DONE -------")


            if mode:
                # results by type
                csv_lines.append([machine])
                csv_lines.append(["id", "AUC", "pAUC"])
                performance = []

            anomaly_score_csv = "{result}/anomaly_score_{machine}_{_id}.csv".format(result=param["result_directory"],
                                                                                     machine=machine,
                                                                                     _id=_id)
                
            file_list = test_dataset.data

            anomaly_score_record = []

            print("\n----------------")
            print("Load Model")

            extractor = load_extractor(int(param['tdim']))
            extractor = extractor.to(device=device)
            extractor.eval()

            flow_model = BuildFlow(flow_steps=int(param['NF_layers']))
            #flow_model = nn.DataParallel(flow_model)
            flow_model.load_state_dict(torch.load(model_file_path), strict=False)
            flow_model = flow_model.to(device=device)
            flow_model.eval()

            print("Finish Loading")

            print("----------------")
            print("Start Model Testing...")

            anomaly_score_list = [0. for file in file_list]
            ground_truth_list = [0 for file in file_list]

            with torch.no_grad():
                for idx, batch_info in enumerate(tqdm(test_dl)):
                    
                    batch, ground_truth = batch_info

                    batch = batch.to(device)
                    feature = extractor(batch)
                    #feature = torch.reshape(feature, (-1, channel, 4, 4))
                    #feature = feature.unsqueeze(2)
                    #feature = feature.unsqueeze(3)

                    output, log_jac_dets = flow_model(feature)

                    mean_output = torch.mean(output, dim=1)
                    #print("Shape of mean output is {shape}".format(shape=mean_output.shape))
                    
                    log_prob = -torch.sum(mean_output**2, dim=(1, 2), keepdim=True) * 0.5 + log_jac_dets
                    anomaly_score = torch.mean(-log_prob)
                    
                    ground_truth_list[idx] = ground_truth.item()
                    anomaly_score_list[idx] = anomaly_score.item()

                    anomaly_score_record.append([os.path.basename(file_list[idx]), anomaly_score.item()])

                if mode:
                    # append AUC and pAUC to lists
                    auc = metrics.roc_auc_score(ground_truth_list, anomaly_score_list)
                    p_auc = metrics.roc_auc_score(ground_truth_list, anomaly_score_list, max_fpr=param["max_fpr"])

                    anomaly_score_record.append([_id, auc, p_auc])
                    #performance.append([auc, p_auc])
                    com.logger.info("AUC of {machine} {_id}: {auc}".format(machine=machine, _id=_id, auc=auc))
                    com.logger.info("pAUC of {machine} {_id}: {p_auc}".format(machine=machine, _id=_id, p_auc=p_auc))

                    # save anomaly score
                    save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_record)
                    com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

            #if mode:
                # calculate averages for AUCs and pAUCs
                #averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
                #csv_lines.append(["Average"] + list(averaged_performance))
                #csv_lines.append([])

    #if mode:
        # output results
        #result_path = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
        #com.logger.info("AUC and pAUC results -> {}".format(result_path))
        #save_csv(save_file_path=result_path, save_data=csv_lines)
