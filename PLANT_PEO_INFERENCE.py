import os 
import cv2
import copy
import wandb
import random
import argparse
import datetime
import warnings
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from KIST_PLANT_MODEL import CNN2RNN
from KIST_PLANT_UTILS import str2bool, img_load, Custom_dataset, CosineAnnealingWarmUpRestarts, EarlyStopping, SmoothCrossEntropyLoss, FocalLossWithSmoothing, FocalLoss, get_train_data, get_test_data

warnings.filterwarnings(action='ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Training', add_help=False)

    # Inference parameters
    parser.add_argument('--model_save_name', default='default', type=str)
    parser.add_argument('--model', default='efficientnet_b3', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--pretrain', default=True, type=str2bool)
    parser.add_argument('--max_len', default=1440, type=int)
    parser.add_argument('--embedding_dim', default=512, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--num_lstm_layers', default=1, type=int)
    parser.add_argument('--conv1_nf', default=128, type=int)
    parser.add_argument('--conv2_nf', default=128, type=int)
    parser.add_argument('--conv3_nf', default=128, type=int)
    parser.add_argument('--lstm_drop_p', default=0.3, type=float)
    parser.add_argument('--conv_drop_p', default=0.3, type=float)
    parser.add_argument('--fc_drop_p', default=0.5, type=float)
    parser.add_argument('--n_fold', default=5, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--device', default='0,1,2,3', type=str)

    return parser


def main(args):

    seed = 10   
    config = {
        # Inference parameters
        'model_save_name': args.model_save_name,
        'model': args.model,
        'batch_size': args.batch_size,
        'pretrain': args.pretrain,
        'max_len': args.max_len,
        'embedding_dim': args.embedding_dim,
        'num_classes': args.num_classes,
        'num_lstm_layers': args.num_lstm_layers,
        'conv1_nf': args.conv1_nf,
        'conv2_nf': args.conv2_nf,
        'conv3_nf': args.conv3_nf,
        'lstm_drop_p': args.lstm_drop_p,
        'conv_drop_p': args.conv_drop_p,
        'fc_drop_p': args.fc_drop_p,
        'n_fold': args.n_fold,
        'num_workers': args.num_workers,
        'device': args.device,
        }
    
    # -------------------------------------------------------------------------------------------
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print('Device: %s' % device)
    if (device.type == 'cuda') or (torch.cuda.device_count() > 1):
        print('GPU activate --> Count of using GPUs: %s' % torch.cuda.device_count())
    config['device'] = device
    
    # -------------------------------------------------------------------------------------------
    
    # Dataload
    test_raw_df = get_test_data('/data/KIST_PLANT/test')
    test_sum_df = pd.read_csv('TEST_foreground_sum_df.csv')
    
    test_mask_img = glob(os.path.join('/home/SY_LEE/KIST_PLANT/DATA_BACKGROUND_TEST/', '*.png'))
    test_mask_img.sort()
    
    test_mask_img = pd.DataFrame(test_mask_img, columns=['mask_img'])
    test_df = pd.concat([test_raw_df, test_mask_img, test_sum_df], axis=1)
    
    
    # Test
    Test_dataset = Custom_dataset(test_df, max_len=config['max_len'], mode='test')
    Test_loader = DataLoader(Test_dataset, batch_size=config['batch_size'], pin_memory=True,
                            num_workers=config['num_workers'], prefetch_factor=config['batch_size']*2, 
                            shuffle=False)
    
    config['num_features'] = Test_dataset[0]['csv_feature'].shape[0]
    config['in_channels'] = Test_dataset[0]['img'].shape[0]
    
    models = []
    
    for fold in range(config['n_fold']):
        model_dict = torch.load('./RESULTS/'+config['model_save_name'] + str(fold+1) + ".pt")
        
        model = CNN2RNN(config).to(config['device'])
        model = nn.DataParallel(model).to(config['device'])
        model.module.load_state_dict(model_dict) if torch.cuda.device_count() > 1 else model.load_state_dict(model_dict)
        
        models.append(model)

    results = []
    for batch_id, batch in tqdm(enumerate(Test_loader), total=len(Test_loader)):
        test_img = torch.tensor(batch['img'], dtype=torch.float32).to(config['device'])
        test_csv_feature = torch.tensor(batch['csv_feature'], dtype=torch.float32).to(config['device'])
                
        for fold, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if fold == 0:
                        output = model(test_img, test_csv_feature)
                    else:
                        output = output+model(test_img, test_csv_feature)

        output = output / config['n_fold']
        output = output.detach().cpu().numpy()
        results.extend(output)
    
    submission = pd.read_csv("/data/KIST_PLANT/sample_submission.csv")
    submission["leaf_weight"] = pd.DataFrame(results)
    
    submission.to_csv("./RESULTS/{}.csv".format(config['model_save_name']), index=False)
    print(config['model_save_name'] + ".csv is saved!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
