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

    # Model parameters
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

    # Optimizer parameters
    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_t', default=10, type=int)
    parser.add_argument('--lr_scheduler', default='CosineAnnealingLR', type=str)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--loss_function', default='L1Loss', type=str)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--label_smoothing', default=0.3, type=float)

    # Training parameters
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--n_fold', default=5, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--text', default='default', type=str)
    parser.add_argument('--device', default='0,1,2,3', type=str)

    return parser


def main(args):

    seed = 10
    suffix = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%y%m%d_%H%M")

    config = {
        # Model parameters
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
        
        # Optimizer parameters
        'optimizer': args.optimizer,
        'lr': args.lr,
        'lr_t': args.lr_t,
        'lr_scheduler': args.lr_scheduler,
        'gamma': args.gamma,
        'loss_function': args.loss_function,
        'patience': args.patience,
        'weight_decay': args.weight_decay,
        'label_smoothing': args.label_smoothing,
        
        # Training parameters
        'epochs': args.epochs,
        'n_fold': args.n_fold,
        'num_workers': args.num_workers,
        'text': args.text,
        'device': args.device
        }
    
    model_save_name='./RESULTS/'+config['text']+"_"+suffix+"("+ str(config['model'])+"_"+\
                                                                str(config['batch_size'])+"_"+\
                                                                str(config['pretrain'])+"__"+\
                                                                str(config['max_len'])+"_"+\
                                                                str(config['embedding_dim'])+"_"+\
                                                                str(config['num_classes'])+"_"+\
                                                                str(config['num_lstm_layers'])+"_"+\
                                                                str(config['conv1_nf'])+"_"+\
                                                                str(config['conv2_nf'])+"_"+\
                                                                str(config['conv3_nf'])+"_"+\
                                                                str(config['lstm_drop_p'])+"_"+\
                                                                str(config['conv_drop_p'])+"_"+\
                                                                str(config['fc_drop_p'])+"_"+\
                                                                str(config['optimizer'])+"_"+\
                                                                str(config['lr'])+"_"+\
                                                                str(config['lr_t'])+"_"+\
                                                                str(config['lr_scheduler'])+"_"+\
                                                                str(config['gamma'])+"_"+\
                                                                str(config['loss_function'])+"_"+\
                                                                str(config['patience'])+"_"+\
                                                                str(config['weight_decay'])+"_"+\
                                                                str(config['label_smoothing'])+")_fold_"
                                                            
    config['model_save_name'] = model_save_name
    print(config['model_save_name']+ ' is start!')
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
    train_raw_df = get_train_data('/data/KIST_PLANT/train')
    train_sum_df = pd.read_csv('TRAIN_foreground_sum_df.csv')

    train_mask_img = glob(os.path.join('/home/SY_LEE/KIST_PLANT/DATA_BACKGROUND_TRAIN/', '*.png'))
    train_mask_img.sort()
    
    train_mask_img = pd.DataFrame(train_mask_img, columns=['mask_img'])
    train_df = pd.concat([train_raw_df, train_mask_img, train_sum_df], axis=1)

    # Cross Validation
    kfold = KFold(n_splits=config['n_fold'],shuffle=True,random_state=seed)
    n_fold = config['n_fold']
    k_valid_f1, k_valid_nmae = [], []

    wandb_name = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%y%m%d_%H%M%S")

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_df)):
        wandb.init(project='ANOMALY_DETECTION', name=wandb_name, entity="sylee1996", config=config, reinit=True)
        
        Train_set = [train_df.iloc[i] for i in train_idx]
        Valid_set = [train_df.iloc[i] for i in valid_idx]
    
        # Train
        Train_dataset = Custom_dataset(Train_set, max_len=config['max_len'], mode='train')
        Train_loader = DataLoader(Train_dataset, batch_size=config['batch_size'], pin_memory=True,
                                num_workers=config['num_workers'], prefetch_factor=config['batch_size']*2, 
                                shuffle=True)

        # Valid
        Valid_dataset = Custom_dataset(Valid_set, max_len=config['max_len'], mode='valid')
        Valid_loader = DataLoader(Valid_dataset, batch_size=config['batch_size'], pin_memory=True,
                                num_workers=config['num_workers'], prefetch_factor=config['batch_size']*2, 
                                shuffle=True)
        config['num_features'] = Train_dataset[0]['csv_feature'].shape[0]
        config['in_channels'] = Train_dataset[0]['img'].shape[0]
        
        model = CNN2RNN(config).to(config['device'])
        # model = CNNRegressor(config).to(config['device'])
        model = nn.DataParallel(model).to(config['device'])

        if config['lr_scheduler'] == 'CosineAnnealingLR':
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['lr_t'], eta_min=0)
            
        elif config['lr_scheduler'] == 'CosineAnnealingWarmUpRestarts':
            optimizer = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=config['weight_decay'])
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=config['lr_t'], eta_max=config['lr'], gamma=config['gamma'], T_mult=1, T_up=0)
        
        if config['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
                
        if config['loss_function'] == 'Focal_with_Lb':
            criterion = FocalLossWithSmoothing(num_classes=config['class_num'],
                                                lb_smooth=config['label_smoothing'],
                                                gamma=2).to(config['device'])
            
        elif config['loss_function'] == 'CE_with_Lb':
            criterion = SmoothCrossEntropyLoss(smoothing=config['label_smoothing']).to(config['device'])
            
        elif config['loss_function'] == 'Focal':
            criterion = FocalLoss(alpha=0.25, gamma=2)
            
        elif config['loss_function'] == 'L1Loss':
            criterion = nn.L1Loss()
            
        scaler = torch.cuda.amp.GradScaler() 
        early_stopping_loss = EarlyStopping(patience=config['patience'], mode='min')
        early_stopping_nmae = EarlyStopping(patience=config['patience'], mode='min')
        
        wandb.watch(model)
        best_loss=100
        best_nmae=0.5
        each_fold_train_loss, train_nmae_list = [], []
        each_fold_valid_loss, valid_nmae_list = [], []

        epochs = config['epochs']
        
        for epoch in range(epochs):
            train_loss, train_nmae = 0, 0
            valid_loss, valid_nmae = 0, 0

            model.train()
            for batch_id, batch in tqdm(enumerate(Train_loader), total=len(Train_loader)):
                
                optimizer.zero_grad()
                train_img = torch.tensor(batch['img'], dtype=torch.float32).to(config['device'])
                train_csv_feature = torch.tensor(batch['csv_feature'], dtype=torch.float32).to(config['device'])
                train_label = torch.tensor(batch['label'], dtype=torch.long).to(config['device'])

                with torch.cuda.amp.autocast():
                    pred = model(train_img, train_csv_feature)
                    # pred = model(train_img)
                loss = criterion(pred.squeeze(1), train_label)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_nmae += loss.item() / np.mean(train_label.tolist())
                
            train_loss = train_loss/len(Train_loader)
            train_nmae = train_nmae/len(Train_loader)
            each_fold_train_loss.append(train_loss)
            train_nmae_list.append(train_nmae)
            scheduler.step()

            model.eval()
            for batch_id, val_batch in tqdm(enumerate(Valid_loader), total=len(Valid_loader)):
                with torch.no_grad():
                    valid_img = torch.tensor(val_batch['img'], dtype=torch.float32).to(config['device'])
                    valid_csv_feature = torch.tensor(val_batch['csv_feature'], dtype=torch.float32).to(config['device'])
                    valid_label = torch.tensor(val_batch['label'], dtype=torch.long).to(config['device'])

                    val_pred = model(valid_img, valid_csv_feature)
                    # val_pred = model(valid_img)
                    val_loss = criterion(val_pred.squeeze(1), valid_label)
                    
                valid_loss += val_loss.item()
                valid_nmae += loss.item() / np.mean(valid_label.tolist())
            
            valid_loss = valid_loss/len(Valid_loader)
            valid_nmae = valid_nmae/len(Valid_loader)
            each_fold_valid_loss.append(valid_loss)
            valid_nmae_list.append(valid_nmae)
            
            print_best = 0    
            # if (each_fold_valid_loss[-1] <= best_loss) or (valid_nmae_list[-1] <= best_nmae):
            if each_fold_valid_loss[-1] <= best_loss:
                
                difference = best_loss - each_fold_valid_loss[-1] 
                if (valid_nmae_list[-1] <= best_nmae):
                    best_nmae = valid_nmae_list[-1] 
                if (each_fold_valid_loss[-1] <= best_loss):
                    best_loss = each_fold_valid_loss[-1]
                
                pprint_best = valid_nmae_list[-1]
                pprint_best_loss = each_fold_valid_loss[-1]
                
                best_idx = epoch+1
                model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict()
                best_model_wts = copy.deepcopy(model_state_dict)
                
                # load and save best model weights
                model.module.load_state_dict(best_model_wts)
                torch.save(best_model_wts, config['model_save_name'] + str(fold+1) + ".pt")
                # print_best = '==> best model saved - %d epoch / %.5f    difference %.5f'%(best_idx, best, difference)
                print_best = '==> best model saved %d epoch / NMAE: %.5f  loss: %.5f  /  difference %.5f'%(best_idx, pprint_best, pprint_best_loss, difference)

            print(f'Fold : {fold+1}/{n_fold}    epoch : {epoch+1}/{epochs}')
            print(f'TRAIN_Loss : {train_loss:.5f}    TRAIN_NMAE : {train_nmae:.5f}')
            print(f'VALID_Loss : {valid_loss:.5f}    VALID_NMAE : {valid_nmae:.5f}    BEST_NMAE : {pprint_best:.5f}    BEST_LOSS : {pprint_best_loss:.5f}')
            print('\n') if type(print_best)==int else print(print_best,'\n')

            wandb.log({
                    "Epoch": epoch+1,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "best_loss": best_loss,
                    "best_nmae": best_nmae,
                })
            # if early_stopping_nmae.step(torch.tensor(valid_nmae_list[-1])) and early_stopping_loss.step(torch.tensor(each_fold_valid_loss[-1])):
            if early_stopping_loss.step(torch.tensor(each_fold_valid_loss[-1])):
                wandb.join()  
                break
            
        # max_index = each_fold_valid_loss.index(min(each_fold_valid_loss))
        wandb.join()   
        # print("VALID Loss: ", each_fold_valid_loss[max_index], ", VALID NMAE: ", valid_nmae_list[max_index])
        print("VALID Loss: ", pprint_best_loss, ", VALID NMAE: ", pprint_best)
            
        # k_valid_f1.append(each_fold_valid_loss[max_index])
        # k_valid_nmae.append(valid_nmae_list[max_index])
        k_valid_nmae.append(pprint_best)
        k_valid_f1.append(pprint_best_loss)
        
    
    print(config['model_save_name'] + ' model is saved!')
    
    print("1Fold - VALID NMAE: ", k_valid_nmae[0], ", 1Fold - VALID Loss: ", k_valid_f1[0])
    print("2Fold - VALID NMAE: ", k_valid_nmae[1], ", 2Fold - VALID Loss: ", k_valid_f1[1])
    print("3Fold - VALID NMAE: ", k_valid_nmae[2], ", 3Fold - VALID Loss: ", k_valid_f1[2])
    print("4Fold - VALID NMAE: ", k_valid_nmae[3], ", 4Fold - VALID Loss: ", k_valid_f1[3])
    print("5Fold - VALID NMAE: ", k_valid_nmae[4], ", 5Fold - VALID Loss: ", k_valid_f1[4])
    
    
    print("k-fold Valid NMAE: ",np.mean(k_valid_nmae),", k-fold Valid Loss: ",np.mean(k_valid_f1))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)


