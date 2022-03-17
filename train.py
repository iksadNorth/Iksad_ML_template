# %%
# pytorch 모듈들.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.transforms import ToTensor

# 기타 모듈들.
import argparse
import wandb

wandb.init(project="complemented_Image_Classification_Pjt", entity="iksadnorth")

# 내가 만든 모듈들.
from dataloader.dataloader import BaseLoader
from trainer import Trainer
from config.container import Container

from utill.saveNload import *
from utill.ConfusionMatrix import Analsis
from utill.PathTree import PathTree
import utill.split as repo_split
from utill.TimeCheck import TimeCheck

# Remote_repo.
from torchvision import transforms as remote_vision_transforms
from torchvision import datasets as remote_vision_dataset
from torchvision import models as remote_vision_model
from torch import optim as remote_optimizer
from torch.optim import lr_scheduler as remote_scheduler
from torch import nn as remote_loss
from torchmetrics import functional as remote_metrics

# Local_repo.
import dataloader.transformer as local_transformer
import dataloader.dataset as local_dataset
import dataloader.init_weight as local_init
import model.model as local_model
import model.optimizer as local_optimizer
# import model.scheduler as local_scheduler
import model.loss as local_loss


# %%
def train(args):
    # 실행 시간 분석
    tc = TimeCheck(False)
    args.tc = tc
    
    tc.mark('설정 준비') 
    # 프로젝트 결과 보관 디렉토리 구조 정의
    archive = PathTree(args.dir_tree)
    archive.project_name = [args.project_name]
    archive.SubFolder = args.SubFolder
    archive.mktree(args.dir_saved)
    
    args.dir_log = f"{args.dir_saved}/{args.project_name}"
    
    # transform 설정.
    args.transform = args.get_obj_with_param('Transform', local_transformer, remote_vision_transforms)
    
    # dataset과 dataloader로 데이터 로드시킴.
    args.dataset = args.get_obj_with_param('DataSet', local_dataset, remote_vision_dataset,transform=args.transform)
    
    # model을 로드하고 가능하다면 GPU 이용할 수 있게 만들기
    if args['Net']['NetSaved']:
        args.net = load(args.net['NetSaved'])
    else:
        args.net = args.get_obj_with_param('Net', local_model, remote_vision_model)
    print("can use", torch.cuda.device_count(), "GPUs")
    args.net = nn.DataParallel(args.net)
    
    # model 파라미터 초기화
    args.init_weight = args.get_obj(args['InitWeight'], local_init)
    args.net.apply(args.init_weight)
    
    # optimizer 설정.
    trainable_params = filter(lambda p: p.requires_grad, args.net.parameters())
    args.optimizer = args.get_obj_with_param('Optimizer', local_optimizer, remote_optimizer, params=trainable_params)
    
    # scheduler 설정.
    args.scheduler = args.get_obj_with_param('Scheduler', None, remote_scheduler, optimizer=args.optimizer)
    
    # loss 설정.
    args.criterion = args.get_obj(args['Criterion'], local_loss, remote_loss)()
    
    # matrics 설정.
    args.main_matric = args.get_obj(args['MainMatric'], None, remote_metrics)
    args.matrics = [args.get_obj(v, None, remote_metrics) for v in args['Matrics']]
    tc.print('설정 완료')
    
    # Config.json 생성
    tc.mark('Config.json 생성')
    path_config = archive.find(args.dir_log, "Config")
    args.record(f"{path_config}/config.json")
    
    break_flag = False
    break_reason = '????'
    # 모델 학습
    for epoch in range(args.epoch):
        for train_dataset, valid_dataset in args.get_obj_with_param('split', repo_split, dataset=args.dataset):
            tc.mark('DataLoader 설정') 
            train_loader = BaseLoader(Container.inherit(args['train']), dataset=train_dataset)
            valid_loader = BaseLoader(Container.inherit(args['valid']), dataset=valid_dataset)
            
            tc.mark('훈련 시작')
            train_preds, train_answers = Trainer.train(args, train_loader)
            
            tc.mark('평가 시작')
            val_preds, val_answers, val_data = Trainer.infer(args, valid_loader)

            # 모델 평가. ############################################################
            tc.mark('Score 계산')
            train_score_name, train_score = args.main_matric.__name__, args.main_matric(train_preds, train_answers)
            val_score_name, val_score = args.main_matric.__name__, args.main_matric(val_preds, val_answers)
            scores = {matric.__name__ : matric(val_preds, val_answers) for matric in args.matrics}
            #########################################################################
            
            # 모델 로깅. ############################################################
            tc.mark('모델 로깅 설정 준비')
            dict_log = {
                f'train_{train_score_name}' : train_score,
                f'valid_{val_score_name}' : val_score,
            }
            dict_log.update({f'valid_{k}' : v for k, v in scores.items()})
            
            # to console
            tc.mark('console 로깅')
            for k, v in dict_log.items():
                print(f'{k} : {v}')
            # to WandB
            tc.mark('WandB 로깅')
            wandb.log(dict_log)
            #########################################################################
            
            # 모델 저장. #############################################################
            tc.mark('모델 저장')
            args.path_save = archive.find(args.dir_log, "Model")
            savetool = SaveTool(args.path_save, args)
            
            savetool.save_per_epoch(epoch)
            savetool.save_best(val_score)
            #########################################################################
            
            # 오답 이유 분석 #########################################################
            tc.mark('오답 이유 분석')
            args.path_artifact = archive.find(args.dir_log, "Artifact")
            CMtools = Analsis(val_preds, val_answers, val_data, args.dataset.classes, args)
            
            tc.mark('confusion_matrix 시작')
            CMtools.confusion_matrix(epoch)
            tc.mark('label_incorrected 시작')
            CMtools.label_incorrected(epoch)
            #########################################################################
            tc.print('오답 이유 분석 종료')
        tc.mark_(f'epoch_{epoch} 종료')
            
        if break_flag:
            print(f'Break Loop!!! Because of {break_reason}')
            break
    
    print('Finished Training')
    tc.print('Finished Training')

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command 창에서의 입력')
    
    parser.add_argument('-c', '--path_config', type=str, help='Path String of Config.json')
    
    args_in_cli = parser.parse_args()
    
    # .json 파일과 CLI에서 받은 args를 Args의 인스턴스로 종합함.
    args = Container(
        argparse=args_in_cli, 
        json_path=args_in_cli.path_config
        )
    
    # args 초기화 과정.
    what_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"사용된 device : {what_device}")
    args.device = torch.device(what_device)
    
    args.project_name = args.project_name if args.project_name else str(id(args))
    wandb.run.name = args.project_name
    wandb.run.save()
    
    args.dir_saved = args.dir_saved if args.dir_saved else '/opt/ml/workspace/template/saved'
    args.save_frequency = args.save_frequency if args.save_frequency else 1
    
    wandb.config.update(args.dict())
    
    train(args)
    
