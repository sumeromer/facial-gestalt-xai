import argparse
import pathlib
import sys
import yaml
import random
import shutil
import numpy as np
import pandas as pd
from loguru import logger

import torch
import torchmetrics
from torch.utils import tensorboard
import datasets

def FacialPhenotypingNet(architecture):
    
    if architecture=='VGGFace2_ResNet50':
        from models.VGGFace2_ResNet import resnet50_adapted
        weight_file = './models/weights/VGGFace2_pretrained_models/resnet50_ft_weight.pkl'
        model = resnet50_adapted(weight_file=weight_file, num_classes=12)
    else:
        raise ValueError('Model architecture not implemented! (only "VGGFace2_ResNet50")')
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(train_loader, model, loss_function, accuracy, optimizer, scheduler, epoch, writer, device):
    model.train()
    train_perf = []
    num_batch = len(train_loader)
    for iteration, train_batch in enumerate(train_loader):
        
        # load images and labels
        images, labels, _, _, _ = train_batch
        images, labels = images.to(device), labels.to(device)

        # PyTorch batch normalization throws an exception when batch is composed of a single sample.
        # However, this can happen. If the first dimension is one, then, we simply repeat it.
        images = images.repeat(2,1,1,1) if images.size(0)==1 else images
        labels = labels.repeat(2) if labels.size(0)==1 else labels

        # forward pass
        output = model(images)

        # calculate loss
        train_loss = loss_function(output, labels)

        # calculate accuracy
        _, predictions = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        train_accuracy = accuracy(predictions.cpu(), labels.cpu())
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        #scheduler.step(epoch+iteration/num_batch)

        train_perf.append([epoch+iteration/num_batch, train_loss.item(), train_accuracy.item()])
        writer.add_scalar('Loss/train', train_loss.item(), epoch*num_batch + iteration)
        writer.add_scalar('Accuracy/train', train_accuracy.item(), epoch*num_batch + iteration )

    scheduler.step()
    train_loss, train_accuracy = np.array(train_perf)[:,1].mean(), np.array(train_perf)[:,2].mean()
    logger.info('[Train]  %02d   Loss %2.3f   Accuracy %2.3f   (lr=%2.6f)'%(epoch, train_loss, train_accuracy, scheduler.optimizer.param_groups[0]['lr']))

def test(test_loader, model, loss_function, accuracy, epoch, writer, device):
    model.eval()
    test_perf = []
    for iteration, val_batch in enumerate(test_loader):
        # load images and labels
        images, labels, _, _, _ = val_batch
        images, labels = images.to(device), labels.to(device)

        # forward pass
        output = model(images)

        # calculate loss
        test_loss = loss_function(output, labels)

        # calculate accuracy
        _, predictions = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        test_accuracy = accuracy(predictions.cpu(), labels.cpu())
        test_perf.append([epoch, test_loss.item(), test_accuracy.item()])        

    test_loss, test_accuracy = np.array(test_perf)[:,1].mean(), np.array(test_perf)[:,2].mean()
    logger.info('[Val]    %02d   Loss %2.3f   Accuracy %2.3f'%(epoch, test_loss, test_accuracy))
    writer.add_scalar('Loss/val', test_loss, epoch+1)
    writer.add_scalar('Accuracy/val', test_accuracy, epoch+1)
    return test_accuracy


def main(args):

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms = True

    # create experiment folder
    output_folder = pathlib.Path('results', args.architecture, args.fold)
    if output_folder.exists() and output_folder.is_dir():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # create tensorboard writer
    writer = tensorboard.SummaryWriter(log_dir=output_folder) 

    # logger
    logger.info('Train NIH-Faces Baseline CNN Model')

    # set GPU device ID
    device = torch.device("cuda:%d"%args.device if torch.cuda.is_available() and args.device>=0 else "cpu")

    # datasets
    # root dataset directory containing:
    # * ./images : containing raw images (N=3547)
    # * ./features/face-parser (segmentation maps of all images)
    # Segmentation maps are not for training baseline model, but later for XAI evaluation.
    train_dataset = datasets.NIHFacesDataset(root_dir=args.dataset_folder,
                                             metadata_file='./metadata/partitions.csv', 
                                             fold=args.fold, 
                                             split='train', 
                                             mean_bgr=None,
                                             image_size=224,
                                             flip=True)

    test_dataset = datasets.NIHFacesDataset(root_dir=args.dataset_folder,
                                            metadata_file='./metadata/partitions.csv',
                                            fold=args.fold, 
                                            split='val', 
                                            mean_bgr=train_dataset.mean_bgr,
                                            image_size=224,
                                            flip=False)
    # data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size, 
                                            shuffle=True,  
                                            num_workers=8, 
                                            pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=args.batch_size, 
                                            shuffle=False, 
                                            num_workers=8, 
                                            pin_memory=False)
   
    model = FacialPhenotypingNet(args.architecture).to(device)
    logger.info('Total number of trainable parameters=%2.3fM'%(count_parameters(model)/10e5))
    

    # Previously (i.e., in GestaltMatcher experiments), I used weighted cross-entropy, 
    # however here, I do not use any additional term or regularizer intentionally.
    # It is better to see the behavior of straightforward finetuned model explanation.
    loss_function = torch.nn.CrossEntropyLoss(reduction='mean').to(device) # weight=class_weights
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes)

    # Optimizer
    optimizer = torch.optim.SGD([
        {"params": model.conv1.parameters()},
        {"params": model.layer1.parameters()},
        {"params": model.layer2.parameters()},
        {"params": model.layer3.parameters()},
        {"params": model.layer4.parameters()},
        {"params": model.fc.parameters(), 'lr': 10*args.learning_rate} 
    ], lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)

    # Learning rate scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=0)
    
    # main loop
    for epoch in range(1, args.num_epoch+1):

        # train
        train(train_loader, model, loss_function, accuracy, optimizer, scheduler, epoch, writer, device)
        
        # test
        test_accuracy = test(test_loader, model, loss_function, accuracy, epoch, writer, device)
        
        # save model
        torch.save(model, pathlib.Path(output_folder, 'epoch-%02d-test_accuracy-%2.3f.pt'%(epoch, test_accuracy) ))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture',   default='VGGFace2_ResNet50',   type=str, help='experiment name')
    parser.add_argument('--seed',           default=42,      type=int, help='random seed')
    parser.add_argument('--device',         default=0,       type=int, help='device: cpu (-1), cuda: 0, 1')
    parser.add_argument('--project_root',   default='./',    type=str, help='project root')
    parser.add_argument('--dataset_folder', type=str, help='Root data directory containing images subfolder with all NIH-Faces.')
    parser.add_argument('--num_classes',    default=12,      type=int,    help='number of classes')

    parser.add_argument('--learning_rate',  default=0.001,   type=float,  help='learning rate')
    parser.add_argument('--weight_decay',   default=0.0,     type=float,  help='weight decay')
    parser.add_argument('--momentum',       default=0.9,     type=float,  help='momentum')
    parser.add_argument('--step_size',      default=25,      type=int,    help='step size')
    parser.add_argument('--num_epoch',      default=35,      type=int,    help='number of epochs')

    parser.add_argument('--batch_size',     default=32,      type=int,    help='batch size')
    parser.add_argument('--fold',           default='fold-1',type=str,    help='fold: 1-5')
    args = parser.parse_args()
    main(args)
