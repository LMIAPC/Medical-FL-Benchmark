import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score
import numpy as np
from sklearn import metrics
import os
import argparse
import copy
from utils import progress_bar, Logger, create_model
import sys


parser = argparse.ArgumentParser(description='Personalized Federated Learning with Distillation')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--data_save', action='store_true', default=False, help='has saved dataset')
parser.add_argument('--dataset', type=str, default='covid', help='name of dataset')
parser.add_argument('--class_num', type=int, default=10, help='number of labeles')
parser.add_argument('--num_clients', type=int, default=1, help='number of clients')
parser.add_argument('--local_epochs', type=int, default=5, help='number of local training epoch')
parser.add_argument('--resolution', type=int, default=256, help='resolution')
parser.add_argument('--algorithm', type=str, default='avg', help='federated algorithm')
parser.add_argument('--alpha1', type = float, default= 0.7, help = 'alpha1')
parser.add_argument('--alpha2', type = float, default= 0.9, help = 'alpha2')
parser.add_argument('--non_iid', action='store_true', default=False, help='labels non-iid distribution')
parser.add_argument('--use_f1', action='store_true', default=False, help='val using f1 score')

args = parser.parse_args()

if os.exists('./logs'):
    os.makedirs('./logs')
sys.stdout = Logger(f'logs/distill_{args.dataset}_{args.non_iid}.log', sys.stdout)
if args.use_f1:
    sys.stdout = Logger(f'logs/f1score_distill_{args.dataset}.log', sys.stdout)

client_num = args.num_clients
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_acc_server = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
criterion = nn.CrossEntropyLoss()
resolution = args.resolution
client_weights = [1./client_num for i in range(client_num)]
N_CLASSES = args.class_num

# data
print('==> Loading data..')
train_dataloaders = []
val_dataloaders = []
if args.data_save:
    train_size = []
    test_size = []
    for i in range(args.num_clients):
        if args.non_iid:
            train_set = torch.load(os.path.join(f'../medical/{args.dataset}/non-iid', f"data{i}.pkl"))
        else:
            train_set = torch.load(os.path.join(f'../medical/{args.dataset}/iid', f"data{i}.pkl"))
        train_size.append(len(train_set))
        train_dataloaders.append(torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2))
        if 'TB' in args.dataset or 'DR' in args.dataset:
            val_set = torch.load(f'../medical/{args.dataset}/data_val{i}.pkl')
        else:
            if args.non_iid:
                val_set = torch.load(f'../medical/{args.dataset}/non-iid/data_val{i}.pkl')
            else:
                val_set = torch.load(f'../medical/{args.dataset}/iid/data_val.pkl')
        val_dataloaders.append(torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2))
    client_weights=[(float(train_size[i]) / sum(train_size)) for i in range(args.num_clients)]
else:
    print('error')
    exit()

# Model
print('==> Building model..')
net = create_model(num_classes=N_CLASSES)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

server_model = net
models = [copy.deepcopy(net) for idx in range(client_num)]
models_deputy = [copy.deepcopy(net) for idx in range(client_num)]
optimizers = [optim.Adam(models[idx].parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-4) for idx in range(client_num)]
optimizers_deputy = [optim.Adam(models_deputy[idx].parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-4) for idx in range(client_num)]
schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[idx], T_max=200) for idx in range(client_num)]
schedulers_deputy = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizers_deputy[idx], T_max=200) for idx in range(client_num)]

def communication(server_model, models, client_weights, paras):
    with torch.no_grad():
        # aggregate params
        for key in server_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
            else:
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float)
                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                for client_idx in range(len(client_weights)):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        if 'running_amp' in key:
            # aggregate at first round only to save communication cost
            server_model.amp_norm.fix_amp = True
            for model in models:
                model.amp_norm.fix_amp = True
    
    return server_model, models

def train(epoch, models, models_deputy, server_model, optimizers, optimizers_deputy, client_weights, local_epochs=5):
    print('\nGlobal Epoch: %d' % epoch)

    paras = []
    
    for client_idx, (model, model_deputy) in enumerate(zip(models, models_deputy)):
        print('client_idx:', client_idx)
        
        train_loss = 0
        correct = 0
        total = 0
        DET_stage = 0
        for ep in range(local_epochs):
            print('Local Epoch: %d' % ep)
            train_loader = train_dataloaders[client_idx]
            train_loss, train_acc, train_f1 = test(model, train_loader)
            train_loss_deputy, train_acc_deputy, train_f1_deputy = test(model_deputy, train_loader)
            alpha1 = args.alpha1
            alpha2 = args.alpha2

            if (train_f1_deputy < alpha1 * train_f1) or DET_stage == 0:
                DET_stage = 1
                print('recover')
            elif (train_f1_deputy >= alpha1 * train_f1 and DET_stage == 1) or (DET_stage >= 2 and train_f1_deputy < alpha2 * train_f1):           
                DET_stage = 2
                print('exchange')
            elif train_f1_deputy >= alpha2 * train_f1 and DET_stage >= 2:          
                DET_stage = 3
                print('sublimate')
            else:
                print('***********************Logic error************************')
                DET_stage = 4
            
            model.train()
            model_deputy.train()
            optimizer = optimizers[client_idx]
            optimizer_deputy = optimizers_deputy[client_idx]

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                output = model(inputs)
                output_deputy = model_deputy(inputs) 

                if DET_stage == 1:
                    # personalized is teacher                
                    loss_ce = criterion(output, targets)
                    loss = loss_ce
                    
                    loss_deputy_ce = criterion(output_deputy, targets)
                    loss_deputy_kl = F.kl_div(F.log_softmax(output_deputy, dim = 1), F.softmax(output.clone().detach(), dim=1), reduction='batchmean')
                    loss_deputy = loss_deputy_ce + loss_deputy_kl
                elif DET_stage == 2:
                    # mutual learning DET_stage = 2
                    loss_ce = criterion(output, targets)
                    loss_kl = F.kl_div(F.log_softmax(output, dim = 1), F.softmax(output_deputy.clone().detach(), dim=1), reduction='batchmean')
                    loss = loss_ce + loss_kl
                        
                    loss_deputy_ce = criterion(output_deputy, targets)           
                    loss_deputy_kl = F.kl_div(F.log_softmax(output_deputy, dim = 1), F.softmax(output.clone().detach(), dim=1), reduction='batchmean')
                    loss_deputy = loss_deputy_ce + loss_deputy_kl
                elif DET_stage == 3:
                    # deputy is teacher
                    loss_ce = criterion(output, targets)
                    loss_kl = F.kl_div(F.log_softmax(output, dim = 1), F.softmax(output_deputy.clone().detach(), dim=1), reduction='batchmean')
                    loss = loss_ce + loss_kl
                        
                    loss_deputy_ce = criterion(output_deputy, targets)           
                    loss_deputy = loss_deputy_ce
                else:
                    # default mutual learning
                    loss_ce = criterion(output, targets)
                    loss_kl = F.kl_div(F.log_softmax(output, dim = 1), F.softmax(output_deputy.clone().detach(), dim=1), reduction='batchmean')
                    loss = loss_ce + loss_kl
                        
                    loss_deputy_ce = criterion(output_deputy, targets)           
                    loss_deputy_kl = F.kl_div(F.log_softmax(output_deputy, dim = 1), F.softmax(output.clone().detach(), dim=1), reduction='batchmean')
                    loss_deputy = loss_deputy_ce + loss_deputy_kl

                loss.backward()
                loss_deputy.backward()
                optimizer.step()
                optimizer_deputy.step()  

                train_loss += loss.item()
                _, predicted = output.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(train_loader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    server_model, models_deputy = communication(server_model, models_deputy, client_weights, paras)


def test(model, test_loader):
    model.eval()
    model.to(device) 
    test_loss = 0
    correct = 0
    targets = []

    label_list_cz = []
    pred_list_cz = []
    output_list_cz = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output = model(data)
        _, pred_cz = output.topk(1, 1, True, True)#
        pred_list_cz.extend(
            ((pred_cz.cpu()).numpy()).tolist())
        label_list_cz.extend(
            ((target.cpu()).numpy()).tolist())
        
        test_loss += criterion(output, target).item()
        pred = output.data.max(1)[1]
        output_list_cz.append(torch.nn.functional.softmax(output, dim=-1).cpu().detach().numpy())
        correct += pred.eq(target.view(-1)).sum().item()

    mean_acc = 100*metrics.accuracy_score(label_list_cz, pred_list_cz)
    print('mean acc: %.3f' % mean_acc)
    f1_macro = 100*metrics.f1_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
    print('f1: %.3f' % f1_macro)
    return test_loss/len(test_loader), mean_acc, f1_macro

def val(epoch, models, server_model, flag=False):
    global best_acc
    global best_acc_server
    
    with torch.no_grad():
        acc_list = []
        label_list_cz = []
        pred_list_cz = []

        for model, valloader in zip(models, val_dataloaders):
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                _, pred_cz = outputs.topk(1, 1, True, True)#
                pred_list_cz.extend(
                    ((pred_cz.cpu()).numpy()).tolist())
                label_list_cz.extend(
                    ((targets.cpu()).numpy()).tolist())

                progress_bar(batch_idx, len(valloader), 'Val Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            acc_list.append(100.*correct/total)
        client_acc = np.mean(acc_list)
        print("client average val Acc:  %.3f" % client_acc)
        if flag:
            f1_macro = 100*f1_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
            print(f'f1 score: {f1_macro}')

        server_model.eval()
        acc_list = []
        for model, valloader in zip(models, val_dataloaders):
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = server_model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(valloader), 'Val Loss: %.3f | Server Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            acc_list.append(100.*correct/total)
        server_acc = np.mean(acc_list)
        print("server average val Acc:  %.3f" % server_acc)
    # Save checkpoint.
    if client_acc > best_acc:
        print('Saving..')
        for i in range(client_num):
            state = {
                'net': models[i].state_dict(),
                'acc': acc_list[i],
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/fed_ckpt{}.pth'.format(i))
        best_acc = client_acc


for epoch in range(start_epoch, 50):
    train(epoch, models, models_deputy, server_model, optimizers, optimizers_deputy, client_weights, args.local_epochs)
    val(epoch, models, server_model, args.use_f1)
    for idx in range(client_num):
        schedulers[idx].step()