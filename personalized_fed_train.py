import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import argparse
import copy
import sys
import numpy as np
from utils import progress_bar, create_model, Logger
import data.dataset_define as dataset_define
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--data_save', action='store_true', default=False, help='has saved dataset')
parser.add_argument('--dataset', type=str, default='covid', help='name of dataset')
parser.add_argument('--class_num', type=int, default=10, help='number of labeles')
parser.add_argument('--num_clients', type=int, default=1, help='number of clients')
parser.add_argument('--local_epochs', type=int, default=5, help='number of local training epoch')
parser.add_argument('--global_epochs', type=int, default=50, help='number of global training epoch')
parser.add_argument('--resolution', type=int, default=32, help='resolution')
parser.add_argument('--algorithm', type=str, default='avg', help='federated algorithm')
parser.add_argument('--non_iid', action='store_true', default=False, help='labels non-iid distribution')
parser.add_argument('--use_f1', action='store_true', default=False, help='val using f1 score')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_f1 = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

resolution = args.resolution 

N_CLASSES = args.class_num
client_num = args.num_clients
client_weights = []
if args.use_f1:
    sys.stdout = Logger(f'test_{args.algorithm}_{args.dataset}.log', sys.stdout)
else:
    sys.stdout = Logger(f'test_{args.algorithm}_{args.dataset}_{args.non_iid}.log', sys.stdout)

print('==> Preparing data..')
def data_preprocess(dataset_name):
    train_dataloaders = []
    train_sizes = []
    val_dataloaders = []
    for idx in range(client_num):
        if 'TB' in dataset_name:
            name = ['ChinaSet_train', 'IndiaSet_train', 'MontgomerySet_train']
            train_dirs = [f'../medical/{dataset_name}/{name[idx]}']
            client_dataset = dataset_define.CenDataset(train_dirs, N_CLASSES, resolution)
            train_sizes.append(len(client_dataset))
            print(f'client {idx} image number: {len(client_dataset)}')
            torch.save(client_dataset, os.path.join(f'../medical/{dataset_name}/iid', f"data{idx}.pkl"), pickle_protocol=4)
            trainloader = torch.utils.data.DataLoader(client_dataset, batch_size=64, shuffle=True, num_workers=2)
            train_dataloaders.append(trainloader)

            name = ['ChinaSet_test', 'IndiaSet_test', 'MontgomerySet_test']
            val_dir = [f'../medical/{dataset_name}/{name[idx]}']
            val_set = dataset_define.CenDataset(val_dir, N_CLASSES, resolution)
            print(f'client {idx} test image number: {len(val_set)}')
            val_dataloaders.append(torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2))
        elif 'DR' in dataset_name:
            name = ['APTOS/train', 'Retino/train', 'IDRID/train']
            train_dirs = [f'../medical/{name[idx]}']
            client_dataset = dataset_define.CenDataset(train_dirs, N_CLASSES, resolution)
            train_sizes.append(len(client_dataset))
            print(f'client {idx} image number: {len(client_dataset)}')
            torch.save(client_dataset, os.path.join(f'../medical/{name[idx]}', f"data{idx}.pkl"), pickle_protocol=4)
            trainloader = torch.utils.data.DataLoader(client_dataset, batch_size=64, shuffle=True, num_workers=2)
            train_dataloaders.append(trainloader)

            val_dir = [f'../medical/{name[idx]}/../val']
            val_set = dataset_define.CenDataset(val_dir, N_CLASSES, resolution)
            print(f'client {idx} val image number: {len(val_set)}')
            val_dataloaders.append(torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2))
            torch.save(val_set, os.path.join(f'../medical/{name[idx]}../val', f"data_val{idx}.pkl"), pickle_protocol=4)
        else:
            if args.non_iid:
                train_dirs = [f'../medical/{dataset_name}/non-iid/client_{idx}']
            else:
                train_dirs = [f'../medical/{dataset_name}/iid/client_{idx}']
            client_dataset = dataset_define.CenDataset(train_dirs, N_CLASSES, resolution)
            torch.save(client_dataset, os.path.join(f'../medical/{dataset_name}/iid', f"data{idx}.pkl"), pickle_protocol=4)
            trainloader = torch.utils.data.DataLoader(client_dataset, batch_size=64, shuffle=True, num_workers=2)
            train_dataloaders.append(trainloader)
            train_sizes.append(len(client_dataset))

            if args.non_iid:
                val_dir = [f'../medical/{dataset_name}/non-iid/val']
            else:
                val_dir = [f'../medical/{dataset_name}/iid/val']
            val_set = dataset_define.CenDataset(val_dir, N_CLASSES, resolution)
            print(f'client {idx} val image number: {len(val_set)}')
            val_dataloaders.append(torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2))

    client_weights = [float(train_sizes[i])/sum(train_sizes) for i in range(client_num)]
    return train_dataloaders, val_dataloaders, client_weights

# data loader
train_dataloaders = []
val_dataloaders = []
if args.data_save:
    N_CLASSES = args.class_num
    train_size = []
    test_size = []
    for i in range(args.num_clients):
        if args.non_iid:
            train_set = torch.load(os.path.join(f'../medical/{args.dataset}/non-iid', f"data{i}.pkl"))
        else:
            train_set = torch.load(os.path.join(f'../medical/{args.dataset}/iid', f"data{i}.pkl"))
        train_size.append(len(train_set))
        print(f'clinet {i} image number: {train_size[i]}')
        train_dataloaders.append(torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4))
        
        if 'TB' in args.dataset or 'DR' in args.dataset:
            val_set = torch.load(f'../medical/{args.dataset}/data_val{i}.pkl')
        else:
            if args.non_iid:
                val_set = torch.load(f'../medical/{args.dataset}/non-iid/data_val{i}.pkl')
            else:
                val_set = torch.load(f'../medical/{args.dataset}/iid/data_val.pkl')
        print(f'client {i} test set image number: {len(val_set)}')
        val_dataloaders.append(torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4))
    client_weights=[(float(train_size[i]) / sum(train_size)) for i in range(args.num_clients)]
else:
    train_dataloaders, val_dataloaders, client_weights = data_preprocess(args.dataset)
print('client weights:', client_weights)

# Model
print('==> Building model..')
net = create_model(num_classes=N_CLASSES)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

models = [copy.deepcopy(net) for idx in range(client_num)]
server_model = net
optimizers = [optim.Adam(models[idx].parameters(), lr=args.lr, betas=(0.9, 0.999),weight_decay=5e-4) for idx in range(client_num)]
schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[idx], T_max=200) for idx in range(client_num)]
init_sensitivity = [torch.zeros(len(list(models[idx].parameters()))).to(device) for idx in range(client_num)]


def communication(server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.algorithm == 'bn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key and 'num_batches_tracked' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

        else:
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float)
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models

# Training
def train(epoch, models, server_model, optimizers, client_weights, local_epochs=5):
    print('\nGlobal Epoch: %d' % epoch)
    
    for client_idx, model in enumerate(models):
        print('client_idx:', client_idx)
        train_loss = 0
        correct = 0
        total = 0

        trainloader = train_dataloaders[client_idx]
        optimizer = optimizers[client_idx]

        for ep in range(local_epochs):
            print('Local Epoch: %d' % ep)
            model.train()
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                  
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    server_model, models = communication(server_model, models, client_weights)
        
def val(epoch, models, flag=False):
    global best_acc
    global best_f1
    
    with torch.no_grad():
        pred_list_cz = []
        label_list_cz = []
        total_correct = 0
        total_val_sample = 0
        i = 0
        acc_list = []
        for model, valloader in zip(models, val_dataloaders):
            test_loss = 0
            correct = 0
            total = 0
            model.eval()
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if flag:
                    _, pred_cz = outputs.topk(1, 1, True, True)
                    pred_list_cz.extend(((pred_cz.cpu()).numpy()).tolist())
                    label_list_cz.extend(((targets.cpu()).numpy()).tolist())

                progress_bar(batch_idx, len(valloader), 'Val Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            client_acc = 100.*correct/total
            print(f'client {i} acc: {client_acc}')
            acc_list.append(client_acc)
            i += 1
            total_correct += correct
            total_val_sample += total
    acc = 100.*total_correct/total_val_sample
    print("client weight average val Acc:  %.3f" % acc)
    print('client average val Acc: %.3f' % np.mean(acc_list))
    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        best_acc = acc
        for i in range(client_num):
            state = {
                'net': models[i].state_dict(),
                'acc': acc_list[i],
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/fed_ckpt{}.pth'.format(i))
        
    if flag:
        f1_macro = 100*f1_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
        print(f'f1 score: {f1_macro}')
        if f1_macro > best_f1:
            best_f1 = f1_macro
            print('best f1 score')


for epoch in range(start_epoch, args.global_epochs):
    train(epoch, models, server_model, optimizers, client_weights, local_epochs=args.local_epochs)
    val(epoch, models, args.use_f1)
    for idx in range(client_num):
        schedulers[idx].step()
