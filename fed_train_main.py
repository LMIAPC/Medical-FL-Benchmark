import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import sys
import os
import argparse
import copy
from utils import progress_bar, Logger, create_model
import data.dataset_define as dataset_define
from sklearn.metrics import f1_score

def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    feature = x
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x, feature


parser = argparse.ArgumentParser(description='Federated Learning Benchmark Main Baselines')
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
parser.add_argument('--mu', type=float, default=0.01, help='parameter used by fedprox/moon')
parser.add_argument('--rho', type=float, default=0.9, help='parameter used by fednova')
parser.add_argument('--restrict', type=float, default=0.5, help='parameter used by fedrs')
parser.add_argument('--non_iid', action='store_true', default=False, help='labels non-iid distribution')
parser.add_argument('--use_f1', action='store_true', default=False, help='evaluate using f1 score')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_f1 = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

resolution = args.resolution 

N_CLASSES = args.class_num
client_num = args.num_clients
client_weights = []

if os.exists('./logs'):
    os.makedirs('./logs')
if args.use_f1:
    sys.stdout = Logger(f'logs/f1score_{args.algorithm}_{args.dataset}.log', sys.stdout)
else:
    sys.stdout = Logger(f'logs/{args.algorithm}_{args.dataset}_{args.non_iid}.log', sys.stdout)

print('==> Preparing data..')
def data_preprocess(dataset_name):
    train_dataloaders = []
    train_sizes = []

    for idx in range(client_num):
        if 'TB' in dataset_name:
            name = ['ChinaSet_train', 'IndiaSet_train', 'MontgomerySet_train']
            train_dirs = [f'../medical/{dataset_name}/{name[idx]}']
        elif 'DR' in dataset_name:
            name = ['APTOS/train', 'Retino/train', 'IDRID/train']
            train_dirs = [f'../medical/{dataset_name}/{name[idx]}']
        if args.non_iid:
            train_dirs = [f'../medical/{dataset_name}/non-iid/client_{idx}']
        else:
            train_dirs = [f'../medical/{dataset_name}/iid/client_{idx}']
        client_dataset = dataset_define.CenDataset(train_dirs, N_CLASSES, resolution)
        train_sizes.append(len(client_dataset))
        print(f'client {idx} image number: {len(client_dataset)}')
        trainloader = torch.utils.data.DataLoader(client_dataset, batch_size=64, shuffle=True, num_workers=2)
        train_dataloaders.append(trainloader)
        if 'TB' in dataset_name or 'DR' in dataset_name:
            torch.save(client_dataset, os.path.join(f'../medical/{dataset_name}', f"data{idx}.pkl"), pickle_protocol=4)
        else:
            if args.non_iid:
                torch.save(client_dataset, os.path.join(f'../medical/{dataset_name}/non-iid', f"data{idx}.pkl"), pickle_protocol=4)
            else:
                torch.save(client_dataset, os.path.join(f'../medical/{dataset_name}/iid', f"data{idx}.pkl"), pickle_protocol=4)
    if 'TB' in dataset_name:
        val_dir = [f'../medical/{dataset_name}/ChinaSet_val', f'../medical/{dataset_name}/IndiaSet_val', f'../medical/{dataset_name}/MontgomerySet_val']
    elif 'DR' in dataset_name:
        val_dir = [f'../medical/{dataset_name}/APTOS/val', f'../medical/{dataset_name}/Retino/val', f'../medical/{dataset_name}/IDRID/val']
    else:
        if args.non_iid:
            val_dir = [f'../medical/{dataset_name}/non-iid/val']
        else:
            val_dir = [f'../medical/{dataset_name}/iid/val']
    valset = dataset_define.CenDataset(val_dir, N_CLASSES, resolution)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
    print(f'val image number: {len(valset)}')
    client_weights = [float(train_sizes[i])/sum(train_sizes) for i in range(client_num)]
    return train_dataloaders, valloader, client_weights

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
    if args.non_iid:
        val_set = torch.load(f'../medical/{args.dataset}/non-iid/data_val.pkl')
    else:
        val_set = torch.load(f'../medical/{args.dataset}/iid/data_val.pkl')
    print(f'test set image number: {len(val_set)}')
    val_dataloaders.append(torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4))
    client_weights=[(float(train_size[i]) / sum(train_size)) for i in range(args.num_clients)]
else:
    train_dataloaders, valloader, client_weights = data_preprocess(args.dataset)
    val_dataloaders.append(valloader)
print('client weights:', client_weights)

# Model
print('==> Building model..')
net = create_model(num_classes=N_CLASSES)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/fed_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

models = [copy.deepcopy(net) for idx in range(client_num)]
server_model = net
optimizers = [optim.Adam(models[idx].parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-4) for idx in range(client_num)]
schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[idx], T_max=200) for idx in range(client_num)]



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
                if args.algorithm == 'nova':
                    tau_eff = sum([client_weights[r]*paras[r] for r in range(len(client_weights))])
                    temp = server_model.state_dict()[key] + tau_eff * (temp - server_model.state_dict()[key])
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(client_weights)):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        if 'running_amp' in key:
            # aggregate at first round only to save communication cost
            server_model.amp_norm.fix_amp = True
            for model in models:
                model.amp_norm.fix_amp = True
    
    return server_model, models

# Training
def train(epoch, models, server_model, optimizers, client_weights, local_epochs=5):
    print('\nGlobal Epoch: %d' % epoch)
    paras = []
    
    for client_idx, model in enumerate(models):
        print('client_idx:', client_idx)
        train_loss = 0
        correct = 0
        total = 0
        if args.algorithm == 'moon':
            cos_sim = nn.CosineSimilarity(dim=-1)
            pre_model = copy.deepcopy(model)
            global_model = copy.deepcopy(server_model)
        if args.algorithm == 'rs':
            ds = train_dataloaders[client_idx].dataset
            uniq_val = np.unique(ds.labels)
            class2data = args.restrict * torch.ones(N_CLASSES)
            for c in uniq_val:
                class2data[c] = 1.0
            class2data = class2data.unsqueeze(dim=0).cuda()
        for ep in range(local_epochs):
            print('Lobal Epoch: %d' % ep)
            model.train()
            trainloader = train_dataloaders[client_idx]
            optimizer = optimizers[client_idx]

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                if args.algorithm == 'moon':
                    with torch.no_grad():
                        _, features_prev_local = pre_model(inputs)
                    del _
                    with torch.no_grad():
                        _, features_global = global_model(inputs)
                    del _
                    torch.cuda.empty_cache()
                    outputs, features = model(inputs)
                else:
                    outputs = model(inputs)

                if args.algorithm == 'rs':
                    outputs *= class2data

                loss = criterion(outputs, targets)

                if args.algorithm == 'prox':
                    w_diff = torch.tensor(0., device=device)
                    for wt, w in zip(server_model.parameters(), model.parameters()):
                        w_diff += torch.pow(torch.norm(w - wt), 2)
                    loss += 0.5 * args.mu * w_diff
                elif args.algorithm == 'moon':
                    # contrastive loss
                    # similarity of positive pair (i.e., w/ global model)
                    pos_similarity = cos_sim(features, features_global).view(-1, 1)
                    # similarity of negative pair (i.e., w/ previous round local model)
                    neg_similarity = cos_sim(features, features_prev_local).view(-1, 1)
                    repres_sim = torch.cat([pos_similarity, neg_similarity], dim=-1)
                    contrast_label = torch.zeros(repres_sim.size(0)).long().cuda()
                    loss_con = criterion(repres_sim, contrast_label)
                    loss  = loss + args.mu * loss_con

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if args.algorithm == 'nova':
            a_i = (local_epochs - args.rho * (1 - pow(args.rho, local_epochs)) / (1 - args.rho)) / (1 - args.rho)
            paras.append(a_i)
            model_para = model.state_dict()
            server_model_para = server_model.state_dict()
            for key in model_para:
                model_para[key] = server_model_para[key] + torch.true_divide(model_para[key] - server_model_para[key], a_i)
            model.load_state_dict(model_para)
    server_model, models = communication(server_model, models, client_weights, paras)
        
def val(epoch, models, server_model, flag=False):
    global best_acc
    global best_f1
    
    with torch.no_grad():
        server_model.eval()
        pred_list_cz = []
        label_list_cz = []
        valloader = val_dataloaders[0]
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.algorithm == 'moon':
                outputs, _ = server_model(inputs)
            else:
                outputs = server_model(inputs)
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
        acc = 100.*correct/total
        print("Server Val Acc:  %.3f" % acc)
    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': server_model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/fed_ckpt.pth')
        best_acc = acc
    if flag:
        f1_macro = 100*f1_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
        print(f'f1 score: {f1_macro}')
        if f1_macro > best_f1:
            best_f1 = f1_macro
            print('best f1 score')


for epoch in range(start_epoch, args.global_epochs):
    train(epoch, models, server_model, optimizers, client_weights, local_epochs=args.local_epochs)
    val(epoch, models, server_model, args.use_f1)
    for idx in range(client_num):
        schedulers[idx].step()
