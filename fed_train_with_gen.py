import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import copy
import os
import argparse
import sys
import torch.nn.functional as F
from utils import progress_bar, Logger, create_model
import data.dataset_define as dataset_define
from sklearn.metrics import f1_score

def communication(server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
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
        if 'running_amp' in key:
            # aggregate at first round only to save communication cost
            server_model.amp_norm.fix_amp = True
            for model in models:
                model.amp_norm.fix_amp = True
    
    return server_model, models

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--use_diffusion', action='store_true', default=False,
                    help='use diffusion data')
parser.add_argument('--data_save', action='store_true', default=False, help='has saved dataset')
parser.add_argument('--dataset_name', type=str, default='', help='name of dataset')
parser.add_argument('--class_num', type=int, default=10, help='number of labeles')
parser.add_argument('--resolution', type=int, default=32, help='resolution')
parser.add_argument('--num_clients', type=int, default=1, help='number of clients')
parser.add_argument('--local_epochs', type=int, default=5, help='number of local training epoch')
parser.add_argument('--beta', type=float, default=None, help='smooth parameters for generated data labels')
parser.add_argument('--use_f1', action='store_true', default=False, help='val using f1 score')
parser.add_argument('--non_iid', action='store_true', default=False, help='use non_iid data partition')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_f1 = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

resolution = args.resolution
N_CLASSES = args.class_num
client_num = args.num_clients

if os.exists('./logs'):
    os.makedirs('./logs')
if args.use_diffusion:
    if args.beta is not None:
        sys.stdout = Logger(f'logs/test_soft_label_new_{args.beta}_{args.dataset_name}.log', sys.stdout)
    else:
        sys.stdout = Logger(f'logs/f1score_diffusion_{args.dataset_name}.log', sys.stdout)
else:
    sys.stdout = Logger(f'logs/test_avg_{args.dataset_name}_{args.non_iid}.log', sys.stdout)

# Data
print('==> Preparing data..')
def data_preprocess(dataset_name):
    train_dataloaders = []
    train_sizes = []
    for idx in range(client_num):
        if args.use_diffusion:
            if args.non_iid:
                train_dirs = [f'../medical/{dataset_name}/non-iid/client_{idx}']
            else:   
                train_dirs = [f'../medical/{dataset_name}/iid/client_{idx}', f'../DDPM/output/{dataset_name}_client_{idx}']
            if 'TB' in dataset_name:
                name = ['ChinaSet_train', 'IndiaSet_train', 'MontgomerySet_train']
                train_dirs = [f'../medical/{dataset_name}/{name[idx]}', f'../DDPM/output/{dataset_name}_{name[idx]}']
            elif 'DR' in dataset_name:
                name = ['APTOS', 'Retino', 'IDRID']
                train_dirs = [f'../medical/{dataset_name}/{name[idx]}', f'../DDPM/output/{dataset_name}_{name[idx]}']
        else:
            if args.non_iid:
                train_dirs = [f'../medical/{dataset_name}/non-iid/client_{idx}']
            else:
                train_dirs = [f'../medical/{dataset_name}/iid/client_{idx}']
            if 'TB' in dataset_name:
                name = ['ChinaSet_train', 'IndiaSet_train', 'MontgomerySet_train']
                train_dirs = [f'../medical/{dataset_name}/{name[idx]}']
            elif 'DR' in dataset_name:
                name = ['APTOS', 'Retino', 'IDRID']
                train_dirs = [f'../medical/{dataset_name}/{name[idx]}/train']
        client_dataset = dataset_define.CenDataset(train_dirs, N_CLASSES, resolution, beta=args.beta)
        train_sizes.append(len(client_dataset))
        print(f'clinet {idx} image number: {len(client_dataset)}')
        trainloader = torch.utils.data.DataLoader(client_dataset, batch_size=64, shuffle=True, num_workers=2)
        train_dataloaders.append(trainloader)
        save_path = ''
        if args.use_diffusion:
            if args.non_iid:
                if args.beta is not None:
                    save_path = f'../medical/{dataset_name}/non-iid/diffusion/new_label'
                else:
                    save_path = f'../medical/{dataset_name}/non-iid/diffusion'
            else:
                if args.beta is not None:
                    save_path = f'../medical/{dataset_name}/iid/diffusion/new_label'
                else:
                    save_path = f'../medical/{dataset_name}/iid/diffusion'
            os.makedirs(save_path, exist_ok=True)
            torch.save(client_dataset, os.path.join(save_path, f"data{idx}.pkl"), pickle_protocol=4)
        else:
            if args.non_iid:
                os.makedirs(f'../medical/{dataset_name}/non-iid', exist_ok=True)
                torch.save(client_dataset, os.path.join(f'../medical/{dataset_name}/non-iid', f"data{idx}.pkl"), pickle_protocol=4)
            else:
                os.makedirs(f'../medical/{dataset_name}/iid', exist_ok=True)
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

train_dataloaders = []
if args.data_save:
    print('saved')
    train_sizes = []
    for i in range(args.num_clients):
        if args.use_diffusion:
            if args.beta is not None:
                train_set = torch.load(os.path.join(f'../medical/{args.dataset_name}/iid/diffusion/', f"data{i}.pkl"))
            else:
                train_set = torch.load(os.path.join(f'../medical/{args.dataset_name}/iid/diffusion/new_label', f"data{i}.pkl"))
        else:
            if args.non_iid:
                train_set = torch.load(os.path.join(f'../medical/{args.dataset_name}/non-iid', f"data{i}.pkl"))
            else:
                train_set = torch.load(os.path.join(f'../medical/{args.dataset_name}/iid', f"data{i}.pkl"))
        train_sizes.append(len(train_set))
        print(f'clinet {i} image number: {len(train_set)}')
        train_dataloaders.append(torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0))
    if args.non_iid:
        val_set = torch.load(f'../medical/{args.dataset_name}/non-iid/data_val.pkl')
    else:
        val_set = torch.load(f'../medical/{args.dataset_name}/iid/data_val.pkl')
    print(f'test image number: {len(val_set)}')
    valloader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)
    client_weights = [float(train_sizes[i])/sum(train_sizes) for i in range(args.num_clients)]
else:
    train_dataloaders, valloader, client_weights = data_preprocess(args.dataset_name)

# Model
print('==> Building model..')
net = create_model(num_classes=N_CLASSES)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

val_criterion = nn.CrossEntropyLoss()
criterion = nn.KLDivLoss(reduction='batchmean') if args.beta is not None else nn.CrossEntropyLoss()
client_models = [copy.deepcopy(net) for idx in range(client_num)]
server_model = net
optimizers = [optim.Adam(client_models[idx].parameters(), lr=args.lr, betas=(0.9, 0.999),weight_decay=5e-4) for idx in range(client_num)]
schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[idx], T_max=200) for idx in range(client_num)]

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint_diffusion'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint_diffusion/ckpt.pth')
    server_model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    for i in range(client_num):
        client_models[i].load_state_dict(checkpoint['net'])

# Training
def train(epoch, models, server_model, optimizers, client_weights, local_epochs=5):
    print('\nGlobal Epoch: %d' % epoch)
    
    for client_idx, model in enumerate(models):
        print('client_idx:', client_idx)
        train_loss = 0
        correct = 0
        total = 0
        for ep in range(local_epochs):
            print('Lobal Epoch: %d' % ep)
            model.train()
            trainloader = train_dataloaders[client_idx]
            optimizer = optimizers[client_idx]

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if args.use_diffusion and args.beta is not None:
                    outputs = F.log_softmax(outputs, dim=1)
                    targets = F.softmax(targets, dim=1)
                    loss = criterion(outputs, targets)
                else:
                    loss = val_criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                if args.use_diffusion and args.beta is not None:
                    correct += predicted.eq(torch.argmax(targets, dim=1)).sum().item()
                else:
                    correct += predicted.eq(targets).sum().item()
                progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    server_model, models = communication(server_model, models, client_weights)

def val(epoch, server_model, flag=False):
    global best_acc
    global best_f1
    server_model.eval()
    test_loss = 0
    correct = 0
    total = 0
    label_list_cz = []
    pred_list_cz = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = server_model(inputs)
            loss = val_criterion(outputs, targets)

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

    if flag:
        f1_macro = 100*f1_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
        print(f'f1 score: {f1_macro}')
        if f1_macro > best_f1:
            best_f1 = f1_macro
            print('best')
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_diffusion'):
            os.mkdir('checkpoint_diffusion')
        torch.save(state, './checkpoint_diffusion/ckpt.pth')
        best_acc = acc

print(client_weights)
for epoch in range(start_epoch, start_epoch+50):
    train(epoch, client_models, server_model, optimizers, client_weights, local_epochs=args.local_epochs)
    val(epoch, server_model, args.use_f1)
    for idx in range(client_num):
        schedulers[idx].step()
