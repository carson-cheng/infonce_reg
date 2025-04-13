# in the readme of my repo: make sure the command is teed out to a file
# in order to do the plotting later
#def setup():
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('base_model', type=str, help="The base model used for the classification test (only accept 'resnet18', 'wide_resnet101_2', and 'vit_b_16' for now)")
parser.add_argument('dataset', type=str, help="The dataset used for evaluation (only accept 'cifar10', 'cifar100', 'cars', 'dogs', and 'flowers' for now)")
args = parser.parse_args()
ds = ['cifar10', 'cifar100', 'cars', 'dogs', 'flowers']
models = ['resnet18', 'wide_resnet101_2', 'vit_b_16']
if args.base_model not in models:
    raise ValueError(f"Model type not supported! (only accept {models} for now)")
if args.dataset not in ['cifar10', 'cifar100', 'cars', 'dogs', 'flowers']:
    raise ValueError(f"Dataset not supported! (only accept {ds} for now)")
num_classes = [10, 100, 196, 120, 102]
args.num_classes = num_classes[ds.index(args.dataset)]
in_features = [512, 2048, 768]
args.in_features = in_features[models.index(args.base_model)]
print(args.num_classes, args.in_features)
from data import get_loaders, set_seeds
import torch
import torchvision
import torch.nn as nn
import math
from torchvision import transforms
import torch.nn.functional as F
import random
from transformers import ViTForImageClassification
from info_nce import InfoNCE, info_nce
### for reproducibility ###
global nn
def setup(args):
    import torch.nn as nn
    print(torch.distributed.is_available())
    MODELTYPE = args.base_model + "_" + args.dataset
    device = torch.device('cuda')
    if args.base_model == 'vit_b_16':
        base_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        class ViTWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                x = self.model(x)
                return x.logits
        net = ViTWrapper(base_model).to(device)  
        net.model.classifier = nn.Identity()
    if args.base_model == 'wide_resnet101_2':
        net = torchvision.models.wide_resnet101_2(weights='DEFAULT')
        net.fc = nn.Identity()
    if args.base_model == 'resnet18':
        net = torchvision.models.resnet18(weights='DEFAULT')
        net.fc = nn.Identity()
    net=net.to(device)
    trainloader, trainloader2, valloader, testloader = get_loaders(args)
    import torch.optim as optim
    import torch.nn as nn
    ce = nn.CrossEntropyLoss()
    infonce = InfoNCE(reduction='none') # useful for making feature masks (reduction = 'none' expands everything out)
    net.fc = nn.Identity()
    #net.model.classifier = nn.Identity()
    lm = nn.Linear(args.in_features, args.num_classes).to(device)
    embed = nn.Embedding(args.num_classes, args.in_features).to(device)
    #criterion = losses.TripletMarginLoss()
    LR = 0.005
    optimizer = optim.SGD(list(net.parameters()) + list(lm.parameters()) + list(embed.parameters()), lr=LR, momentum=0.9, weight_decay=1e-05) # manually turn it to 0.0005 after epoch 8
    #base_optimizer = optim.SGD
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.8) # maybe this schedule isn't the best...
    globals().update(locals())
net=None
def evaluate_model(loader, phase, limit=5000, model=net):
    correct = 0
    total = 0
    counter = 0
    with torch.no_grad():
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            #print(labels)
            outputs = model(images)
            #print(outputs.shape)
            # the class with the highest energy is what we choose as prediction
            #print(outputs.data)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total >= limit:
                break
    print(f'{phase} accuracy: {100 * correct / total} ({correct} / {total}) %')
    return correct, total_loss / math.ceil(limit / 16)
record = 0
import math
strengths=[0,0.01,0.1] # regularization strength
for test_item in range(7):
    setup(args)
    record, record_fn = 0, ""
    for epoch in range(6):
        print(scheduler.get_last_lr())
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            #'''
            if i % 625 == 0 and (epoch != 0 or i != 0):
                # sample 2000 data points from the train and test sets
                net.eval()
                train_corrects, train_loss = evaluate_model(trainloader2, "train", limit=1000, model=nn.Sequential(net, lm))
                val_corrects, val_loss = evaluate_model(valloader, "val", limit=10000, model=nn.Sequential(net, lm))
                fn = f"models/{val_corrects}_{MODELTYPE}_lr{LR}_{epoch}epochs_{i}_batches.pth"
                torch.save(net.state_dict(), fn)
                #test_corrects, test_loss = evaluate_model(testloader2, "test", limit=10000, model=nn.Sequential(net, lm))
                net.train()
                print(train_loss)
                print(val_loss)
                if val_corrects > record:
                    record, record_fn = val_corrects, fn
            #'''
            if epoch == 5:
                break # stop training here, the last "epoch" is for reporting
            # get the inputs; data is a list of [inputs, labels]
            def closure():
                features = net(inputs)
                y = lm(features)
                loss = ce(y, labels)
                loss += 0.1 * infonce(features, labels.unsqueeze(1).expand([-1, outputs.shape[1]]).to(torch.float32))
                loss.backward()
                return loss
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            logits = lm(outputs)
            if test_item%3==0:
                infonce_loss = infonce(outputs, labels.unsqueeze(1).expand([-1, outputs.shape[1]]).to(torch.float32)) # sample-wise regularization (hard label)
            elif test_item%3==1:
                infonce_loss = infonce(labels.unsqueeze(1).expand([-1, outputs.shape[1]]).to(torch.float32).T, outputs.T) # feature-wise regularization)
            else:
                infonce_loss = infonce(outputs, embed(labels)) # sample-wise regularization (embed)
            ce_loss = ce(logits, labels)
            loss = ce_loss + strengths[math.ceil(test_item/3)] * torch.mean(infonce_loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 125 == 124:
                print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 125}')
                running_loss, running_var_loss = 0.0, 0.0
                with torch.no_grad():
                    counter = 0
            if i % 625 == 624:
                scheduler.step()
        #scheduler.step()
    print('Finished Training')
    print("Loading best checkpoint")
    print(record_fn)
    net.load_state_dict(torch.load(record_fn, weights_only=True))
    net.eval()
    test_corrects, test_loss = evaluate_model(testloader, "test", limit=10000, model=nn.Sequential(net, lm))
    fn = f"models/{test_corrects}_{test_item}_" +record_fn.split("/")[1]
    torch.save(net.state_dict(), fn)
    net.train()