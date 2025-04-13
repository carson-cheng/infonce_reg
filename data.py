import torchvision
import torchvision.transforms as transforms
import transformers
import torch
import random
import numpy as np
def set_seeds(seed):
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    transformers.set_seed(seed)
    transformers.enable_full_determinism(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
def get_loaders(args):
    transform = transforms.Compose(
    [transforms.ToTensor(),
    #transforms.Normalize((0.5), (0.5))]) # for mnist (1-channel)
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if args.dataset in ['cars', 'dogs', 'flowers']:
        batch_size = 16
    elif args.dataset in ['cifar10', 'cifar100']:
        batch_size = 32
    set_seeds(0)
    if args.dataset == 'flowers':
        trainset = torchvision.datasets.Flowers102(root="./data", split="train",
                                            download=True, transform=transform)
        valset = torchvision.datasets.Flowers102(root="./data", split="val",
                                            download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
        set_seeds(1)
        trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4) # for eval
        set_seeds(0)
        testset = torchvision.datasets.Flowers102(root="./data", split="test",
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=4)
        set_seeds(1)
        #testloader2 = torch.utils.data.DataLoader(testset, batch_size=batch_size,
        #                                    shuffle=False, num_workers=4) # for eval     
        set_seeds(0)
    elif args.dataset == 'cifar10':
        train_indexes, val_indexes = range(0, 45000), range(45000, 50000)
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=True, transform=transform)
        valset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=True, transform=transform)
        trainset, valset = torch.utils.data.Subset(trainset, train_indexes), torch.utils.data.Subset(valset, val_indexes)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
        set_seeds(1)
        trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4) # for eval
        set_seeds(0)
        testset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=4)
    elif args.dataset == 'cifar100':
        train_indexes, val_indexes = range(0, 45000), range(45000, 50000)
        trainset = torchvision.datasets.CIFAR100(root="./data", train=True,
                                            download=True, transform=transform)
        valset = torchvision.datasets.CIFAR100(root="./data", train=True,
                                            download=True, transform=transform)
        trainset, valset = torch.utils.data.Subset(trainset, train_indexes), torch.utils.data.Subset(valset, val_indexes)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
        set_seeds(1)
        trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4) # for eval
        set_seeds(0)
        testset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=4)
    elif args.dataset == 'cars':
        train_path = '/root/.cache/kagglehub/datasets/jutrera/stanford-car-dataset-by-classes-folder/versions/2/car_data/car_data/train'
        test_path = '/root/.cache/kagglehub/datasets/jutrera/stanford-car-dataset-by-classes-folder/versions/2/car_data/car_data/test'
        val_indexes = random.sample(list(range(8144)), 1000)
        train_indexes = [x for x in range(8144) if x not in val_indexes]
        trainset = torchvision.datasets.ImageFolder(root=train_path,
                                            transform=transform)
        valset = torchvision.datasets.ImageFolder(root=train_path,
                                            transform=transform)
        trainset, valset = torch.utils.data.Subset(trainset, train_indexes), torch.utils.data.Subset(valset, val_indexes)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
        set_seeds(1)
        trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4) # for eval
        set_seeds(0)
        testset = torchvision.datasets.ImageFolder(root=test_path,
                                        transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=4)
    elif args.dataset == 'dogs':
        train_path = '/root/.cache/kagglehub/datasets/miljan/stanford-dogs-dataset-traintest/versions/1/cropped/train'
        test_path = '/root/.cache/kagglehub/datasets/miljan/stanford-dogs-dataset-traintest/versions/1/cropped/test'
        val_indexes = random.sample(list(range(12000)), 1000)
        train_indexes = [x for x in range(12000) if x not in val_indexes]
        trainset = torchvision.datasets.ImageFolder(root=train_path,
                                            transform=transform)
        valset = torchvision.datasets.ImageFolder(root=train_path,
                                            transform=transform)
        trainset, valset = torch.utils.data.Subset(trainset, train_indexes), torch.utils.data.Subset(valset, val_indexes)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
        set_seeds(1)
        trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4) # for eval
        set_seeds(0)
        testset = torchvision.datasets.ImageFolder(root=test_path,
                                        transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=4)
    return trainloader, trainloader2, valloader, testloader