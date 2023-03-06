import argparse
import torch
from torchvision import transforms, models
from torchvision.datasets import ImageFolder



def ArgumentReader():
    '''
    Parses Arguments from the command line
    Input : None
    
    Output : data_dir, save_dir, arch, lr, hidden, epochs, use_gpu
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(dest="data_dir",help="Data directory of the training  images")
    parser.add_argument('--save_dir',dest="save_dir",help="Directory for saving checkpoints",default='.')
    parser.add_argument('--arch',dest='arch',help='Architecure for the base model',choices=['vgg11','vgg13','vgg16','vgg19'],default='vgg16')
    parser.add_argument('--learning_rate',dest='lr',help='Sets the learning rate for the model',default=0.01,type=float)
    parser.add_argument('--hidden_units',dest='hidden',help='Sets the number of input and output hidden units for tthe classifier',default=512,type=int)
    parser.add_argument('--epochs',dest='epochs',help='Sets the number of training epochs',default=8,type=int)
    parser.add_argument('--gpu',dest='use_gpu',help='Use GPU for training if available',action='store_true')

    args = parser.parse_args()

    return args.data_dir, args.save_dir, args.arch,args.lr, args.hidden, args.epochs ,args.use_gpu



def ImageTransforms():
    '''
    Defines tranformations to be applied to the training , validation and test images respectively.
    Input : None

    Output : train_transform, valid_transform, test_transform
    '''
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
    
    return train_transform, valid_transform, test_transform



def DataLoader(data_dir,transforms):
    """
    Creates a generator that supplies training, validation and test images
    Input : data_dir , transforms

    Output : train_loader, valid_loader, test_loader
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_t , valid_t, test_t = transforms

    train_set = ImageFolder(train_dir, transform = train_t)
    valid_set = ImageFolder(valid_dir, transform = valid_t)
    test_set = ImageFolder(test_dir, transform = test_t)


    trainloader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_set, batch_size = 64)

    return trainloader, validloader, testloader,train_set.class_to_idx



def ModelCreator(arch) :
    model = None
    if arch == "vgg11":
        model = models.vgg11(pretrained=True)

    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)

    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)

    elif arch == "vgg19":
        model = models.vgg19(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    return model