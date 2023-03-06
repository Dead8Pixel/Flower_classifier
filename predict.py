import argparse
from utils import ModelCreator
import torch
from torch import nn
from torchvision import models
from PIL import Image
import json
import numpy as np


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(dest="image_path",help="Path to the image used for prediciton")
    parser.add_argument(dest="checkpoint",help="Path to the trained model checkpoint")
    parser.add_argument('--top_k',dest='topk',help="Number of the top classes to display",type=int,default=1)
    parser.add_argument('--category_names',dest='cat_names',help="Json file that matches class labels with flowers category names",default='cat_to_name.json')
    parser.add_argument('--gpu',action='store_true',dest='use_gpu',help="Use gpu for model inference")
    args = parser.parse_args()

    return args.image_path, args.checkpoint, args.topk, args.cat_names, args.use_gpu

def model_loader(filepath):
    checkpoint = torch.load(filepath)
    
    #Create the model with the parameters stored and load saved state
    mod = ModelCreator(checkpoint['arch'])
        
    classifier = nn.Sequential(
          nn.Linear(*checkpoint['input_hidden']),
          nn.ReLU(),
          nn.Dropout(p=0.25),
          nn.Linear(*checkpoint['hidden_output']),
          nn.LogSoftmax(dim = 1)
        )
    mod.classifier = classifier
    mod.load_state_dict(checkpoint['model_state_dict'])
    mod.class_to_idx = checkpoint['class_to_idx']
    
    return mod

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Torch tensor
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    #Resize the image
    im = Image.open(image)
    
    im.resize((256,256))
    #Center crop 
    left = int(im.size[0]/2-224/2)
    upper = int(im.size[1]/2-224/2)
    right = left +224
    lower = upper + 224
    
    im = im.crop((left, upper,right,lower))
    ##Normalize
    np_image = np.array(im)
    np_image = np_image / np.array([255,255,255]).reshape(1,1,3)
    np_image = ( np_image - np.array([0.485, 0.456, 0.406]).reshape(1,1,3) ) / np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    np_image = np_image.transpose((2, 0, 1))
    
    
    return torch.Tensor(np_image).type(torch.FloatTensor) 



def predict(model,image,topk,use_gpu,cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''  
    device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
    model = model.to(device)
    image = image.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        probas, labels = torch.exp(output).topk(topk,dim=1)
    
    #Defining a reverse dictionnary to find the classes
    reverse_idx = dict()
    for i in model.class_to_idx :
        reverse_idx[str(model.class_to_idx[i])] = i
        
    labels = labels.to('cpu')
    probas = probas.to('cpu')
    
    labels = labels[0].numpy()
    probas = probas[0].numpy()
    
    for i in range(len(labels)):
        labels[i] = reverse_idx[str(labels[i])]
    
    names = [cat_to_name[str(i)] for i in list(labels)]


    return list(probas), names


################################################ Main Program ######################################################################

#Parsing arguments from the command line
imgpath, checkpt , topk, cat_names_file, use_gpu = argument_parser()

#Load dictionnary for label / category mapping
with open(cat_names_file, 'r') as f:
    cat_to_name = json.load(f)


model = model_loader(checkpt)
image = process_image(imgpath).unsqueeze(0)

##Predit topk classes for the image
probas , classes = predict(model,image,topk,use_gpu,cat_to_name)


print(f"The top {topk} category prediction{'s' if topk > 1 else ''} for input image {'are' if topk > 1 else 'is'} : ")

for x,y in zip(probas,classes):
    print(f'The flower is a {y} with a probability of {x * 100:.2f}%')



