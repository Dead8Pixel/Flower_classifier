from utils import ArgumentReader,ImageTransforms, DataLoader, ModelCreator
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
from torchvision.datasets import ImageFolder



#Read Arguments from the command line
data_dir, save_dir, arch, learnrate, hidden, epochs , gpu  = ArgumentReader()


#Load the image transforms
transforms = ImageTransforms()


#Load images into loaders

train_loader, valid_loader, test_loader ,class_to_idx = DataLoader(data_dir,transforms)

##Load pretrained model
model = ModelCreator(arch)

classifier = nn.Sequential(
          nn.Linear(25088, hidden),
          nn.ReLU(),
          nn.Dropout(p=0.25),
          nn.Linear(hidden, 102),
          nn.LogSoftmax(dim = 1)
        )

model.classifier = classifier


#Define the optimizer and loss function

device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr = learnrate,momentum=0.9)
model.to(device)

#Training Loop
steps = 0
print_every = 10
running_loss = 0

print("Begin Training")

for i in range(epochs):
    for images, labels in train_loader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        #Forward pass, backward pass and loss calculation
        out = model.forward(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            
            model.eval()
            
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    log_ps = model.forward(inputs)
                    batch_loss = criterion(log_ps, labels)
                    valid_loss += batch_loss.item()

                    # Calculate validation accuracy
                    ps = torch.exp(log_ps)
                    top_p, top_cl = ps.topk(1, dim=1)
                    compare = top_cl == labels.view(*top_cl.shape)
                    accuracy += torch.mean(compare.type(torch.FloatTensor)).item()

            print(f"Epoch {i+1}/{epochs}, "
                  f"Train loss: {running_loss/print_every:.3f}, "
                  f"Validation loss: {valid_loss/len(valid_loader):.3f}, "
                  f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
            
            running_loss = 0

            model.train()

print("Training Finished...")


#Model evaluation with test dataset

print('\n\nModel Evaluation...')
model.eval()
test_loss = 0
accuracy = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
                    
        log_ps = model.forward(inputs)
        batch_loss = criterion(log_ps, labels)
        test_loss += batch_loss.item()

        # Calculate validation accuracy
        ps = torch.exp(log_ps)
        top_p, top_cl = ps.topk(1, dim=1)
        compare = top_cl == labels.view(*top_cl.shape)
        accuracy += torch.mean(compare.type(torch.FloatTensor)).item()

print(f"Test accuracy: {accuracy/len(test_loader):.3f}")


## Saving Checkpoint
checkpoint = {
    'epochs': epochs,
    'learnrate': learnrate,
    'arch' : arch,
    'input_hidden': (25088,hidden),
    'hidden_output' : (hidden,102),
    'model_state_dict' : model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_to_idx': class_to_idx
}

torch.save(checkpoint, save_dir + '/checkpoint.pth')

print("Finished, your model is ready to predict!")