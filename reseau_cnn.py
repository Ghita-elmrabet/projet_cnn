import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import numpy as np


#%% Hyper-parameters of the model
num_epochs = 15
batch_size = 4
learning_rate = 0.01

# Database path 
path_train='/content/drive/MyDrive/Database/imgs_train'
path_test= '/content/drive/MyDrive/Database/imgs_test' 

#%% Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

classes = ('0:presence', '1:absence')

train_dataset = torchvision.datasets.ImageFolder(root=path_train, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder(root=path_test, transform=transforms.ToTensor())

# Show 4 pairs of data
plt.figure(1)
for i in range(4):
    image, label = train_dataset[i]
    plt.subplot('14{}'.format(i))
    plt.imshow(transforms.ToPILImage()(image))
    plt.title('True label {}'.format(classes[label]))
plt.pause(0.1)

#%% Data loader 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)



#Define the pretrained model
model_resnet18 = models.resnet18(pretrained=True)
model_resnet18 = model_resnet18.to(device)

#Number of parameters of the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of parameters = {}'.format(count_parameters(model_resnet18)))

#Cross entropy loss
criterion = nn.CrossEntropyLoss()

#optimizer : Stochastic gradient descent   
optimizer = torch.optim.SGD(model_resnet18.parameters(), lr=learning_rate)

# Test the model
def validation(test_loader, model):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct, total)


# Train the model
num_training_data = len(train_dataset)
num_batch = len(train_dataset) 
training_loss_v = []
valid_acc_v = []
for epoch in range(num_epochs):
    loss_tot = 0
    model_resnet18.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model_resnet18(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad() #set gradients of all parameters to zero
        loss.backward()
        optimizer.step()
        
        loss_tot += loss.item()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, num_batch, loss.item()/len(labels)))
            
    (correct, total) = validation(test_loader, model_resnet18)
    print ('Epoch [{}/{}], Training Loss: {:.4f}, Valid Acc: {} %'
           .format(epoch+1, num_epochs, loss_tot/num_training_data, 100 * correct / total))
    training_loss_v.append(loss_tot/num_training_data)
    valid_acc_v.append(correct / total)

# Save the model checkpoint
torch.save(model_resnet18.state_dict(), 'model.ckpt')

#%% plot results
# The training loss
plt.figure(2)
plt.clf()
plt.plot(np.array(training_loss_v),'r',label='Training loss')
plt.legend()

# Validation accuracy
plt.figure(3)
plt.clf()
plt.plot(np.array(valid_acc_v),'g',label='Validation accuracy')
plt.legend()
