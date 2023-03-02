# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:12:11 2023

@author: Bram
"""

import argparse
import cv2
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torchvision.transforms as trans
import numpy as np
import matplotlib.pyplot as plt

# Initialize argument parser for easy
parser = argparse.ArgumentParser("MiniImagenet")

# Model arguments
parser.add_argument("--num_filters", type=int, default=16)

# Training arguments
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate hyperparameter for the AdamW optimizer")
# parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay hyperparameter for the AdamW optimizer")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of runs over all of the training data")
parser.add_argument("--batch_size", type=int, default=1, help="The batchsize that is fed to the model")
parser.add_argument("--num_workers", type=int, default=1, help="The number of processes that work for the dataloader")
# parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cpu:0"], help="Specify the device on which you run the code, gpu(cuda) or cpu")
parser.add_argument("--benchmark_cudnn", type=bool, default=False)

args = parser.parse_args()

print('Pytorch Version:' , torch.__version__)

data_transforms = {
    'train': trans.Compose([
        trans.RandomResizedCrop(224),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        # when do the transform, image pixels are compressed 
        # from (0,255) to (0,1) then we do the normalization
        trans.Normalize([0.485, 0.456, 0.406], # mean of RGB
                        [0.229, 0.224, 0.225]) # std of RGB
    ]),
    'val': trans.Compose([
        trans.Resize(256),
        trans.CenterCrop(224),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=data_transforms['train'])

val_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=data_transforms['val'])

class_names = train_data.classes
dataset_sizes = {'train': len(train_data), 'val': len(val_data)}

dataloaders = {
    'train': Data.DataLoader(
        dataset=train_data,
        batch_size=1,
        shuffle=True,
#         num_workers=4
    ),

    'val': Data.DataLoader(
        dataset=val_data,
        batch_size=1,
        shuffle=True,
#         num_workers=4
    )
}

# Set up the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
if(args.benchmark_cudnn):
    torch.backends.cudnn.benchmark = True

# denormalize and show an image 
def imshow(image, title=None):
    image = image.numpy().transpose((1, 2, 0)) 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    if title:
        plt.title(title)
        
print(dataloaders['train'])

# get a batch of training data
images, classes = next(iter(dataloaders['train']))

# make a grid from batch
images = torchvision.utils.make_grid(images)

imshow(images, title=[class_names[x] for x in classes])


vgg = torchvision.models.vgg16(pretrained=True)
vgg = vgg.to(device)
print('VGG16 Architecture:\n', vgg)

# reconstruct VGG16, i.e. remove the classifier and replace it with GAP
class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True) 
        self.conv = nn.Sequential(
            self.vgg.features, 
            self.vgg.avgpool 
        )
         
        # self.fc = nn.Linear(512, num_of_class)
        # as we use ImageNet, num_of_class=1000
        self.fc = nn.Linear(512, 1000)

    
    def forward(self,x):    
        x = self.conv(x) # -> (512, 7, 7)
        
        
        # we use GAP to replace the fc layer, therefore we need to
        # convert (512,7,7) to (512, 7x7)(i.e. each group contains 7x7=49 values), 
        # then convert (512, 7x7) to (512, 1x1) by mean(1)(i.e. average 49 values in each group), 
        # and finally convert (512, 1) to (1, 512) 
        x = x.view(512,7*7).mean(1).view(1,-1) # -> (1, 512)
        
        # FW^T = S
        # where F is the averaged feature maps, which is of shape (1,512)
        # W is the weights for fc layer, which is of shape (1000, 512)
        # S is the scores, which is of shape (1, 1000)
        x = self.fc(x) # -> (1, 1000)
        return x 


def train_model(model, loss_fn, optimizer, scheduler, num_epochs=5):
    """
    net: the model to be trained
    loss_fn: loss function
    scheduler: torch.optim.lr_scheduler
    """
    
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)
        
        step_loss = 0.0
        epoch_accuracy = 0.0
        
        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # set to training mode
            else:
                model.eval() # set to evaluate mode
            
            step_loss = 0.0
            step_corrects = 0
            
            for step, (images, labels) in enumerate(dataloaders[phase]): 
                images = images.to(device)
                labels = labels.to(device)
               
                # forward pass, compute loss and make predictions
                outputs = model(images) 
                preds = torch.max(outputs, 1)[1]
                loss = loss_fn(outputs, labels)
                
                # backward pass and update weights if in training phase
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # compute step loss and step corrects
                step_loss += loss.item() * images.size(0) # loss.item() extracts the loss's value
                step_corrects += torch.sum(preds == labels.data)
            
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = step_loss / dataset_sizes[phase]
            epoch_accuracy = step_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_accuracy))

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_accuracy)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_accuracy)

            # deep copy the model
            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
            
        print()
        
    print('Best Validation Accuracy: {:4f}'.format(best_accuracy))

    # draw the loss history and accuracy history
    x = np.arange(num_epochs)
    plt.subplot(221)
    plt.plot(x, train_loss, c='red', label='train loss')
    plt.plot(x, val_loss, c='blue', label='val loss')
    plt.legend(loc='best')

    plt.subplot(222)
    plt.plot(x, train_acc, c='red', label='train acc')
    plt.plot(x, val_acc, c='blue', label='val acc')
    plt.legend(loc='best')

    plt.show()
    
    # load best model weights
    model.load_state_dict(best_model_weights)
    return model

# generic function to display predictions for a few images
def visualize_model(model, num_images=6):
    was_training = model.training # if true, the model is in training mode otherwise in evaluate mode
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    
    for step, (images, labels) in enumerate(dataloaders['val']): 
        outputs = model(images)
        preds = torch.max(outputs, 1)[1]
        
        for i in range(images.size(0)):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[i]]))
            imshow(images.cpu().data[i])
            
            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    
    model.train(mode=was_training)
    
    
model = VGG()

trainable_parameters = []
for name, p in model.named_parameters():
    if "fc" in name:
        trainable_parameters.append(p)

loss_fn = torch.nn.CrossEntropyLoss()

# all parameters are being optimized
optimizer = torch.optim.SGD(trainable_parameters, lr=0.001, momentum=0.9)

# decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(model, loss_fn, optimizer, exp_lr_scheduler, num_epochs=25)


visualize_model(model)


params = list(model.fc.parameters())
weight = np.squeeze(params[0].data.numpy())
print('weight.shape', weight.shape)

image, label = next(iter(dataloaders['val']))

model.eval()
scores = model(image) # get the raw scores
probs = F.softmax(scores, dim=1).data.squeeze() # use softmax to generate the probability distribution for the scores
probs, idx = probs.sort(0, True) # sort the probability distribution in descending order, and idx[0] is the predicted class
print('sum of probabilities: %.0f'%torch.sum(probs).numpy())
print('true class: ', class_names[label])
print('predicated class: ', class_names[idx[0].numpy()])

def return_CAM(feature_conv, weight, class_idx):
    """
    return_CAM generates the CAMs and up-sample it to 224x224
    arguments:
    feature_conv: the feature maps of the last convolutional layer
    weight: the weights that have been extracted from the trained parameters
    class_idx: the label of the class which has the highest probability
    """
    size_upsample = (224, 224)
    
    # we only consider one input image at a time, therefore in the case of 
    # VGG16, the shape is (1, 512, 7, 7)
    bz, nc, h, w = feature_conv.shape 
    output_cam = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w))# -> (512, 49)
        cam = np.matmul(weight[idx], beforeDot) # -> (1, 512) x (512, 49) = (1, 49)
        cam = cam.reshape(h, w) # -> (7 ,7)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


feature_maps = model.conv(image) # get the feature maps of the last convolutional layer
print('feature_maps.shape: ', feature_maps.detach().numpy().shape)

CAMs = return_CAM(feature_maps.detach().numpy(), weight, [idx.numpy()[0]]) # generate the CAM for the input image
heatmap = cv2.applyColorMap(CAMs[0], cv2.COLORMAP_JET)


print('original image shape: ', image.reshape((3, 224, 224)).numpy().transpose((1,2,0)).shape)
print('heatmap.shape:', heatmap.shape)
image = image.reshape((3, 224, 224)).numpy().transpose((1, 2, 0)) 
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image = image * std + mean
image = np.clip(image, 0, 1)

plt.imshow(image)
plt.show()
plt.imshow(heatmap)
plt.show()

result = 0.5 * heatmap + 0.5 * image
cv2.imwrite('cam.png', result)