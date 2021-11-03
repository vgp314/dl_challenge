import torch.nn as nn
import torch
from torchvision import transforms, utils


class MnistConvNet(nn.Module):
  '''
    Model for Detecting Numbers (Based on Mnist Dataset)

  '''
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,padding=(1,1)) 
    self.batch_norm1 = nn.BatchNorm2d(64) 
    self.pool1 = nn.MaxPool2d(kernel_size=2)

    self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=(1,1))
    self.batch_norm2 = nn.BatchNorm2d(32)
    self.pool2 = nn.MaxPool2d(kernel_size=2)
    
    self.fc1 = nn.Linear(7*7*32, 512)
    self.relu1 = nn.ReLU(inplace=True)

    self.fc2 = nn.Linear(512, 256)
    self.relu2 = nn.ReLU(inplace=True)

    self.fc3 = nn.Linear(256, 10)


  def forward(self, image):
    
    x = self.conv1(image)
    x = self.batch_norm1(x)
    x = self.pool1(x)
    
    x = self.conv2(x) 
    x = self.batch_norm2(x)
    x = self.pool2(x)


    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.relu1(x)
    
    x = self.fc2(x)
    x = self.relu2(x)
    
    x = self.fc3(x)
    
    return x


def get_generator_block(input_dim, output_dim):
    '''Base Sequencial Block for DigitCalculator'''

    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )



class DigitCalculator(nn.Module):
  '''Do all four calculations at same time (forward)'''

  def __init__(self, mnist_model):
    super().__init__()
    
    self.mnist_model = mnist_model
    
    self.sum = nn.Sequential(
            get_generator_block(20, 512),
            get_generator_block(512, 128),
            get_generator_block(128,19), # 19 possibilities (0 .. 18)
        )
        
    self.min = nn.Sequential(
            get_generator_block(20, 512),
            get_generator_block(512, 128),
            get_generator_block(128,19), # 19 possibilities (-9 .. 9)
        )
    self.mult = nn.Sequential(
            get_generator_block(20, 512),
            get_generator_block(512, 128),
            get_generator_block(128,82), # range of 82 possible values (0 .. 81)
        )
    
    self.div = nn.Sequential(
            get_generator_block(20, 512),
            get_generator_block(512, 128),
            get_generator_block(128,57), #range of 57 diferent values (9/0, 9/1 .... 1/9)
        )
    
    


  def forward(self, image1,image2):
    '''
      forward all four possible operations
    '''
    mnist_model = self.mnist_model.eval()
    for param in mnist_model.parameters():
        param.requires_grad = False
    embedding1 = torch.nn.functional.softmax(mnist_model(image1))
    embedding2 = torch.nn.functional.softmax(mnist_model(image2))

    embedding = torch.cat([embedding1,embedding2],axis=1)
    sum_result = self.sum(embedding)
    minus_result = self.min(embedding)
    mult_result = self.mult(embedding)
    div_result = self.div(embedding)
    
    return sum_result,minus_result,mult_result,div_result
