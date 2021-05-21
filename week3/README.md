# Week 3 pytorch assignment for group 11

Your code MUST be:

## The Network
![image](https://user-images.githubusercontent.com/13148910/119126865-3b017280-ba51-11eb-85b0-eed94144130b.png)
The Conv2d-9 layer outputs a softmax predicting the MNIST image. This is concatenated with the one hot representation of a random number between 0 and 9 and passed through Fully connected Layers, as shown in the model summary

### Loss Functions
For the training the MNIST dataset, we used the <b> NLL Loss </b> and for the adder network, used the <b> MSELoss </b>

## Data Generation
A random number is generated between 0 and 9 and added to the true label of the MNIST dataset. </br>
Hence the inputs for the dataset are: </br>
* random number 
* and the MNIST image </br>
The outputs to the network are: </br>
* and the MNIST true label 
* MNIST true label + random number </br>


```
class CombinedData(Dataset):
  def __init__(self, datasets):
    self.mnist_data = datasets
    # print(type(datasets))

  def __getitem__(self, index):
    x1, y1 = self.mnist_data[index]
    x2 = torch.randint(0, 9, (1,)).item()
    y2 = y1+x2
    # print(F.one_hot(torch.tensor(x2)))
    return x1, F.one_hot(torch.tensor(x2),10), y1, y2
    
  def __len__(self):
    return len(self.mnist_data)
```
## <b> Combining the two inputs </b>

Concatenated the pure softmax output from the MNIST network with the one_hot representation of the random number.

`in2 = torch.cat((out1, x2.view(-1, 10)), dim=1)`

## Evaluating Results
For every epoch, we calculate the number of correct predictions and the total loss from each batch. After training the entire epoch this is used to get the overall loss and accuracy of the trained model. These numbers give an idea of how the model is progressing. Ideally, the loss should decrease and the accuracy should increase with each epoch. 
There were instances when no change in these numbers were seen and that was due to some error in the model structure. 

## Final numbers
These are the final Accuracy numbers that were seen on the dataset.
```
Epoch 000: | Train Loss: 6.70477 | Train Acc Out1: 71.430| Train Acc Out2: 34.238
Epoch 001: | Train Loss: 0.47971 | Train Acc Out1: 97.013| Train Acc Out2: 48.327
Epoch 002: | Train Loss: 0.31413 | Train Acc Out1: 98.237| Train Acc Out2: 49.175
Epoch 003: | Train Loss: 0.21965 | Train Acc Out1: 98.825| Train Acc Out2: 49.998
Epoch 004: | Train Loss: 0.18871 | Train Acc Out1: 99.005| Train Acc Out2: 50.107
Epoch 005: | Train Loss: 0.16075 | Train Acc Out1: 99.185| Train Acc Out2: 50.060
Epoch 006: | Train Loss: 0.12929 | Train Acc Out1: 99.342| Train Acc Out2: 49.965
Epoch 007: | Train Loss: 0.10946 | Train Acc Out1: 99.477| Train Acc Out2: 49.973
Epoch 008: | Train Loss: 0.08979 | Train Acc Out1: 99.575| Train Acc Out2: 50.288
Epoch 009: | Train Loss: 0.07565 | Train Acc Out1: 99.655| Train Acc Out2: 50.242
Epoch 010: | Train Loss: 0.07393 | Train Acc Out1: 99.645| Train Acc Out2: 49.853
```
