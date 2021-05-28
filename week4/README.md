# Problem Statement:

1. **Part 1**:  Back propagation for a 2 layer 4 neuron model on an Excel sheet.
 
 #### Refer to link below for details on neural network calculations. 
 

https://github.com/MittalNeha/Extensive_Vision_AI6/tree/main/week4/Back_Propagation

<img src="https://github.com/MittalNeha/Extensive_Vision_AI6/blob/main/week4/images/image-20210528070301014.png?raw=true" style="zoom: 80%;" />

The equations for backpropagation of this network are derived [here](https://docs.google.com/document/d/e/2PACX-1vQObVYuAyGTwL0GUH90K_917ICkiwYK_zi8FdhCiLpoW4eGAwoQEfssxqOSB0134g5E7eFjid9PtITv/pub)








2. **Part 2**:  

Squeeze and Expand

|No.	|Param	|Model	|Accuracy (10ep)	|Kernel progression	|Remarks
|--|:-------------:|------:|----------|-------------|------|
1	|6.38M||99.26|    | Original/base.|
2	|0.92M|Net|99.05|12->24->48->96->192->384   |Tops at 99.17|
3	|221K|Net2|98.89|32->64->32->MP->32->64->128->32->MP->32->64->128   |Used 1x1 to reduce kernels|
4	|94K|Net3|98.93|24->48->64->24->MP->24->48->64->24->MP->48->10  |Starts at 95.75%|
5 |64K|Net4|98.93|In Net3, changed 24->48->64 to 24->36->48	  |Epoch1 10.28%, Epoch2 94.59%|
6	|39K|Net5|98.72|In Net4, changed 24->36->48 to 24->24->36   |Epoch1 90.39%, Epoch2 97.5%. This model saturates quickly.|
7					
