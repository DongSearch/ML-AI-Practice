# 📘 CNN(Convolutional Neural Network)
![1_O06nY1U7zoP4vE5AZEnxKA](https://github.com/user-attachments/assets/7238bdaa-65ad-458a-84a7-b16ab8a4362e)
from https://medium.com/@Suraj_Yadav/in-depth-knowledge-of-convolutional-neural-networks-b4bfff8145ab

## 📌 Baisc Concept
o it consists basically three components : Convolutional layer, Pooling, Fully-Connected Layer
- CL : basically, the kernel with specific size,stride(how many pixels it moves each step), zero-pedding(how many pixels are added addictionally) moves over image map and extracts main features 
- pooling : it is similar to CL, but main difference is its main purpose is to reduce freature map size. it has two types in large scale. max pooling extracts the maximum value to capture the most stand-out features and average pooling uses for preserving overall information while wanting to reduce size
- FCN : FCN take the flattend feature maps and do classification or regression by combining all extracted features 

### 1️⃣ where it is used? what is the advantage compared to normal Linear Layer
- it is used to extract features and analyse them
- preserve spatial relationships : while a FC layer ignores the spaital information, it still keeps the relative positions between pixels, which allow the model to rcognize spatial patterns
- fewer parameters : same filter can be used over the whole image set(parameter sharing), which leads to save memory
-  translation invariance : since the same filter is applied to the whole image samely, small variance of object movement can be still detected effectively
  
### 2️⃣ why should we use porper zero-pedding?
- edge information can be less considered(underrepresented) compared to the center beacause of its structure, so by adding zero-padding, this issue can be alleviated as well as permetting to preserve the spatial size of the feature map
