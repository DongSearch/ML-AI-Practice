# 📘 Loss Function

## 📌 Regression
Regression is a task where the model predict a continuous value
- stock price, age prediction

### 1️⃣ MAE(Mean Absolute error)
- average absolute difference between the predicted value and target value
- robust to outliers, easy to interpret || slower convergence, no differentiable at 0
- feature extraction 
  
### 2️⃣ MSE(Mean Squared Error)
- average squared error between the predicted value and target value
- smooth and fully differentiable, faster convergence || high sensitive to outlier 

### 3️⃣ Huber(combination between MAE and MSE)
- behave like MAE in large error, while do like MSE in small error
- set the boundary between them by adjusting delta
- robust to outlier(MAE), stable gradient(MSE) || setting delta is quite tricky

<img width="713" height="470" alt="image" src="https://github.com/user-attachments/assets/49b37169-866b-499d-9020-4c8df1655d76" />

## 📌 Classification
it is a task where the model predicts a category(discrete)
- image classification, recognition

### 1️⃣ Cross Entropy Loss (CE)
- measures the distance between true class distribution(one-hot) and predicted probability distribution(softmax)
- encourages the model to assign high probability to the correct class
- standard loss for multi-class classification
- strong gradient when prediction is wrong → fast learning || overconfident predictions, sensitive to label noise
- **Label smoothing!!
  modifies the target distribution by softening one-hot labels
  prevents the model from becoming overconfident
  improves generalization & calibration acts as a form of regularization || too much smoothing → underfitting, lower max confidence
  <img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/69721d8e-88c2-4dc5-90c1-0893b7afa2e9" />

  
### 2️⃣ BCE (Binary Cross Entropy)
- measures the distance between true binary label and predicted probability
- assumes independent Bernoulli distribution
- used for binary classification or multi-label classification
- strong gradient near wrong predictions || unstable when probabilities are exactly 0 or 1

### 3️⃣ NLL (Negative Log Likelihood)
- measures the negative log probability of the correct class
- expects log-probabilities as input (log_softmax)
- mathematically equivalent to Cross Entropy || requires manual log_softmax
