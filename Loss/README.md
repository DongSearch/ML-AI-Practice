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

