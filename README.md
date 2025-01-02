# Deep-Learning-Approaches-for-Emotion-Detection
## Overview
This project focuses on exploring deep learning approaches for emotion detection in Twitter data. It evaluates the effectiveness of various models—BiLSTM, CNN, GRU, ANN, and RNN—in accurately classifying emotions from a large-scale dataset of 393,822 annotated tweets.

## Key Features
- **Model Comparison:** Investigates and compares the performance of Bidirectional LSTM (BiLSTM), Convolutional Neural Networks (CNN), Gated Recurrent Units (GRU), Artificial Neural Networks (ANN), and Recurrent Neural Networks (RNN) for emotion detection tasks.
- **Dataset and Labels:** Utilizes a diverse dataset of 393,822 tweets annotated with six distinct emotions: anger, fear, joy, love, sadness, and surprise. This dataset enables robust training and evaluation of emotion classification models. 
- **Performance Evaluation:** Measures model performance using metrics such as accuracy, precision, recall, and F1-score to assess their ability to detect emotions accurately across different models.
- **Real-Time Application:** Implements a user-friendly interface using Streamlit to demonstrate practical application, allowing real-time emotion classification from text input.

## Dataset
The emotion detection dataset consists of 393,822 English tweets labeled with six emotions: anger, fear, joy, love, sadness, and surprise. This large and varied dataset facilitates comprehensive model training and evaluation across different emotional states.
- **Anger:** 17.5%
- **Fear:** 15.2%
- **Joy:** 21.8%
- **Love:** 14.6%
- **Sadness:** 19.1%
- **Surprise:** 11.8%
This distribution ensures that models are trained on a balanced representation of emotions, enhancing their ability to generalize to real-world applications effectively.


## Colab Notebook (Click to View)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XI6mBPTJeitoQjk7qt9XLUV78LvryGSL?usp=sharing)
## Dataset Overview
**Dataset Link:** [Emotions Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data)
![image](https://github.com/user-attachments/assets/87c7f734-0936-47fd-95ba-a18a044a22f4)
![image](https://github.com/user-attachments/assets/576d4d50-c542-436c-991b-15105a8cc18b)
![newplot](https://github.com/user-attachments/assets/21dcf6b9-5181-4ba3-95f3-1975f8227976)



## Algorithm Implementation
The algorithm implementation for emotion classification in this study follows a systematic approach encompassing dataset preparation, preprocessing, data splitting, model training, evaluation, and prediction. Leveraging various deep learning architectures, the implementation provides profound insights into the effectiveness of detecting emotions from textual data.
Dataset: The study utilizes a dataset comprising 393,822 English Twitter messages annotated with six distinct emotions: sadness, joy, love, anger, fear, and surprise. This dataset is well-suited for sentiment analysis, emotion classification, and real-time social media monitoring, offering a comprehensive representation of emotional expressions crucial for understanding public sentiment.
Preprocessing: Before model training, preprocessing steps involve removing duplicates, handling missing values, tokenizing text into individual tokens, and mapping words to indices. Sequences are standardized by padding or truncating to a maximum length of 128 tokens. Labels are converted into categorical format, ensuring uniformity and enhancing model performance.
Dataset Splitting: The dataset is split into training (80%) and testing (20%) sets using stratified sampling to maintain balanced label distribution across both sets. This ensures robust model training and evaluation, with the model tested on unseen data to assess generalization capabilities.
Model Training: Five deep learning models—BiLSTM, CNN, GRU, ANN, and RNN—are evaluated for emotion detection. BiLSTM captures contextual nuances in sequential data, CNN identifies local text patterns effectively, GRU handles long-term dependencies efficiently, ANN serves as a benchmark, and RNN processes sequential data but with limitations in long-term dependency handling.
Model Evaluation: Each model's performance is assessed using accuracy, precision, recall, and F1-score metrics. This evaluation provides insights into their effectiveness in detecting and classifying emotions, linking results to practical applications across healthcare, business, and public opinion analysis.
Prediction: The trained models predict emotion labels for new text inputs based on learned patterns, with prediction accuracy indicating model performance on unseen data.
This structured approach ensures a thorough analysis of emotional expressions in text data, laying the groundwork for advancing emotion detection capabilities across diverse applications.

## Result
### DEEP LEARNING ALGORITHMS
| Algorithm               | Accuracy | Precision | Recall | F1-Score |
|-------------------------|----------|-----------|--------|----------|
| BiLSTM                  | 0.99     | 0.94      | 0.95   | 0.94     |
| CNN                     | 0.96     | 0.95      | 0.92   | 0.93     |
| GRU                     | 0.96     | 0.94      | 0.96   | 0.96     |
| ANN                     | 0.75     | 0.67      | 0.75   | 0.70     |
| RNN                     | 0.34     | 0.30      | 0.30   | 0.25     |



#### BiLSTM
![image](https://github.com/user-attachments/assets/62b4a556-ae99-4585-91cb-95fd7927008b)
#### CNN
![image](https://github.com/user-attachments/assets/0831b4fc-ebc1-413a-b749-d7c7b0a6458e)
#### GRU
![image](https://github.com/user-attachments/assets/bc5c8146-ea69-450b-9222-478a6d758bd3)
#### ANN
![image](https://github.com/user-attachments/assets/6d50d7e4-82ef-4e75-b83a-4397caf1c13f)
#### RNN   
![image](https://github.com/user-attachments/assets/049d737c-9e7a-47f4-952c-c51647405215)

