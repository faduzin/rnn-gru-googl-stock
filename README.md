# Google Stock Price Prediction using GRU RNN

This project leverages a Gated Recurrent Unit (GRU) based Recurrent Neural Network (RNN) to analyze and forecast Google's stock prices. The model is designed to capture the temporal dependencies in historical financial data, enabling us to uncover market trends and volatility patterns over time.

## Table of Contents

1. [Background](#background)
2. [What is a GRU RNN?](#what-is-a-gru-rnn)
3. [Tools I Used](#tools-i-used)
4. [The Process](#the-process)
5. [Data Overview and Analysis](#data-overview-and-analysis)
6. [The Analysis](#the-analysis)
7. [What I Learned](#what-i-learned)
8. [Skills Practiced](#skills-practiced)
9. [Conclusion](#conclusion)
10. [Contact](#contact)
11. [Repository Structure](#repository-structure)

## Background

This project was created as part of an exploratory analysis into financial time-series forecasting. By applying a GRU RNN model to Google's historical stock data, the goal is to predict future stock trends and provide insights into the market dynamics between 2006 and 2018.

## What is a GRU RNN?

### Recurrent Neural Networks (RNNs)
Recurrent Neural Networks (RNNs) are a class of neural networks particularly suited for processing sequential data. Unlike traditional feedforward neural networks, RNNs have loops within their architecture, allowing them to maintain a hidden state that persists across time steps. This "memory" enables RNNs to understand context and temporal dynamics, which is essential for tasks such as time-series forecasting, natural language processing, and speech recognition.

However, traditional RNNs often struggle with learning long-term dependencies due to issues like the vanishing gradient problem. As information is propagated back through many layers during training, gradients can diminish, making it hard for the network to learn from earlier inputs in a sequence.

### Gated Recurrent Units (GRUs)
GRUs are an advanced variant of RNNs designed to overcome some of these limitations. They incorporate gating mechanisms that regulate the flow of information through the network:

- **Update Gate:**  
  Controls how much of the previous hidden state should be carried forward to the current state. This gate allows the model to retain important long-term information while discarding less relevant details.

- **Reset Gate:**  
  Determines how to combine the new input with the previous hidden state. By selectively "resetting" the hidden state, the reset gate enables the network to forget irrelevant past information and better adapt to new inputs.

The GRU architecture simplifies the design compared to other gated models (like LSTMs) while still effectively capturing both short-term and long-term dependencies. This balance makes GRUs particularly effective for forecasting tasks, such as predicting stock prices where both recent trends and historical context are important.


## Tools I Used

- **Programming Language:** Python 3.12.9  
- **Deep Learning Frameworks:** TensorFlow/Keras (or PyTorch)  
- **Data Manipulation:** Pandas, NumPy  
- **Data Visualization:** Matplotlib, Seaborn  
- **Preprocessing & Metrics:** Scikit-learn  
- **IDE:** Visual Studio Code (VSCode)

## The Process

1. **Data Preprocessing:**  
   - Data cleaning and normalization  
   - Splitting the dataset into training and testing subsets

2. **Model Building:**  
   - Designing a GRU-based RNN architecture  
   - Compiling the model with an appropriate optimizer and loss function

3. **Training:**  
   - Feeding historical stock data into the model  
   - Tuning hyperparameters to optimize performance

4. **Evaluation:**  
   - Assessing the model using metrics such as Mean Squared Error (MSE)  
   - Visualizing the predicted versus actual stock prices

5. **Prediction:**  
   - Generating forecasts to analyze future stock trends

## Data Overview and Analysis

The dataset used in this project contains historical stock price data for Google (GOOGL) covering the period from January 1, 2006, to January 1, 2018. Below is an overview of the dataset and key insights from the exploratory data analysis (EDA):

### Dataset Overview

- **Date:** Trading date
- **Open:** Opening price of the stock on a given day
- **High:** Highest price recorded during the day
- **Low:** Lowest price recorded during the day
- **Close:** Closing price of the stock
- **Volume:** Number of shares traded during the day
- **Name:** Stock ticker symbol (GOOGL)

### Exploratory Data Analysis (EDA)

- **No Missing Values:**  
  All columns are fully populated, eliminating the need for handling missing data.

- **Total Trading Days:**  
  The dataset comprises 3,019 daily records.

- **Stock Price Ranges:**  
  - *Minimum Close Price:* $128.85  
  - *Maximum Close Price:* $1,085.09  
  - *Mean Close Price:* $428.04

- **Stock Volatility:**  
  The closing price has a standard deviation of $236.34, indicating significant fluctuations.

- **Trading Volume:**  
  Daily trading volumes vary considerably, with an average of 3.55 million shares and a peak of 41.18 million shares.

- **Trend Analysis (2006-2018):**  
  The overall trend shows consistent long-term growth in Google's stock price, punctuated by periods of high volatility. Notably, there is a sharp increase from 2015 to 2018, reflecting strong market confidence.

## The Analysis

This section describes the experimental analysis where four different GRU model configurations were trained and compared. The models differ by the number of GRU layers and the number of units per layer:

- **Single GRU Layer with 15 Units (single_15):**  
  This model has a single GRU layer containing 15 units. It achieved the closest performance to the real values.

- **Single GRU Layer with 50 Units (single_50):**  
  A single GRU layer with 50 units. Its performance was very close to the single_15 model, indicating that increasing the number of units slightly did not significantly alter the outcome in this case.

- **Double GRU Layers with 15 Units Each (double_15):**  
  This configuration stacks two GRU layers with 15 units each. While the predictions were a little less accurate compared to the single-layer models, the output lines were more stable, suggesting better smoothing of the time series.

- **Double GRU Layers with 50 Units Each (double_50):**  
  A deeper model with two GRU layers, each having 50 units. This model performed the worst, with predictions considerably far from the actual values.

### Loss Curves and Model Predictions

Below are placeholders for the images of the loss curves for each model configuration (in the following order):

1. **Single GRU Layer with 15 Units (single_15) Loss Curve:**  
   ![Loss Curve - Single_15](<assets\single_layer_15_gru_loss.png>)

2. **Single GRU Layer with 50 Units (single_50) Loss Curve:**  
   ![Loss Curve - Single_50](<assets\single_layer_50_gru_loss.png>)

3. **Double GRU Layers with 15 Units (double_15) Loss Curve:**  
   ![Loss Curve - Double_15](<assets\double_layer_15_gru_loss.png>)

4. **Double GRU Layers with 50 Units (double_50) Loss Curve:**  
   ![Loss Curve - Double_50](<assets\double_layer_50_gru_loss.png>)

Finally, the following image shows a comparative graph with the true values and the predictions from each model:

- **True Values vs. Predictions:**  
  ![Predictions Comparison](<assets\predictions-comparrison.png>)

### Summary of Findings

- **Best Performance:**  
  The **single_15** model was the closest to the real values, with the **single_50** model also delivering very similar results.

- **Stability vs. Accuracy:**  
  The **double_15** model, although slightly less accurate, produced more stable prediction lines.

- **Underperformance:**  
  The **double_50** model did not perform well; its predictions were significantly off compared to the other models.

- **Overall Insight:**  
  Despite experimenting with various architectures, none of the models managed to closely match the true stock prices, indicating that further tuning or more complex architectures might be necessary to improve forecasting accuracy.


## What I Learned

- The effectiveness of GRU RNNs in modeling complex time-series data.
- The critical role of thorough data preprocessing in enhancing model performance.
- Insights into the dynamics of stock market trends and volatility.
- The importance of tuning and validating deep learning models on financial datasets.

## Skills Practiced

- Deep Learning and Neural Network Design
- Time-Series Data Analysis and Forecasting
- Data Visualization and Statistical Analysis
- Python Programming and Model Deployment
- Research and Critical Analysis of Financial Data

## Conclusion

This project demonstrates the potential and challenges of using GRU RNNs for forecasting stock prices with historical data from Google. While the single-layer models (particularly single_15 and single_50) yielded predictions closest to the real values, none of the configurations perfectly captured the true dynamics of the market. These insights highlight the need for further exploration into model architectures and hyperparameter tuning for improved performance.


## Contact

For any questions, suggestions, or further discussion, please feel free to open an issue on the repository or contact me via my [GitHub profile](https://github.com/faduzin).

## Repository Structure

```bash
├── assets/ # Supplementary assets (e.g., images, charts) 
├── data/ # Raw and processed data files 
├── notebooks/ # Jupyter notebooks
├── src/ # Source code for model training and evaluation 
├── .gitignore # Git ignore file to exclude unnecessary files 
├── LICENSE # License information (MIT) 
├── README.md # Project documentation and overview 
└── requirements.txt # List of Python dependencies
```
This structure keeps the project organized and facilitates easy navigation through the code, data, and analysis resources.
