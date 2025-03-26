# Sentiment Analysis on US Airline Reviews  
## Overview  
This project performs sentiment analysis on US airline reviews using a deep learning model with LSTM (Long Short-Term Memory) networks. The dataset used is `Tweets.csv`, which contains customer feedback labeled with sentiment categories.  

## Features  
- Preprocessing and tokenization of text data  
- Sentiment factorization for binary classification (positive/negative)  
- LSTM-based deep learning model for sentiment prediction  
- Accuracy and loss visualization  
- Custom function for real-time sentiment prediction  

## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/repository-name.git
   cd repository-name
   ```  
2. Install dependencies:  
   ```bash
   pip install pandas matplotlib tensorflow
   ```  

## Usage  
1. Place `Tweets.csv` in the project directory.  
2. Run the script:  
   ```bash
   python Sentiment Analysis.py
   ```  
3. To predict sentiment for custom text, modify the `test_sentence1` and `test_sentence2` in the script and execute it.  

## Output  
- **Plots**: Accuracy and loss graphs are saved as `Accuracy plot.jpg` and `Loss plot.jpg`.  
- **Sentiment Prediction**: The function `predict_sentiment(text)` predicts the sentiment of any given input sentence.  

## Dependencies  
- Python  
- TensorFlow  
- Pandas  
- Matplotlib  

## License  
This project is open-source under the MIT License.  
