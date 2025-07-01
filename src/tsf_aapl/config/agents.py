from crewai import Agent

from src.tsf_aapl.tools.data_raw_tool import DataCollectionTool
from src.tsf_aapl.tools.data_feature_tool import Feature_Tools
from src.tsf_aapl.tools.data_sentiment_tool import SentimentAnalysisTool
from src.tsf_aapl.tools.data_sentiment_tool import SentimentAnalysisTool
from src.tsf_aapl.tools.predict_today_tool import PredictTodayTool
from src.tsf_aapl.tools.predict_update_tool import PredictUpdateTool
from src.tsf_aapl.tools.evaluate_model_tool import EvaluateModelTool
from src.tsf_aapl.tools.email_notification_tool import NotifyTool
data_collector_agent = Agent(
    role="Stock Data & Sentiment Collector",
    goal=(
        "Fetch daily AAPL stock data from AlphaVantage and calculate sentiment scores "
        "from related news articles using NewsAPI and VADER. Ensure historical dataset is complete."
    ),
    backstory=(
        "You are a highly skilled financial data engineer and NLP analyst. Each day after the market closes, "
        "you retrieve the most recent AAPL stock prices and collect news articles about Apple from the previous 24 hours. "
        "You analyze them using VADER sentiment to enrich the dataset with financial sentiment indicators."
    ),
    allow_delegation=True,
    tools=[DataCollectionTool()],
    verbose=True,
    memory=True
)
sentiment_agent = Agent(
    role="Sentiment Analyzer",
    goal="Analyze news sentiment related to AAPL and enrich the dataset",
    backstory="You are an NLP expert that analyzes financial sentiment using VADER and NewsAPI.",
    tools=[SentimentAnalysisTool()],
    verbose=True,
    memory=True,
    allow_delegation=True
)
data_feature_agent = Agent(
    role = "Create more feature and preprocessing",
    goal = "Transform raw AAPL stock price data into clean, scaled features suitable for LSTM prediction." ,
    backstory = "You are a data engineer with deep experience in financial markets."\
    "Your job is to extract key technical indicators from raw stock data and prepare it"\
   " as high-quality input for deep learning models like LSTM."\
    "You ensure that data is clean, consistent, and rich with informative features.",
    allow_delegation=True,
    tools=[Feature_Tools()],
    verbose=True,
    memory=True
)
predict_update_agent = Agent(
    role="Model Drift Handler & Trainer",
    goal=(
        "Ensure the LSTM model remains accurate and up-to-date by detecting data drift using KL divergence "
        "and updating the model incrementally with the latest data."
    ),
    backstory=(
        "You are a vigilant AI engineer maintaining the health of a financial LSTM prediction model. "
        "Your job is to monitor for drift between recent samples, apply random projection if drift is found, "
        "and fine-tune the model on the most recent sample. You must always log the update, including date, true vs predicted value, "
        "error, KL divergence, and drift status into 'logs/lstm_online_log.csv'."
    ),
    tools=[PredictUpdateTool()],
    verbose=True,
    memory=True
)

predict_price_agent = Agent(
    role="Stock Price Forecaster",
    goal="Predict today's AAPL stock closing price based on processed features.",
    backstory="You are an AI financial analyst responsible for making daily AAPL stock price " \
    "predictions using an LSTM model and preprocessed inputs.",
    tools=[PredictTodayTool()],
    verbose=True,
    memory=True
)
evaluation_agent = Agent(
    role="Model Evaluator",
    goal="Assess the accuracy of the AAPL prediction model using recent logs.",
    backstory=(
        "You are a model evaluation expert. Your job is to regularly evaluate how well "
        "the LSTM model is predicting AAPL prices over time and record performance metrics."
    ),
    tools=[EvaluateModelTool()],
    verbose=True,
    memory=True
)
email_notifier_agent = Agent(
    role="Notification Dispatcher",
    goal="Send a daily summary email with AAPL forecast, model update status, and performance evaluation.",
    backstory=(
        "You are an automation agent responsible for notifying stakeholders about the current status of the AAPL prediction pipeline. "
        "Each day, you gather prediction logs, drift update results, evaluation metrics, and send them via email to the project team."
    ),
    tools=[NotifyTool()],
    verbose=True,
    memory=True,
    allow_delegation=False
)