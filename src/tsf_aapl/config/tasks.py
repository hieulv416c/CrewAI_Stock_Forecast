from crewai import Task
from src.tsf_aapl.tools.data_raw_tool import DataCollectionTool
from src.tsf_aapl.tools.data_feature_tool import Feature_Tools
from src.tsf_aapl.tools.data_sentiment_tool import SentimentAnalysisTool
def build_data_collection_task(agent):
    return Task(
        description=(
            "Your task is to fetch the latest AAPL stock price using AlphaVantage.\n"
            "Use ONLY this exact tool call:\n"
            "```\n"
            "Thought: I will fetch new AAPL data using the data_collection_tool.\n"
            "Action: data_collection_tool\n"
            "Action Input: {}\n"
            "```\n"
            "If no new data is found, do NOT try to use the tool again.\n"
            "Instead, return:\n"
            "```\n"
            "Thought: No new data to add. Work is done.\n"
            "Final Answer: Data is already up-to-date.\n"
            "```"
        ),
        expected_output=(
            "A new row for the most recent trading day is appended to 'aapl_raw.csv' if not present yet. "
            "File remains up to date and sentiment columns are preserved."
        ),
        agent=agent,
        async_execution=False
    )

def build_sentiment_task(agent):
    return Task(
        description=(
            "Use the `data_sentiment_tool` to update the `aapl_raw.csv` file with sentiment scores for each trading day.\n"
            "You should:\n"
            "- Query NewsAPI for AAPL news within the past 24h of each date.\n"
            "- Apply VADER sentiment analysis.\n"
            "- Write back 4 new columns: sentiment_positive, sentiment_negative, sentiment_neutral, sentiment_score.\n"
            "- Skip rows that already have sentiment_score.\n\n"
            " Once finished, you MUST delegate the next task to your coworker `Create more feature and preprocessing`.\n"
            "Use the following format:\n"
            "```\n"
            "Thought: Sentiment is updated. I will now delegate the next task.\n"
            "Action: Delegate work to coworker\n"
            "Action Input: {\n"
            "  \"coworker\": \"Create more feature and preprocessing\",\n"
            "  \"task\": \"Please process 'aapl_raw.csv' to compute SMA, EMA, RSI, MACD indicators, normalize features, create sliding windows, and save them for LSTM.\",\n"
            "  \"context\": \"The sentiment columns are now available. Next step is feature engineering.\"\n"
            "}\n"
            "```"
        ),
        expected_output=(
            "Sentiment columns are successfully added to all applicable rows in aapl_raw.csv, "
            "and the next task has been delegated to the feature engineer."
        ),
        agent=agent,
        async_execution=False
    )

def build_data_feature_task(agent):
    return Task(
        description=(
            "Regardless of whether there is new data or not, you MUST regenerate all financial features from the latest 'aapl_raw.csv' using the `data_feature_tool`.\n\n"
            "Steps:\n"
            "- Compute SMA, EMA, RSI, MACD\n"
            "- Normalize with MinMaxScaler\n"
            "- Save: X_lstm.npy, y_lstm.npy, date_y_lstm.npy, X_today.npy\n\n"
            "Once done, you MUST delegate the next task to `Model Drift Handler & Trainer` like this:\n"
            "```\n"
            "Thought: Features are ready. I will now delegate to the model updater.\n"
            "Action: Delegate work to coworker\n"
            "Action Input: {\n"
            "  \"coworker\": \"Model Drift Handler & Trainer\",\n"
            "  \"task\": \"Please update the LSTM model using latest sample and log drift detection results.\",\n"
            "  \"context\": \"Features are saved and ready. Time to update model.\"\n"
            "}\n"
            "```"
        ),
        expected_output="Features regenerated and task delegated to drift handler.",
        agent=agent,
        async_execution=False
    )


def build_predict_update_task(agent):
    return Task(
        description=(
            "You MUST use `predict_update_tool` to update the LSTM model with the latest data.\n"
            "Just call the tool like this:\n"
            "```\n"
            "Thought: Updating model\n"
            "Action: predict_update_tool\n"
            "Action Input: {}\n"
            "```"
        ),
        expected_output=(
            "Final Answer: Model updated with latest sample. Drift status and error logged to `logs/lstm_online_log.csv`."
        ),
        agent=agent,
        async_execution=False
    )




def build_prediction_task(agent):
    return Task(
        description=(
            "Use the PredictTodayTool to forecast today's AAPL stock closing price based on the most recent processed input data.\n\n"
            "Steps:\n"
            "- Load input features from `Input/X_lstm.npy`\n"
            "- Load the trained LSTM model from `models/model_today.pth`\n"
            "- Load scalers from `Input/feature_scalers.pkl`\n"
            "- Run prediction for the latest day in the data\n"
            "- Apply inverse transform (if target scaler exists) to obtain real USD price\n"
            "- Log the prediction result to `logs/predict_log.csv` in format: date, scaled, unscaled\n\n"
            "Make sure to return both scaled and unscaled (USD) predictions clearly."
        ),
        expected_output=(
            "Final Answer: The predicted AAPL stock price for today. Include both the scaled and unscaled values (if available), "
            "and confirm that the result has been logged to `logs/predict_log.csv`."
        ),
        agent=agent
    )
def build_evaluation_task(agent):
    return Task(
        description=(
            "Use `evaluate_model_tool` to evaluate the performance of the AAPL stock price prediction model.\n"
            "You MUST respond in the format:\n"
            "```\n"
            "Thought: Let's evaluate the model performance\n"
            "Action: evaluate_model_tool\n"
            "Action Input: {}\n"
            "```"
        ),
        expected_output=(
            "Final Answer: Evaluation completed. MAE, RMSE, and MAPE logged for both 5 and 30-day windows if available."
        ),
        agent=agent,
        async_execution=False
    )

def build_email_notification_task(agent):
    return Task(
        description=(
            "Use `notify_tool` to send an email summarizing the daily AAPL forecast results.\n\n"
            "Steps:\n"
            "- Load data from: `predict_log.csv`, `lstm_online_log.csv`, `lstm_eval_short_log.csv`, and `log.txt`\n"
            "- Include in email:\n"
            "    + Forecast date and predicted price\n"
            "    + Actual vs predicted price, error, drift detection result\n"
            "    + Model evaluation scores (MAE, RMSE, MAPE)\n"
            "    + Pipeline status (SUCCESS/FAILURE)\n"
            "- Send to default email address or one specified in arguments\n\n"
            "Make sure the email is successfully delivered and all logs are correctly referenced."
        ),
        expected_output="Final Answer: Email with forecast summary and model status has been sent successfully.",
        agent=agent,
        async_execution=False
    )
