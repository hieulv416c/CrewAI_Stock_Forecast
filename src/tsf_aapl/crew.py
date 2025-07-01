from crewai import Crew, Process
from config.agents import data_collector_agent, data_feature_agent, sentiment_agent
from config.agents import predict_price_agent, predict_update_agent, evaluation_agent, email_notifier_agent
from config.tasks import build_data_collection_task, build_data_feature_task, build_sentiment_task
from config.tasks import build_prediction_task, build_predict_update_task, build_evaluation_task, build_email_notification_task

data_collection_task = build_data_collection_task(data_collector_agent)
data_sentiment_task= build_sentiment_task(sentiment_agent)
data_feature_task = build_data_feature_task(data_feature_agent)
predict_update_task = build_predict_update_task(predict_update_agent)
predict_price_task = build_prediction_task(predict_price_agent)
evaluate_task = build_evaluation_task(evaluation_agent)
email_notification_task = build_email_notification_task(email_notifier_agent)
crew = Crew(
    agents=[data_collector_agent, sentiment_agent , data_feature_agent, predict_update_agent, predict_price_agent, evaluation_agent, email_notifier_agent],
    tasks=[data_collection_task, data_sentiment_task, data_feature_task, predict_update_task, predict_price_task, evaluate_task, email_notification_task],
    process=Process.sequential
)

