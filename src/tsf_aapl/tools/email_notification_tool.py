from crewai.tools import BaseTool
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import pandas as pd
from pathlib import Path

class NotifyTool(BaseTool):
    name: str = "notify_tool"
    description: str = "Send email with forecast result and model status"

    def _run(self, to_email: str = None) -> str:
        print(f"[DEBUG] Raw input to_email: {to_email}")
        if not to_email or "default_email" in to_email:
            to_email = "hieulv22416c@st.uel.edu.vn"
        try:
            # === Create email content ===
            ROOT = Path(__file__).resolve().parent.parent
            LOG_DIR = ROOT / "logs"

            predict_log_path = LOG_DIR / "predict_log.csv"
            online_log_path = LOG_DIR / "lstm_online_log.csv"
            eval_short_path = LOG_DIR / "lstm_eval_short_log.csv"
            pipeline_log_path = ROOT / "log.txt"

            df_pred = pd.read_csv(predict_log_path, usecols=[0, 1, 2], names=["Date", "Pred_Scaled", "Pred_Real"], header=0)
            latest_pred = df_pred.tail(1).iloc[0]
            date_pred = latest_pred["Date"]
            pred_real = float(latest_pred["Pred_Real"])

            df_online = pd.read_csv(online_log_path, on_bad_lines='skip')
            df_online.columns = ["Date", "True", "Predicted", "Error", "KL", "Drift"]
            latest_update = df_online.tail(1).iloc[0]
            true_val = float(latest_update["True"])
            pred_val = float(latest_update["Predicted"])
            error = float(latest_update["Error"])
            drift = str(latest_update["Drift"])

            df_eval = pd.read_csv(eval_short_path, header=None)
            latest_eval = df_eval.tail(1).iloc[0]
            mae, rmse, mape = latest_eval[1], latest_eval[2], latest_eval[3]

            pipeline_status = "SUCCESS"
            if pipeline_log_path.exists():
                with open(pipeline_log_path, "r", encoding="utf-8") as f:
                    log_content = f.read().lower()
                    if "error" in log_content or "fail" in log_content:
                        pipeline_status = "FAILURE"

            body = f"""Date of forecast: {date_pred}
Forecast price (AAPL): {pred_real:.2f} USD

Model update status:
- Actual price: {true_val:.2f} USD
- Predicted price: {pred_val:.2f} USD
- Error: {error:.2f} USD
- Drift: {drift}

Model evaluation (last 5 days):
- MAE: {mae:.4f}
- RMSE: {rmse:.4f}
- MAPE: {mape:.2f}%

Pipeline status: {pipeline_status}
"""

            # === Send email ===
            from_email = "maiyeuuel123@gmail.com"
            password = "mlehylonyuittgnj"

            subject = "[AAPL Forecast] Daily forecast report"

            msg = MIMEMultipart()
            msg["From"] = from_email
            msg["To"] = to_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(from_email, password)
                server.sendmail(from_email, to_email, msg.as_string())

            return f" Email has been sent to {to_email}!"

        except Exception as e:
            return f" Error while sending email: {e}"

if __name__ == "__main__":
    tool = NotifyTool()
    output = tool._run()
    print("[TOOL RESULT]")
    print(output)
