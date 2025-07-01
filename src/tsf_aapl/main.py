import os
from datetime import datetime
from dotenv import load_dotenv
from crew import crew

# Setup log path
log_dir = os.path.dirname(__file__)
log_path = os.path.join(log_dir, "log.txt")
os.makedirs(log_dir, exist_ok=True)

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")  # Also print to terminal

# Start log
log(" Task started")
log("-" * 50)

try:
    # Load environment
    load_dotenv()
    model = os.getenv("MODEL")
    provider = os.getenv("LITELLM_PROVIDER")
    log(f"[INFO] MODEL = {model}")
    log(f"[INFO] PROVIDER = {provider}")

    # Run crew
    log("Running CrewAI pipeline...")
    result = crew.kickoff()
    log(" Crew finished execution successfully.")
    log("---- Output Start ----")
    log(result)
    log("---- Output End ----")

except Exception as e:
    log(f" ERROR: {str(e)}")

finally:
    log("-" * 50)
    log(" Task ended.\n")


