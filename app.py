import os
import pandas as pd
import logging
from flask import Flask, jsonify
from scholarship_recommender import ScholarshipRecommender
from apscheduler.schedulers.background import BackgroundScheduler
from flask import abort

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Initialize the ScholarshipRecommender
db_path = 'firebase-credentials.json'
if not db_path:
    logging.error("FIREBASE_CREDENTIALS environment variable not set.")
    raise EnvironmentError("FIREBASE_CREDENTIALS environment variable is required")

try:
    scholarship_data_path = pd.read_csv('./data/scholarships.csv')
except FileNotFoundError:
    logging.error("Scholarship data file not found.")
    raise FileNotFoundError("Scholarship data file is required")

recommender = ScholarshipRecommender(db_path, scholarship_data_path)

def update_recommendations():
    try:
        recommender.process_users()
        logging.info("Recommendations updated successfully.")
    except Exception as e:
        logging.error(f"Error updating recommendations: {e}")

# Set up scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_recommendations, trigger="interval", hours=24)
scheduler.start()

@app.route('/')
def home():
    return "Scholarship Recommender is running!"

@app.route('/update', methods=['POST'])
def manual_update():
    try:
        recommender.process_users()
        return jsonify({"status": "success", "message": "Recommendations updated"}), 200
    except Exception as e:
        logging.error(f"Error in manual update: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/test/<user_id>')
def test_user(user_id):
    try:
        recommender.test_single_user(user_id)
        return jsonify({"status": "success", "message": f"Tested recommendations for user {user_id}"}), 200
    except Exception as e:
        logging.error(f"Error testing user {user_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
