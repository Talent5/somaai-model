from scholarship_recommender import ScholarshipRecommender
import firebase_admin
from firebase_admin import credentials, firestore
import os
from apscheduler.schedulers.background import BackgroundScheduler  # Import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
from flask import Flask, jsonify  # Import Flask and jsonify

# Initialize Flask
app = Flask(__name__)

# Initialize Firebase (only once)
if not firebase_admin._apps:
    cred = credentials.Certificate(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the scholarship recommender
scholarship_data_path = './data/scholarships.csv'
recommender = ScholarshipRecommender(db, scholarship_data_path, model_dir='./models')

def run_recommendations():
    """Function to run the recommendation process."""
    print(f"\nStarting recommendation update at {datetime.now()}")
    recommender.process_users()
    print(f"Finished recommendation update at {datetime.now()}")

# Scheduling with APScheduler (use BackgroundScheduler)
scheduler = BackgroundScheduler()  
scheduler.add_job(run_recommendations, CronTrigger.from_crontab('0 0 * * *'))
scheduler.start()

# Example Flask endpoint
@app.route('/recommendations/<user_id>')
def get_recommendations(user_id):
    """
    Endpoint to get recommendations for a specific user.
    """
    # TODO: Implement logic to retrieve recommendations for user_id from 
    # the recommender or your data store.
    recommendations = recommender.get_recommendations_for_user(user_id)  # Example 
    return jsonify(recommendations) 

# Run Flask app if script is run directly
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)