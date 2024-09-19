import os
import sys
import logging
from flask import Flask
from recommender import ScholarshipRecommender
import threading
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the recommender
db_path = 'firebase-credentials.json'
scholarship_data_path = 'data/scholarships.csv'

try:
    recommender = ScholarshipRecommender(db_path, scholarship_data_path)
    logger.info("ScholarshipRecommender initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ScholarshipRecommender: {str(e)}")
    sys.exit(1)

def update_recommendations_continuously():
    while True:
        try:
            logger.info("Updating recommendations...")
            recommender.process_users()
            logger.info("Recommendations updated. Waiting for 60 seconds...")
            time.sleep(60)  # Wait for 60 seconds
        except Exception as e:
            logger.error(f"Error in update_recommendations_continuously: {str(e)}")
            time.sleep(60)  # Wait before retrying

@app.route('/')
def home():
    return "Scholarship Recommender is running!"

@app.route('/update_recommendations')
def update_recommendations():
    try:
        recommender.process_users()
        return "Recommendations updated successfully!"
    except Exception as e:
        logger.error(f"Error in update_recommendations: {str(e)}")
        return f"Error updating recommendations: {str(e)}", 500

if __name__ == '__main__':
    # Start the continuous update process in a separate thread
    update_thread = threading.Thread(target=update_recommendations_continuously)
    update_thread.start()

    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)