import os
import logging
from flask import Flask, jsonify, request
from scholarship_recommender import ScholarshipRecommender
import firebase_admin
from firebase_admin import credentials
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Firebase
def init_firebase():
    try:
        if not firebase_admin._apps:
            firebase_config = {
                'type': os.environ['FIREBASE_TYPE'],
                'project_id': os.environ['FIREBASE_PROJECT_ID'],
                'private_key_id': os.environ['FIREBASE_PRIVATE_KEY_ID'],
                'private_key': os.environ['FIREBASE_PRIVATE_KEY'].replace('\\n', '\n'),
                'client_email': os.environ['FIREBASE_CLIENT_EMAIL'],
                'client_id': os.environ['FIREBASE_CLIENT_ID'],
                'auth_uri': os.environ['FIREBASE_AUTH_URI'],
                'token_uri': os.environ['FIREBASE_TOKEN_URI'],
                'auth_provider_x509_cert_url': os.environ['FIREBASE_AUTH_PROVIDER_X509_CERT_URL'],
                'client_x509_cert_url': os.environ['FIREBASE_CLIENT_X509_CERT_URL'],
            }
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {str(e)}")
        raise

# Initialize the recommender
def init_recommender():
    try:
        scholarship_data_path = './data/scholarships.csv'
        db_path = firebase_config  # Make sure this is defined
        recommender = ScholarshipRecommender(
            db_path=db_path,
            scholarship_data_path=scholarship_data_path,
            model_dir='./models'
        )
        logger.info("ScholarshipRecommender initialized successfully")
        return recommender
    except Exception as e:
        logger.error(f"Failed to initialize ScholarshipRecommender: {str(e)}")
        raise

# Set up scheduler
def init_scheduler(recommender):
    try:
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            func=recommender.process_users,
            trigger=IntervalTrigger(hours=24),
            id='scholarship_recommendation_job',
            name='Generate scholarship recommendations periodically',
            replace_existing=True
        )
        scheduler.start()
        logger.info("Scheduler started successfully")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {str(e)}")
        raise

# Initialize components
init_firebase()
recommender = init_recommender()
init_scheduler(recommender)

@app.route('/')
def home():
    return "Scholarship Recommender API is running!"

@app.route('/users')
def fetch_users():
    try:
        users = recommender.get_all_users()
        return jsonify(users)
    except Exception as e:
        logger.error(f"Error fetching users: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/user/<user_id>')
def user_data(user_id):
    try:
        user = recommender.get_user(user_id)
        if user:
            return jsonify(user)
        else:
            return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        logger.error(f"Error fetching user data: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/login', methods=['POST'])
def login_user():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        password = data.get('password')
        user = recommender.get_user(user_id)
        if user and user.get('password') == password:
            return jsonify(user), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/recommendations/all')
def all_user_recommendations():
    try:
        all_recommendations = recommender.get_all_recommendations()
        return jsonify(all_recommendations)
    except Exception as e:
        logger.error(f"Error fetching all recommendations: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/recommendations/<user_id>')
def specific_user_recommendations(user_id):
    try:
        user = recommender.get_user(user_id)
        if user:
            matches = recommender.find_matching_scholarships(user)
            return jsonify([{
                'title': scholarship.get('title'),
                'score': score
            } for scholarship, score in matches])
        else:
            return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        logger.error(f"Error fetching recommendations for user {user_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/generate_recommendations', methods=['POST'])
def generate_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        user = recommender.get_user(user_id)
        if user:
            matches = recommender.find_matching_scholarships(user)
            recommender.save_recommendations(user_id, matches)
            return jsonify({'message': 'Recommendations generated successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
