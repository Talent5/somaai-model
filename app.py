import os
import logging
from flask import Flask, jsonify, request
from scholarship_recommender import ScholarshipRecommender
import firebase_admin
from firebase_admin import credentials, firestore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# --- Configuration ---
SCHOLARSHIP_DATA_PATH = './data/scholarships.csv'
MODEL_DIR = './models'

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Flask App ---
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# --- Initialize Firebase ---
try:
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
    db = firestore.client()
    logger.info("Firebase initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Firebase: {str(e)}")
    exit(1)

# --- Initialize Recommender ---
try:
    recommender = ScholarshipRecommender(db=db, scholarship_data_path=SCHOLARSHIP_DATA_PATH, model_dir=MODEL_DIR)
    logger.info("Scholarship Recommender initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize ScholarshipRecommender: {str(e)}")
    exit(1)

# --- Scheduler ---
scheduler = BackgroundScheduler()

@scheduler.scheduled_job(IntervalTrigger(hours=0.002))
def run_recommendation_job():
    """Generate and save recommendations for all users."""
    with app.app_context():
        try:
            recommender.process_users(min_score=0.15)
        except Exception as e:
            logger.error(f"Error in scheduled recommendation job: {str(e)}")

scheduler.start()
logger.info("Scheduler started.")

# --- Helper Functions ---

def _get_user_data(user_id: str) -> dict:
    """Helper function to fetch user data from Firestore."""
    user_ref = db.collection('users').document(user_id)
    user_doc = user_ref.get()
    if user_doc.exists:
        return user_doc.to_dict()
    else:
        return None

# --- Error Handling ---

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad Request'}), 400

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.exception('An error occurred during a request.')
    return jsonify({'error': 'Internal server error'}), 500

# --- API Endpoints ---

@app.route('/')
def home():
    return jsonify({"message": "Scholarship Recommender API is running!"}), 200

# --- User Endpoints ---

@app.route('/users', methods=['GET', 'POST'])
def handle_users():
    """Handles GET (fetch all users) and POST (create user) requests for users."""
    if request.method == 'GET':
        try:
            users = recommender.get_all_users()
            return jsonify(users), 200
        except Exception as e:
            logger.error(f"Error fetching users: {str(e)}")
            return jsonify({'error': 'Failed to fetch users'}), 500
    elif request.method == 'POST':
        try:
            data = request.get_json()
            user_id = data.get('userId')
            if not user_id:
                return jsonify({'error': 'userId is required'}), 400

            existing_user = _get_user_data(user_id)
            if existing_user:
                return jsonify({'error': 'User already exists'}), 400

            db.collection('users').document(user_id).set(data)
            logger.info(f"Created new user: {user_id}")
            return jsonify({'message': 'User created successfully', 'userId': user_id}), 201
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return jsonify({'error': 'Failed to create user'}), 500

@app.route('/users/<string:user_id>', methods=['GET', 'PUT', 'DELETE'])
def handle_user(user_id):
    """Handles GET (fetch user), PUT (update user), and DELETE (delete user) requests."""
    try:
        user_ref = db.collection('users').document(user_id)

        if request.method == 'GET':
            user = _get_user_data(user_id)
            if user:
                return jsonify(user), 200
            else:
                return jsonify({'error': 'User not found'}), 404
        elif request.method == 'PUT':
            if not user_ref.get().exists:
                return jsonify({'error': 'User not found'}), 404
            data = request.get_json()
            user_ref.update(data)
            logger.info(f"Updated user profile: {user_id}")
            return jsonify({'message': 'User profile updated successfully'}), 200
        elif request.method == 'DELETE':
            if not user_ref.get().exists:
                return jsonify({'error': 'User not found'}), 404
            user_ref.delete()
            # You might also want to delete associated recommendations here
            logger.info(f"Deleted user: {user_id}")
            return jsonify({'message': 'User deleted successfully'}), 200
    except Exception as e:
        logger.error(f"Error handling user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to handle user request'}), 500

# --- Recommendation Endpoints ---

@app.route('/recommendations', methods=['GET'])
def get_all_recommendations():
    """Get recommendations for all users."""
    try:
        all_recommendations = recommender.get_all_recommendations()
        return jsonify(all_recommendations), 200
    except Exception as e:
        logger.error(f"Error fetching all recommendations: {str(e)}")
        return jsonify({'error': 'Failed to fetch recommendations'}), 500

@app.route('/users/<string:user_id>/recommendations', methods=['GET'])
def get_recommendations_for_user(user_id):
    """Get recommendations for a specific user."""
    try:
        recommendations = recommender.get_recommendations_for_user(user_id)
        if recommendations is not None:
            return jsonify(recommendations), 200
        else:
            return jsonify({'error': 'User not found or no recommendations'}), 404
    except Exception as e:
        logger.error(f"Error fetching recommendations for user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to fetch recommendations'}), 500

@app.route('/users/<user_id>/generate_recommendations', methods=['POST'])
def generate_recommendations_for_user(user_id):
    """Generate recommendations for a specific user on demand."""
    try:
        user = recommender.get_user(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        matches = recommender.find_matching_scholarships(user)
        recommender.save_recommendations(user_id, matches)
        return jsonify({'message': f'Recommendations generated and saved for user {user_id}'}), 200
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to generate recommendations'}), 500

# --- Authentication Endpoint (Optional) ---

@app.route('/login', methods=['POST'])
def login_user():
    """Login user (You should implement proper authentication here)."""
    data = request.get_json()
    user_id = data.get('userId')
    password = data.get('password')

    # **IMPORTANT:** Replace this placeholder with your actual authentication logic
    # (e.g., using Firebase Authentication or another authentication system)

    # Placeholder - Example using a hardcoded user
    if user_id == 'testuser' and password == 'testpassword':
        return jsonify({'message': 'Login successful', 'userId': user_id}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
