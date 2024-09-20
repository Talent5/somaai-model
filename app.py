import os
from flask import Flask, jsonify
from scholarship_recommender import ScholarshipRecommender
import firebase_admin
from firebase_admin import credentials
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

app = Flask(__name__)

# Initialize Firebase
if not firebase_admin._apps:
    firebase_config = {
        'type': os.environ['FIREBASE_TYPE'],
        'project_id': os.environ['FIREBASE_PROJECT_ID'],
        'private_key_id': os.environ['FIREBASE_PRIVATE_KEY_ID'],
        'private_key': os.environ['FIREBASE_PRIVATE_KEY'].replace('\\n', '\n'),  # Ensure newlines are properly formatted
        'client_email': os.environ['FIREBASE_CLIENT_EMAIL'],
        'client_id': os.environ['FIREBASE_CLIENT_ID'],
        'auth_uri': os.environ['FIREBASE_AUTH_URI'],
        'token_uri': os.environ['FIREBASE_TOKEN_URI'],
        'auth_provider_x509_cert_url': os.environ['FIREBASE_AUTH_PROVIDER_X509_CERT_URL'],
        'client_x509_cert_url': os.environ['FIREBASE_CLIENT_X509_CERT_URL'],
    }
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)

# Initialize the recommender
scholarship_data_path = './data/scholarships.csv'
recommender = ScholarshipRecommender(
    db_path=firebase_config,
    scholarship_data_path=scholarship_data_path,
    model_dir='./models'
)

# Set up scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=recommender.process_users,
    trigger=IntervalTrigger(hours=24),
    id='scholarship_recommendation_job',
    name='Generate scholarship recommendations every 24 hours',
    replace_existing=True)
scheduler.start()

@app.route('/')
def home():
    return "Scholarship Recommender is running!"

@app.route('/user/<user_id>')
def user_data(user_id):
    # Fetch and display user data
    user = recommender.get_user(user_id)
    if user:
        return jsonify(user)  # Customize the response as needed
    else:
        return jsonify({'error': 'User not found'}), 404

@app.route('/recommendations/all')
def all_user_recommendations():
    # Fetch and display recommendations for all users
    all_recommendations = recommender.get_all_recommendations()  # Implement this method
    return jsonify(all_recommendations)

@app.route('/recommendations/<user_id>')
def specific_user_recommendations(user_id):
    # Fetch and display recommendations for a specific user
    user = recommender.get_user(user_id)
    if user:
        matches = recommender.find_matching_scholarships(user)
        return jsonify([{
            'title': scholarship['title'],
            'score': score
        } for scholarship, score in matches])
    else:
        return jsonify({'error': 'User not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
