import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
import firebase_admin
from firebase_admin import credentials, firestore
import re
from datetime import datetime, timedelta
import schedule
import time
import joblib
from typing import Tuple
import uuid
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScholarshipRecommender:
    """
    This class handles the scholarship recommendation process. 
    It loads scholarship data, preprocesses it, builds a recommendation model, 
    and provides functionality to find and save scholarship recommendations for users.
    """

    def __init__(self, db_path: str, scholarship_data_path: str, model_dir: str = 'models'):
        """
        Initializes the ScholarshipRecommender with database connection, data loading,
        and model setup. 

        Args:
            db_path (str): Path to the Firebase credentials JSON file.
            scholarship_data_path (str): Path to the CSV file containing scholarship data.
            model_dir (str, optional): Directory to save/load models. Defaults to 'models'.
        """
        self.db = self._setup_firestore(db_path)
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if self._models_exist():
            self._load_models()
        else:
            self.scholarships = self._load_and_clean_scholarships(scholarship_data_path)
            self.feature_matrix, self.tfidf, self.svd, self.scaler = self._create_feature_matrix()
            self.kmeans = self._cluster_scholarships()
            self._save_models()

    def _setup_firestore(self, db_path: str) -> firestore.Client:
        """Establishes a connection to the Firestore database.

        Args:
            db_path (str): Path to the Firebase credentials JSON file.

        Returns:
            firestore.Client: Firestore client instance.
        """
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(db_path)
                firebase_admin.initialize_app(cred)
            return firestore.client()
        except Exception as e:
            logging.error(f"Failed to set up Firestore: {str(e)}")
            raise

    def _models_exist(self) -> bool:
        """Checks if all model files exist in the specified directory.

        Returns:
            bool: True if all models exist, False otherwise.
        """
        return all(os.path.exists(os.path.join(self.model_dir, f)) for f in 
                ['scholarships.joblib', 'feature_matrix.joblib', 'tfidf.joblib', 
                    'svd.joblib', 'scaler.joblib', 'kmeans.joblib'])

    def _save_models(self):
        """Saves the trained models to the specified directory."""
        joblib.dump(self.scholarships, os.path.join(self.model_dir, 'scholarships.joblib'))
        joblib.dump(self.feature_matrix, os.path.join(self.model_dir, 'feature_matrix.joblib'))
        joblib.dump(self.tfidf, os.path.join(self.model_dir, 'tfidf.joblib'))
        joblib.dump(self.svd, os.path.join(self.model_dir, 'svd.joblib'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))
        joblib.dump(self.kmeans, os.path.join(self.model_dir, 'kmeans.joblib'))
        logging.info("Models saved successfully.")

    def _load_models(self):
        """Loads the trained models from the specified directory."""
        self.scholarships = joblib.load(os.path.join(self.model_dir, 'scholarships.joblib'))
        self.feature_matrix = joblib.load(os.path.join(self.model_dir, 'feature_matrix.joblib'))
        self.tfidf = joblib.load(os.path.join(self.model_dir, 'tfidf.joblib'))
        self.svd = joblib.load(os.path.join(self.model_dir, 'svd.joblib'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
        self.kmeans = joblib.load(os.path.join(self.model_dir, 'kmeans.joblib'))
        logging.info("Models loaded successfully.")

    def _load_and_clean_scholarships(self, file_path: str) -> pd.DataFrame:
        """Loads the scholarship data from a CSV file, cleans it,
        and removes duplicates or very similar scholarships.

        Args:
            file_path (str): Path to the scholarship data CSV file.

        Returns:
            pd.DataFrame: Cleaned and de-duplicated scholarship data.
        """
        try:
            df = pd.read_csv(file_path)
            df = self._clean_scholarships(df)
            df = self._remove_duplicates_and_similar(df)
            logging.info(f"Loaded, cleaned, and deduplicated. Remaining scholarships: {len(df)}")
            return df
        except Exception as e:
            logging.error(f"Failed to load and clean scholarships: {str(e)}")
            raise

    def _clean_scholarships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs data cleaning on the scholarship DataFrame.

        Args:
            df (pd.DataFrame): Raw scholarship data.

        Returns:
            pd.DataFrame: Cleaned scholarship data.
        """
        def clean_text(text: str) -> str:
            """Cleans text data by removing special characters and converting to lowercase."""
            if pd.isna(text):
                return ''
            text = re.sub(r'[^\w\s]', ' ', str(text))
            return text.lower().strip()

        def parse_date(date_string):
            """Parses date strings into datetime objects."""
            if pd.isna(date_string):
                return pd.NaT
            date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%B %d, %Y', '%d %B %Y', '%Y/%m/%d']
            for date_format in date_formats:
                try:
                    return pd.to_datetime(date_string, format=date_format)
                except ValueError:
                    continue
            return pd.NaT

        text_columns = ['title', 'field_of_study', 'benefits', 'location', 'university', 
                        'About', 'Description', 'Applicable_programmes', 'Eligibility']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

        if 'deadline' in df.columns:
            df['deadline'] = df['deadline'].apply(parse_date)

        df = df.fillna('')
        return df

    def _remove_duplicates_and_similar(self, df: pd.DataFrame, similarity_threshold: float = 0.95) -> pd.DataFrame:
        """Removes duplicate and very similar scholarships based on their titles and descriptions.

        Args:
            df (pd.DataFrame): Scholarship DataFrame.
            similarity_threshold (float, optional): Cosine similarity threshold for considering scholarships as duplicates. 
                                                    Defaults to 0.95.

        Returns:
            pd.DataFrame: De-duplicated scholarship DataFrame.
        """
        df = df.drop_duplicates()  # Drop exact duplicates first

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['title'] + ' ' + df['Description'])
        cosine_sim = cosine_similarity(tfidf_matrix)
        distance_matrix = np.clip(1 - cosine_sim, 0, None)  # Convert similarity to distance

        dbscan = DBSCAN(eps=1-similarity_threshold, min_samples=2, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)

        # Keep only one scholarship from each cluster (representing very similar scholarships)
        unique_scholarships = df[labels == -1]  # Scholarships not in any cluster
        for cluster in set(labels):
            if cluster != -1:
                cluster_scholarships = df[labels == cluster]
                unique_scholarships = pd.concat([unique_scholarships, cluster_scholarships.iloc[[0]]])

        return unique_scholarships.reset_index(drop=True)

    def _create_feature_matrix(self) -> Tuple[np.ndarray, TfidfVectorizer, TruncatedSVD, StandardScaler]:
        """Creates a numerical feature matrix from the scholarship data using TF-IDF, dimensionality reduction,
        and feature scaling.

        Returns:
            Tuple[np.ndarray, TfidfVectorizer, TruncatedSVD, StandardScaler]: 
                - Feature matrix (numpy array)
                - Fitted TF-IDF vectorizer
                - Fitted Truncated SVD model
                - Fitted StandardScaler model
        """
        features = ['field_of_study', 'location', 'university', 'About', 'Description', 
                    'Applicable_programmes', 'Eligibility']
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000) 
        feature_matrix = tfidf.fit_transform(self.scholarships[features].apply(lambda x: ' '.join(x), axis=1))

        svd = TruncatedSVD(n_components=100, random_state=42)  # Reduce dimensionality
        feature_matrix_reduced = svd.fit_transform(feature_matrix)

        scaler = StandardScaler()  # Scale features
        feature_matrix_normalized = scaler.fit_transform(feature_matrix_reduced)

        return feature_matrix_normalized, tfidf, svd, scaler

    def _cluster_scholarships(self) -> KMeans:
        """Clusters the scholarships based on their feature vectors using K-Means clustering.

        Returns:
            KMeans: Fitted KMeans model.
        """
        kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)  # You can adjust the number of clusters
        self.scholarships['cluster'] = kmeans.fit_predict(self.feature_matrix)
        return kmeans

    def calculate_similarity(self, user_profile: dict) -> np.ndarray:
        """Calculates the cosine similarity between a user's profile and all scholarships.

        Args:
            user_profile (dict): User profile data.

        Returns:
            np.ndarray: Array of cosine similarity scores.
        """
        user_text = ' '.join([
            str(user_profile.get('intendedFieldOfStudy', '')),
            str(user_profile.get('preferredLocation', '')),
            str(user_profile.get('educationLevel', '')),
            str(user_profile.get('courseOfStudy', '')),
            str(user_profile.get('degreeType', '')),
            str(user_profile.get('financialNeed', '')),
            str(user_profile.get('incomeBracket', ''))
        ])
        user_vector = self.tfidf.transform([user_text])
        user_vector_reduced = self.svd.transform(user_vector)
        user_vector_normalized = self.scaler.transform(user_vector_reduced)
        return cosine_similarity(user_vector_normalized, self.feature_matrix)[0]

    def find_matching_scholarships(self, user_profile: dict, 
                                    min_score: float = 0.3,
                                    top_n: int = 10, 
                                    deadline_boost_days: int = 30,
                                    diversity_factor: float = 0.5) -> list:
        """Finds the best matching scholarships for a user based on similarity, 
        eligibility, deadline proximity, and diversity.

        Args:
            user_profile (dict): User profile data.
            min_score (float, optional): Minimum similarity score for a scholarship to be considered. 
                                        Defaults to 0.3.
            top_n (int, optional): Maximum number of recommendations to return. Defaults to 10.
            deadline_boost_days (int, optional): Number of days before the deadline to apply a boost. 
                                                Defaults to 30.
            diversity_factor (float, optional): Factor to promote diversity from different clusters.
                                                Defaults to 0.5.

        Returns:
            list: List of tuples (scholarship, score), sorted by score in descending order.
        """
        similarities = self.calculate_similarity(user_profile)

        scores = []
        for idx, similarity in enumerate(similarities):
            scholarship = self.scholarships.iloc[idx]

            if not self._is_eligible(user_profile, scholarship):
                continue  # Skip ineligible scholarships

            score = similarity * self._calculate_attribute_match_score(user_profile, scholarship)

            # Deadline Boost:
            deadline = scholarship.get('deadline')
            if deadline and isinstance(deadline, datetime):
                days_until_deadline = (deadline - datetime.now()).days
                if 0 <= days_until_deadline <= deadline_boost_days:
                    deadline_boost = 1 + (1 - days_until_deadline / deadline_boost_days) 
                    score *= deadline_boost

            scores.append((scholarship, score, idx))  # Store index for cluster diversity

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Apply Diversity: Prioritize scholarships from different clusters
        final_recommendations = []
        used_clusters = set()
        for scholarship, score, idx in scores:
            cluster = self.scholarships.iloc[idx]['cluster']
            if len(final_recommendations) >= top_n:
                break
            if score < min_score:
                continue
            if cluster not in used_clusters or score > scores[0][1] * diversity_factor:  # Prioritize higher scores
                final_recommendations.append((scholarship, score))
                used_clusters.add(cluster)

        return final_recommendations

    def _calculate_attribute_match_score(self, user_profile: dict, scholarship: pd.Series) -> float:
        """Calculates an additional score based on how well the scholarship attributes match the user profile.

        Args:
            user_profile (dict): User profile data.
            scholarship (pd.Series): Scholarship data.

        Returns:
            float: Attribute match score (0 to 1).
        """
        match_score = 0
        total_weights = 0 

        # Field of Study (weight = 0.25)
        if user_profile.get('intendedFieldOfStudy') and user_profile['intendedFieldOfStudy'].lower() in scholarship['field_of_study'].lower():
            match_score += 0.25
            total_weights += 0.25

        # Degree Type (weight = 0.20)
        if user_profile.get('degreeType') and user_profile['degreeType'].lower() in scholarship['Applicable_programmes'].lower():
            match_score += 0.20
            total_weights += 0.20

        # Location (weight = 0.15)
        user_location = user_profile.get('preferredLocation', user_profile.get('countryName', ''))
        if user_location and user_location.lower() in scholarship['location'].lower():
            match_score += 0.15
            total_weights += 0.15

        # Financial Need (weight = 0.15) - Adjust logic based on your data structure
        if user_profile.get('financialNeed', False) and 'need-based' in scholarship.get('scholarship_type', '').lower():
            match_score += 0.15
            total_weights += 0.15

        # Normalize the score based on used weights:
        return match_score / total_weights if total_weights > 0 else 0.0

    def _is_eligible(self, user_profile: dict, scholarship: pd.Series) -> bool:
        """Checks if the user meets the basic eligibility criteria of a scholarship.

        Args:
            user_profile (dict): User profile data.
            scholarship (pd.Series): Scholarship data.

        Returns:
            bool: True if the user is eligible, False otherwise.
        """
        # Basic eligibility check based on education level and course of study
        if scholarship['Eligibility']:
            eligibility_lower = scholarship['Eligibility'].lower()
            if user_profile.get('educationLevel', '').lower() not in eligibility_lower:
                return False
            if user_profile.get('courseOfStudy', '').lower() not in eligibility_lower:
                return False
        return True

    def save_recommendations(self, user_id: str, matches: list) -> None:
        """Saves the scholarship recommendations for a user to the Firestore database.

        Args:
            user_id (str): The ID of the user.
            matches (list): List of tuples (scholarship, score) representing the recommendations.
        """
        try:
            existing_recommendations = self.db.collection('scholarship_recommendations').document(user_id).get().to_dict()
            existing_ids = {}
            if existing_recommendations and 'recommendations' in existing_recommendations:
                existing_ids = {rec['title']: rec['id'] for rec in existing_recommendations['recommendations']}

            recommendations = []
            for scholarship, score in matches:
                unique_id = existing_ids.get(scholarship.get('title'), str(uuid.uuid4()))
                recommendations.append({
                    'id': unique_id,
                    'title': scholarship.get('title', ''),
                    'deadline': str(scholarship.get('deadline', '')),
                    'amount': scholarship.get('Grant', ''),  # Adjust field name if needed
                    'application_link': scholarship.get('application_link-href', ''),
                    'eligibility': scholarship.get('Eligibility', ''),
                    'description': scholarship.get('Description', ''),
                    'application_process': scholarship.get('application_process', ''),  # Add if available
                    'score': float(score),
                    'cluster': int(scholarship.get('cluster', -1)) 
                })

            self.db.collection('scholarship_recommendations').document(user_id).set({
                'recommendations': recommendations,
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            logging.info(f"Saved recommendations for user: {user_id}")
        except Exception as e:
            logging.error(f"Failed to save recommendations for user {user_id}: {str(e)}")

    def process_users(self, min_score: float = 0.15) -> None:
        """Processes all users from the database, generates recommendations, and saves them.

        Args:
            min_score (float, optional): Minimum score for a recommendation to be saved. Defaults to 0.15.
        """
        try:
            users = self.get_all_users()
            total_users = len(users)
            total_scholarships = 0
            processed_users = 0

            print(f"Processing recommendations for {total_users} users:")
            print("-------------------------------------------------")

            for user in users:
                user_id = user.get('userId', 'Unknown')
                first_name = user.get('firstName', 'Unknown')
                last_name = user.get('lastName', 'Unknown')

                matches = self.find_matching_scholarships(user, min_score=min_score)
                num_matches = len(matches)
                total_scholarships += num_matches 

                self.save_recommendations(user_id, matches)

                print(f"User: {first_name} {last_name}")
                print(f"User ID: {user_id}")
                print(f"Number of matched scholarships: {num_matches}")
                print("-------------------------------------------------")

                processed_users += 1

            avg_scholarships = total_scholarships / total_users if total_users > 0 else 0

            print("\nSummary:")
            print(f"Total users processed: {processed_users}")
            print(f"Total scholarships matched: {total_scholarships}")
            print(f"Average scholarships per user: {avg_scholarships:.2f}")

        except Exception as e:
            logging.error(f"Error processing users: {str(e)}")

    def get_all_users(self) -> list:
        """Retrieves all user data from the Firestore database.

        Returns:
            list: List of user dictionaries.
        """
        try:
            users_ref = self.db.collection('users')
            return [doc.to_dict() for doc in users_ref.stream()]
        except Exception as e:
            logging.error(f"Failed to get users: {str(e)}")
            return []

    def test_single_user(self, user_id: str, min_score: float = 0.3) -> None:
        """Tests the recommendation system for a single user ID.

        Args:
            user_id (str): The ID of the user.
            min_score (float, optional): Minimum score to consider a recommendation. Defaults to 0.3.
        """
        try:
            user_ref = self.db.collection('users').document(user_id)
            user = user_ref.get().to_dict()
            if user:
                matches = self.find_matching_scholarships(user, min_score=min_score)
                logging.info(f"Top scholarship matches for {user.get('firstName', '')} {user.get('lastName', '')}:")
                for scholarship, score in matches:
                    logging.info(f"Title: {scholarship['title']}")
                    logging.info(f"Score: {score:.2f}")
                    # Log other relevant scholarship details here
                    logging.info("---")
            else:
                logging.warning(f"No user found with ID: {user_id}")
        except Exception as e:
            logging.error(f"Error testing single user: {str(e)}")

    def run_continuously(self, interval_hours: int = 24):
        """Runs the scholarship recommendation process periodically.

        Args:
            interval_hours (int, optional): Interval in hours to run the process. Defaults to 24.
        """
        def job():
            """Job function to be scheduled."""
            print(f"\nStarting recommendation update at {datetime.now()}")
            self.process_users()
            print(f"Finished recommendation update at {datetime.now()}")

        # Schedule the job
        schedule.every(interval_hours).hours.do(job)

        print(f"Scheduler set to run every {interval_hours} hours.")
        print("Press Ctrl+C to stop the program.")

        # Run the job immediately for the first time
        job()

        # Keep the script running
        while True:
            schedule.run_pending()
            time.sleep(1)