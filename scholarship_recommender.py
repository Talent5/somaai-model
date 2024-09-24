import os
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
import joblib
from typing import Tuple, List, Union
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScholarshipRecommender:
    """
    Handles scholarship recommendations.
    Loads data, preprocesses, builds model, and recommends scholarships.
    """
    def __init__(self, db: firestore.Client, scholarship_data_path: str, model_dir: str = 'models'):
        self.db = db
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

    def _models_exist(self) -> bool:
        """Checks if model files exist."""
        return all(os.path.exists(os.path.join(self.model_dir, f)) for f in
                   ['scholarships.joblib', 'feature_matrix.joblib', 'tfidf.joblib',
                    'svd.joblib', 'scaler.joblib', 'kmeans.joblib'])

    def _save_models(self):
        """Saves the trained models."""
        joblib.dump(self.scholarships, os.path.join(self.model_dir, 'scholarships.joblib'))
        joblib.dump(self.feature_matrix, os.path.join(self.model_dir, 'feature_matrix.joblib'))
        joblib.dump(self.tfidf, os.path.join(self.model_dir, 'tfidf.joblib'))
        joblib.dump(self.svd, os.path.join(self.model_dir, 'svd.joblib'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))
        joblib.dump(self.kmeans, os.path.join(self.model_dir, 'kmeans.joblib'))
        logging.info("Models saved successfully.")

    def _load_models(self):
        """Loads the trained models."""
        self.scholarships = joblib.load(os.path.join(self.model_dir, 'scholarships.joblib'))
        self.feature_matrix = joblib.load(os.path.join(self.model_dir, 'feature_matrix.joblib'))
        self.tfidf = joblib.load(os.path.join(self.model_dir, 'tfidf.joblib'))
        self.svd = joblib.load(os.path.join(self.model_dir, 'svd.joblib'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
        self.kmeans = joblib.load(os.path.join(self.model_dir, 'kmeans.joblib'))
        logging.info("Models loaded successfully.")

    def _load_and_clean_scholarships(self, file_path: str) -> pd.DataFrame:
        """Loads and cleans scholarship data, removing duplicates."""
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
        """Cleans the scholarship DataFrame."""
        def clean_text(text: str) -> str:
            """Cleans text data."""
            if pd.isna(text):
                return ''
            text = re.sub(r'[^\w\s]', ' ', str(text))
            return text.lower().strip()

        def parse_date(date_string):
            """Parses date strings."""
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
        """Removes duplicate and similar scholarships."""
        df = df.drop_duplicates()

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['title'] + ' ' + df['Description'])
        cosine_sim = cosine_similarity(tfidf_matrix)
        distance_matrix = np.clip(1 - cosine_sim, 0, None)

        dbscan = DBSCAN(eps=1 - similarity_threshold, min_samples=2, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)

        unique_scholarships = df[labels == -1]
        for cluster in set(labels):
            if cluster != -1:
                cluster_scholarships = df[labels == cluster]
                unique_scholarships = pd.concat([unique_scholarships, cluster_scholarships.iloc[[0]]])

        return unique_scholarships.reset_index(drop=True)

    def _create_feature_matrix(self) -> Tuple[np.ndarray, TfidfVectorizer, TruncatedSVD, StandardScaler]:
        """Creates a numerical feature matrix."""
        features = ['field_of_study', 'location', 'university', 'About', 'Description',
                    'Applicable_programmes', 'Eligibility']
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        feature_matrix = tfidf.fit_transform(self.scholarships[features].apply(lambda x: ' '.join(x), axis=1))

        svd = TruncatedSVD(n_components=100, random_state=42)
        feature_matrix_reduced = svd.fit_transform(feature_matrix)

        scaler = StandardScaler()
        feature_matrix_normalized = scaler.fit_transform(feature_matrix_reduced)

        return feature_matrix_normalized, tfidf, svd, scaler

    def _cluster_scholarships(self) -> KMeans:
        """Clusters scholarships using KMeans."""
        kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
        self.scholarships['cluster'] = kmeans.fit_predict(self.feature_matrix)
        return kmeans

    def calculate_similarity(self, user_profile: dict) -> np.ndarray:
        """Calculates cosine similarity between user and scholarships."""
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
        """Finds matching scholarships based on similarity and eligibility."""
        similarities = self.calculate_similarity(user_profile)

        scores = []
        for idx, similarity in enumerate(similarities):
            scholarship = self.scholarships.iloc[idx]

            if not self._is_eligible(user_profile, scholarship):
                continue

            score = similarity * self._calculate_attribute_match_score(user_profile, scholarship)

            deadline = scholarship.get('deadline')
            if deadline and isinstance(deadline, datetime):
                days_until_deadline = (deadline - datetime.now()).days
                if 0 <= days_until_deadline <= deadline_boost_days:
                    deadline_boost = 1 + (1 - days_until_deadline / deadline_boost_days)
                    score *= deadline_boost

            scores.append((scholarship, score, idx))

        scores.sort(key=lambda x: x[1], reverse=True)

        final_recommendations = []
        used_clusters = set()
        for scholarship, score, idx in scores:
            cluster = self.scholarships.iloc[idx]['cluster']
            if len(final_recommendations) >= top_n:
                break
            if score < min_score:
                continue
            if cluster not in used_clusters or score > scores[0][1] * diversity_factor:
                final_recommendations.append((scholarship, score))
                used_clusters.add(cluster)

        return final_recommendations

    def _calculate_attribute_match_score(self, user_profile: dict, scholarship: pd.Series) -> float:
        """Calculates attribute match score."""
        match_score = 0
        total_weights = 0

        if user_profile.get('intendedFieldOfStudy') and user_profile['intendedFieldOfStudy'].lower() in scholarship['field_of_study'].lower():
            match_score += 0.25
            total_weights += 0.25

        if user_profile.get('degreeType') and user_profile['degreeType'].lower() in scholarship['Applicable_programmes'].lower():
            match_score += 0.20
            total_weights += 0.20

        user_location = user_profile.get('preferredLocation', user_profile.get('countryName', ''))
        if user_location and user_location.lower() in scholarship['location'].lower():
            match_score += 0.15
            total_weights += 0.15

        if user_profile.get('financialNeed', False) and 'need-based' in scholarship.get('scholarship_type', '').lower():
            match_score += 0.15
            total_weights += 0.15

        return match_score / total_weights if total_weights > 0 else 0.0

    def _is_eligible(self, user_profile: dict, scholarship: pd.Series) -> bool:
        """Checks basic eligibility."""
        if scholarship['Eligibility']:
            eligibility_lower = scholarship['Eligibility'].lower()
            if user_profile.get('educationLevel', '').lower() not in eligibility_lower:
                return False
            if user_profile.get('courseOfStudy', '').lower() not in eligibility_lower:
                return False
        return True

    def save_recommendations(self, user_id: str, matches: list) -> None:
        """Saves recommendations to Firestore."""
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
                    'amount': scholarship.get('Grant', ''),
                    'application_link': scholarship.get('application_link-href', ''),
                    'eligibility': scholarship.get('Eligibility', ''),
                    'description': scholarship.get('Description', ''),
                    'application_process': scholarship.get('application_process', ''),
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
        """Processes users, generates recommendations."""
        try:
            users = self.get_all_users()
            for user in users:
                user_id = user.get('userId')
                if user_id:
                    matches = self.find_matching_scholarships(user, min_score=min_score)
                    self.save_recommendations(user_id, matches)
                    logging.info(f"Recommendations generated and saved for user: {user_id}")
        except Exception as e:
            logging.error(f"Error processing users: {str(e)}")

    def get_all_users(self) -> List[dict]:
        """Retrieves all users from Firestore."""
        try:
            users_ref = self.db.collection('users')
            return [doc.to_dict() for doc in users_ref.stream()]
        except Exception as e:
            logging.error(f"Failed to get users: {str(e)}")
            return []

    def get_user(self, user_id: str) -> Union[dict, None]:
        """Retrieves a user from Firestore."""
        try:
            user_ref = self.db.collection('users').document(user_id)
            user = user_ref.get()
            if user.exists:
                return user.to_dict()
            else:
                return None
        except Exception as e:
            logging.error(f"Failed to get user {user_id}: {str(e)}")
            return None

    def get_recommendations_for_user(self, user_id: str) -> Union[dict, None]:
        """Retrieves recommendations for a user."""
        try:
            recommendations_ref = self.db.collection('scholarship_recommendations').document(user_id)
            recommendations = recommendations_ref.get()

            if recommendations.exists:
                return recommendations.to_dict()
            else:
                logging.info(f"No recommendations found for user: {user_id}")
                return None
        except Exception as e:
            logging.error(f"Failed to get recommendations for user {user_id}: {str(e)}")
            return None

    def get_all_recommendations(self) -> dict:
        """Retrieves all scholarship recommendations from Firestore."""
        try:
            recommendations_ref = self.db.collection('scholarship_recommendations').stream()
            all_recommendations = {doc.id: doc.to_dict() for doc in recommendations_ref}
            return all_recommendations
        except Exception as e:
            logging.error(f"Failed to get all recommendations: {str(e)}")
            return {}
