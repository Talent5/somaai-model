import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import firebase_admin
from firebase_admin import credentials, firestore
import re
from datetime import datetime
import uuid
import logging
from typing import List, Dict, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScholarshipRecommender:
    def __init__(self, db_path: str, scholarship_data_path: str):
        self.db = self._setup_firestore(db_path)
        self.scholarships = self._load_and_clean_scholarships(scholarship_data_path)
        self.feature_matrix, self.tfidf, self.svd, self.scaler = self._create_feature_matrix()
        self.kmeans = self._cluster_scholarships()
        self.rf_classifier = self._train_rf_classifier()

    def _setup_firestore(self, db_path: str) -> firestore.Client:
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(db_path)
                firebase_admin.initialize_app(cred)
            return firestore.client()
        except Exception as e:
            logging.error(f"Failed to set up Firestore: {str(e)}")
            raise

    def _load_and_clean_scholarships(self, file_path: str) -> pd.DataFrame:
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
        def clean_text(text: str) -> str:
            if pd.isna(text):
                return ''
            text = re.sub(r'[^\w\s]', ' ', str(text))
            return text.lower().strip()

        def parse_date(date_string):
            date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%B %d, %Y', '%d %B %Y', '%Y/%m/%d']
            for date_format in date_formats:
                try:
                    return pd.to_datetime(date_string, format=date_format)
                except ValueError:
                    continue
            return pd.NaT

        text_columns = ['title', 'field_of_study', 'benefits', 'location', 'university', 'About', 'Description', 'Applicable_programmes', 'Eligibility']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

        if 'deadline' in df.columns:
            df['deadline'] = df['deadline'].apply(parse_date)

        df = df.fillna('')
        return df

    def _remove_duplicates_and_similar(self, df: pd.DataFrame, similarity_threshold: float = 0.95) -> pd.DataFrame:
        df = df.drop_duplicates()
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['title'] + ' ' + df['Description'])
        cosine_sim = cosine_similarity(tfidf_matrix)
        distance_matrix = np.clip(1 - cosine_sim, 0, None)

        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=1-similarity_threshold, min_samples=2, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)

        unique_scholarships = df[labels == -1]
        for cluster in set(labels):
            if cluster != -1:
                cluster_scholarships = df[labels == cluster]
                unique_scholarships = pd.concat([unique_scholarships, cluster_scholarships.iloc[[0]]])

        return unique_scholarships.reset_index(drop=True)

    def _create_feature_matrix(self) -> Tuple[np.ndarray, TfidfVectorizer, TruncatedSVD, StandardScaler]:
        features = ['field_of_study', 'location', 'university', 'About', 'Description', 'Applicable_programmes', 'Eligibility']
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        feature_matrix = tfidf.fit_transform(self.scholarships[features].apply(lambda x: ' '.join(x), axis=1))

        svd = TruncatedSVD(n_components=100, random_state=42)
        feature_matrix_reduced = svd.fit_transform(feature_matrix)

        scaler = StandardScaler()
        feature_matrix_normalized = scaler.fit_transform(feature_matrix_reduced)

        return feature_matrix_normalized, tfidf, svd, scaler

    def _cluster_scholarships(self) -> KMeans:
        kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
        self.scholarships['cluster'] = kmeans.fit_predict(self.feature_matrix)
        return kmeans

    def _train_rf_classifier(self) -> RandomForestClassifier:
        # Simulate user preferences (you'd replace this with actual user data)
        y = np.random.randint(0, 2, size=len(self.scholarships))
        X_train, X_test, y_train, y_test = train_test_split(self.feature_matrix, y, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logging.info(f"Random Forest Classifier Metrics - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

        return rf

    def calculate_similarity(self, user_profile: Dict[str, Any]) -> np.ndarray:
        user_text = ' '.join([
            user_profile.get('intendedFieldOfStudy', ''),
            user_profile.get('preferredLocation', ''),
            user_profile.get('educationLevel', ''),
            user_profile.get('courseOfStudy', ''),
            user_profile.get('degreeType', ''),
            user_profile.get('financialNeed', ''),
            user_profile.get('incomeBracket', '')
        ])
        user_vector = self.tfidf.transform([user_text])
        user_vector_reduced = self.svd.transform(user_vector)
        user_vector_normalized = self.scaler.transform(user_vector_reduced)
        return cosine_similarity(user_vector_normalized, self.feature_matrix)[0]

    def find_matching_scholarships(self, user_profile: Dict[str, Any], min_score: float = 0.3) -> List[Tuple[pd.Series, float]]:
        similarities = self.calculate_similarity(user_profile)
        rf_predictions = self.rf_classifier.predict_proba(self.feature_matrix)[:, 1]

        # Adjust weights based on the strength of the user profile
        profile_strength = self._calculate_profile_strength(user_profile)
        similarity_weight = 0.7 + (0.2 * (1 - profile_strength))
        rf_weight = 1 - similarity_weight

        combined_scores = similarity_weight * similarities + rf_weight * rf_predictions

        min_scholarships, max_scholarships = 5, 30
        num_scholarships = int(min_scholarships + (max_scholarships - min_scholarships) * profile_strength)
        similarity_threshold = max(min_score, 0.5 - (0.4 * profile_strength))

        matches = []
        for idx, score in enumerate(combined_scores):
            if score >= similarity_threshold:
                scholarship = self.scholarships.iloc[idx]
                if self._is_eligible(user_profile, scholarship):
                    matches.append((scholarship, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:num_scholarships]

    def _calculate_profile_strength(self, user_profile: Dict[str, Any]) -> float:
        profile_fields = ['intendedFieldOfStudy', 'preferredLocation', 'educationLevel', 'courseOfStudy', 'degreeType', 'financialNeed', 'incomeBracket']
        return sum(1 for field in profile_fields if user_profile.get(field)) / len(profile_fields)

    def _is_eligible(self, user_profile: Dict[str, Any], scholarship: pd.Series) -> bool:
        # Implement eligibility checks based on user profile and scholarship requirements
        # This is a placeholder implementation and should be expanded based on actual requirements
        if scholarship['Eligibility']:
            eligibility_lower = scholarship['Eligibility'].lower()
            if user_profile.get('educationLevel', '').lower() not in eligibility_lower:
                return False
            if user_profile.get('courseOfStudy', '').lower() not in eligibility_lower:
                return False
        return True

    def save_recommendations(self, user_id: str, matches: List[Tuple[pd.Series, float]]) -> None:
        try:
            # Fetch existing recommendations
            existing_recommendations = self.db.collection('scholarship_recommendations').document(user_id).get().to_dict()
            existing_ids = {}
            if existing_recommendations and 'recommendations' in existing_recommendations:
                existing_ids = {rec['title']: rec['id'] for rec in existing_recommendations['recommendations']}

            recommendations = []
            for scholarship, score in matches:
                # Use existing ID if available, otherwise generate a new one
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

                matches = self.find_matching_scholarships(user, min_score)
                num_matches = len(matches)

                self.save_recommendations(user_id, matches)

                print(f"User: {first_name} {last_name}")
                print(f"User ID: {user_id}")
                print(f"Number of matched scholarships: {num_matches}")
                print("-------------------------------------------------")

                total_scholarships += num_matches
                processed_users += 1

            avg_scholarships = total_scholarships / total_users if total_users > 0 else 0

            print("\nSummary:")
            print(f"Total users processed: {processed_users}")
            print(f"Total scholarships matched: {total_scholarships}")
            print(f"Average scholarships per user: {avg_scholarships:.2f}")

        except Exception as e:
            logging.error(f"Error processing users: {str(e)}")

    def get_all_users(self) -> List[Dict[str, Any]]:
        try:
            users_ref = self.db.collection('users')
            return [doc.to_dict() for doc in users_ref.stream()]
        except Exception as e:
            logging.error(f"Failed to get users: {str(e)}")
            return []

    def test_single_user(self, user_id: str, min_score: float = 0.3) -> None:
        try:
            user_ref = self.db.collection('users').document(user_id)
            user = user_ref.get().to_dict()
            if user:
                matches = self.find_matching_scholarships(user, min_score)
                logging.info(f"Top scholarship matches for {user.get('firstName', '')} {user.get('lastName', '')}:")
                logging.info(f"Number of matches: {len(matches)}")
                for scholarship, score in matches:
                    logging.info(f"Title: {scholarship['title']}")
                    logging.info(f"University: {scholarship['university']}")
                    logging.info(f"Score: {score:.2f}")
                    logging.info(f"Cluster: {scholarship['cluster']}")
                    logging.info(f"Application Link: {scholarship['application_link-href']}")
                    logging.info(f"Deadline: {scholarship['deadline']}")
                    logging.info(f"Amount: {scholarship['Grant']}")
                    logging.info(f"Eligibility: {scholarship['Eligibility']}")
                    logging.info(f"Description: {scholarship['Description']}")
                    logging.info("---")
            else:
                logging.warning(f"No user found with ID: {user_id}")
        except Exception as e:
            logging.error(f"Error testing single user: {str(e)}")