from typing import Dict, Any
from app.models.similarity_model import EmailClassifierModel
from app.features.factory import FeatureGeneratorFactory
from app.dataclasses import Email

class EmailTopicInferenceService:
    """Service that orchestrates email topic classification using feature similarity matching"""
    
    def __init__(self):
        self.model = EmailClassifierModel()
        self.feature_factory = FeatureGeneratorFactory()
    
    def classify_email(self, email: Email) -> Dict[str, Any]:
        """Classify an email into topics using generated features"""
        
        # Step 1: Generate features from email
        features = self.feature_factory.generate_all_features(email)
        
        # Step 2: Classify using features
        predicted_topic = self.model.predict(features)
        topic_scores = self.model.get_topic_scores(features)
        
        # Return comprehensive results
        return {
            "predicted_topic": predicted_topic,
            "topic_scores": topic_scores,
            "features": features,
            "available_topics": self.model.topics,
            "email": email
        }
    
    def classify_by_stored_emails(self, email: Email) -> Dict[str, Any]:
        """Classify email by comparing it with stored emails using embeddings"""
        import json
        import numpy as np
        from app.features.generators import EmailEmbeddingsFeatureGenerator

        email_file = "data/emails.json"

        # Load stored emails
        with open(email_file, "r") as f:
            stored_emails = json.load(f)

        if not stored_emails:
            raise ValueError("No stored emails available")

        embedding_generator = EmailEmbeddingsFeatureGenerator()

        # Generate embedding for input email
        input_features = embedding_generator.generate_features(email)
        input_embedding = np.array(input_features["average_embedding"])

        best_score = -1
        best_topic = None

        # Compare with each stored email
        for stored in stored_emails:

            stored_email_obj = Email(
                subject=stored["subject"],
                body=stored["body"]
            )

            stored_features = embedding_generator.generate_features(stored_email_obj)
            stored_embedding = np.array(stored_features["average_embedding"])

            # Cosine similarity
            dot_product = np.dot(input_embedding, stored_embedding)
            norm_input = np.linalg.norm(input_embedding)
            norm_stored = np.linalg.norm(stored_embedding)

            if norm_input == 0 or norm_stored == 0:
                continue

            cosine_similarity = dot_product / (norm_input * norm_stored)

            if cosine_similarity > best_score:
                best_score = cosine_similarity
                best_topic = stored.get("ground_truth")

        if best_topic is None:
            raise ValueError("No matching email found")

        return {
            "predicted_topic": best_topic,
            "similarity_score": float(best_score)
        }    

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the inference pipeline"""
        return {
            "available_topics": self.model.topics,
            "topics_with_descriptions": self.model.get_all_topics_with_descriptions()
        }