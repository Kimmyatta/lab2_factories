from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email
import json
import os

router = APIRouter()

class EmailRequest(BaseModel):
    subject: str
    body: str
    mode: Optional[str] = "topic"

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    message: str
    email_id: int

class TopicCreateRequest(BaseModel):
    name: str
    description: str

class EmailCreateRequest(BaseModel):
    subject: str
    body: str
    ground_truth: Optional[str] = None   

@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        # TOPIC MODE (default)
        if request.mode == "topic":
           result = inference_service.classify_email(email)
        
           return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"]
        )
        # EMAIL SIMILARITY MODE
        elif request.mode == "email":

            result = inference_service.classify_by_stored_emails(email)

            return EmailClassificationResponse(
                predicted_topic=result["predicted_topic"],
                topic_scores={},
                features={},
                available_topics=[]
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use 'topic' or 'email'.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

@router.get("/pipeline/info") 
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()

@router.post("/topics")
async def add_topic(request: TopicCreateRequest):
    try:
        topic_file = "data/topic_keywords.json"

        # Load existing topics
        with open(topic_file, "r") as f:
            topics = json.load(f)

        # Add new topic dynamically
        topics[request.name] = {
            "description": request.description
        }

        # Save back to file
        with open(topic_file, "w") as f:
            json.dump(topics, f, indent=2)

        return {
            "message": f"Topic '{request.name}' added successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emails", response_model=EmailAddResponse)
async def add_email(request: EmailCreateRequest):
    try:
        email_file = "data/emails.json"

        # Load existing emails
        with open(email_file, "r") as f:
            emails = json.load(f)

        # Create new email object
        new_email = {
            "subject": request.subject,
            "body": request.body,
            "ground_truth": request.ground_truth
        }

        # Add to list
        emails.append(new_email)

        # Save back to file
        with open(email_file, "w") as f:
            json.dump(emails, f, indent=2)

        return EmailAddResponse(
            message="Email stored successfully",
            email_id=len(emails) - 1
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# TODO: LAB ASSIGNMENT - Part 2 of 2  
# Create a GET endpoint at "/features" that returns information about all feature generators
# available in the system.
#
# Requirements:
# 1. Create a GET endpoint at "/features"
# 2. Import FeatureGeneratorFactory from app.features.factory
# 3. Use FeatureGeneratorFactory.get_available_generators() to get generator info
# 4. Return a JSON response with the available generators and their feature names
# 5. Handle any exceptions with appropriate HTTP error responses
#
# Expected response format:
# {
#   "available_generators": [
#     {
#       "name": "spam",
#       "features": ["has_spam_words"]
#     },
#     ...
#   ]
# }
#
# Hint: Look at the existing endpoints above for patterns on error handling
# Hint: You may need to instantiate generators to get their feature names

