import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import joblib
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Configuration
DATASET_PATH = "/content/drive/My Drive/optimized_chatbot_dataset_cleaned.jsonl"
MODEL_PATH = "/content/drive/My Drive/chatbot_model.h5"
VECTORIZER_PATH = "/content/drive/My Drive/tfidf_vectorizer.pkl"

def clean_answer(answer):
    """Remove <start> and <end> tags from answers"""
    if isinstance(answer, str):
        answer = answer.replace("<start>", "").replace("<end>", "").strip()
    return answer

class TrainedModelChatbot:
    def __init__(self, dataset_path, model_path, vectorizer_path):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.df = None

    def load_resources(self):
        """Load dataset, trained model, and vectorizer"""
        # Load and clean dataset
        self.df = pd.read_json(self.dataset_path, lines=True)
        self.df['answer'] = self.df['answer'].apply(clean_answer)

        # Load trained model if exists
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)

        # Load vectorizer if exists
        if os.path.exists(self.vectorizer_path):
            self.vectorizer = joblib.load(self.vectorizer_path)
        else:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.vectorizer.fit(self.df['question'].apply(self._clean_text))

    def _clean_text(self, text):
        """Clean input text"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def predict_answer(self, question):
        """Predict answer using model or similarity"""
        try:
            # Clean question
            cleaned_q = self._clean_text(question)

            # Vectorize question
            question_vec = self.vectorizer.transform([cleaned_q])

            # Use model if available
            if self.model is not None:
                predictions = self.model.predict(question_vec)
                best_idx = np.argmax(predictions)
            else:
                # Fallback to similarity search
                question_vectors = self.vectorizer.transform(self.df['question'].apply(self._clean_text))
                similarity_scores = cosine_similarity(question_vec, question_vectors).flatten()
                best_idx = similarity_scores.argmax()

            return self.df['answer'].iloc[best_idx]

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

# Initialize chatbot
print("Loading chatbot resources...")
chatbot = TrainedModelChatbot(DATASET_PATH, MODEL_PATH, VECTORIZER_PATH)
chatbot.load_resources()
print("Chatbot ready!")

# Interactive session
print("\n=== Question Answering Chatbot ===")
print("Type 'exit' to quit\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    response = chatbot.predict_answer(user_input)
    print(f"Bot: {response}\n")
