import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
ALLOWED_LOCATIONS = [
    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
    "El Paso, Texas", "Escondido, California", "Fresno, California",
    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
    "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
    "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
]

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            query_string = parse_qs(environ["QUERY_STRING"])
            location = query_string.get("location", [None])[0]
            start_date = query_string.get("start_date", [None])[0]
            end_date = query_string.get("end_date", [None])[0]

            filtered_reviews = reviews
            if location:
                if location in ALLOWED_LOCATIONS:
                    filtered_reviews = [review for review in reviews if review["Location"] == location]
                else:
                    filtered_reviews = []

            if start_date:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
                filtered_reviews = [
                    review for review in filtered_reviews
                    if datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") >= start_datetime
                ]

            if end_date:
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
                filtered_reviews = [
                    review for review in filtered_reviews
                    if datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") <= end_datetime
                ]

            for review in filtered_reviews:
                review["sentiment"] = self.analyze_sentiment(review["ReviewBody"])

            filtered_reviews.sort(key=lambda x: x["sentiment"]["compound"], reverse=True)

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            request_body_size = int(environ.get('CONTENT_LENGTH', 0))
            request_body = environ['wsgi.input'].read(request_body_size)
            request_data = parse_qs(request_body.decode('utf-8'))

            review_body = request_data.get("ReviewBody", [None])[0]
            location = request_data.get("Location", [None])[0]

            if not review_body or not location or location not in ALLOWED_LOCATIONS:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Bad Request"]

            new_review = {
                "ReviewId": str(uuid.uuid4()),
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            reviews.append(new_review)

            response_body = json.dumps(new_review, indent=2).encode("utf-8")
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]
            

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()