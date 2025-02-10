from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# Download once, not in every request
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

@app.route("/", methods=["GET", "POST"])
def sentimentRequest():
    output = {}

    # Handle JSON, form, and query parameter inputs
    if request.method == "POST":
        data = request.json or request.form
    else:
        data = request.args

    sentence = data.get("q")

    # Validate input
    if not sentence:
        return jsonify({"error": "No text provided"}), 400

    # Analyze sentiment
    score = sid.polarity_scores(sentence)['compound']
    sentiment = "Positive" if score > 0 else "Negative"
    
    output["sentiment"] = sentiment
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
