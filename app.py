from flask import Flask, request, jsonify
pip install textblob
import openai
import joblib

app = Flask(__name__)

openai.api_key = "YOUR_OPENAI_API_KEY"

# Load your trained ML model (e.g., price prediction model)
model = joblib.load('price_model.pkl')

@app.route('/negotiate', methods=['POST'])
def negotiate():
    user_price = request.json['price']

    # Use your ML model to predict an optimal price (as an example)
    optimal_price = model.predict([[user_price]])[0]

    # Chatbot logic (same as before)
    bot_message = f"The optimal price we can offer is ${optimal_price:.2f}. How would you like to proceed?"

    return jsonify({
        'user_price': user_price,
        'bot_message': bot_message,
        'final_price': optimal_price
    })

from textblob import TextBlob

@app.route('/negotiate', methods=['POST'])
def negotiate():
    user_message = request.json['message']

    # Analyze sentiment
    sentiment = TextBlob(user_message).sentiment.polarity

    # Adjust negotiation based on sentiment
    if sentiment > 0.5:
        bot_message = "You seem very polite, so we’ll offer a better price!"
    else:
        bot_message = "We can't offer any additional discounts at this time."

    return jsonify({
        'message': user_message,
        'bot_response': bot_message
    })


if __name__ == '__main__':
    app.run(debug=True)
