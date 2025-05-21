
from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__, template_folder='templates')

# Load the trained model with greetings included
model = joblib.load('ml_chatbot_model_with_greetings.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
     return jsonify({"error": "No message provided"}), 400

    try:
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba([user_input])[0]
            max_proba = max(probas)
            predicted_category = model.classes_[probas.argmax()]

            if max_proba < 0.10:
                response = "I'm not sure what you mean. Can you try rephrasing?"
            else:
                response = get_response(predicted_category)
        else:
            predicted_category = model.predict([user_input])[0]
            response = get_response(predicted_category)
    except Exception as e:
        print("Prediction error:", e)
        response = "Something went wrong."

    return jsonify({"response": response})

def get_response(category):
    responses = {
        "greeting": "Hello! How can I assist you with the ML course today?",
    "enroll_info": "You can enroll at: https://www.verylovelymlcourse.com/enroll",
    "course_content": "The course covers supervised/unsupervised learning, deep learning, and NLP.",
    "access_help": "Try resetting your password. If issues continue, contact lovely@chatbotproject.com.",
    "certificate": "You will receive a downloadable certificate after course completion.",
    "payment": "The course costs KRW 10000. We accept card and PayPal. Refunds are available within 7 days.",
    "duration": "It's an 8-week course, around 2–3 hours per week.",
    "benefits": "By completing the course, you'll gain practical skills in ML, earn a certificate, and improve your job prospects.",
    "study_method": "The course uses a mix of pre-recorded video lessons and assignments. Occasionally, there are interactions with professors through Q&A or webinars.",
    "goodbye": "Glad I could help! Let me know if you have more questions. 😊"
    }
    return responses.get(category, "Sorry, I don't understand that question.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

