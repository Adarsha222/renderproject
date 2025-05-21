from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib

# Training data
X = [
   "Hello", "Hi", "Hey", "Good morning", "Good evening", "Howdy", "Yo", "What's up?", "Hi there", "Greetings",

    # Enroll Info
    "How can I enroll?", "Where do I sign up?", "I want to join the course", "Tell me how to register",
    "Enrollment process?", "Sign up instructions?", "How to enroll for ML course?", "How to join?",
    "Course admission info", "Joining the course",

    # Course Content
    "What topics are covered?", "Is deep learning included?", "What will I learn?", "Give me a course outline",
    "Course modules?", "Topics in this course?", "Will I learn NLP?", "Explain course syllabus",
    "What areas of ML are taught?", "Course content overview",

    # Access Help
    "I can't access my course", "Having trouble logging in", "My course isn't opening", "Login not working",
    "I forgot my password", "Can't log in", "Access problem", "Course page is blank", "Stuck on login screen",
    "Need help logging in",

    # Certificate
    "Will I get a certificate?", "Is certificate provided?", "How do I download my certificate?",
    "Certificate issue", "Where's my certificate?", "I finished but no certificate", 
    "Certificate not showing", "How to claim certificate?", "Do I get proof of completion?", 
    "When is certificate given?",

    # Payment
    "How much does the course cost?", "Is there a refund?", "Payment methods?", "Do you accept PayPal?",
    "I want a refund", "Price of the course?", "Course fee?", "Payment problem", 
    "I paid but can't access", "Where to pay?","How to pay",

    # Duration
    "How long is the course?", "Course duration?", "Total time to finish?", "Is it 4 weeks long?",
    "Length of the course?", "Duration details?", "What’s the time commitment?", "When does the course end?",
    "Weeks of study?", "How many hours per week?",

    # Goodbye
    "Thanks", "Thank you", "Goodbye", "Bye", "See you", "Later", "Appreciate it", "That’s all", 
    "Good night", "Take care"]

y = [ "greeting"] * 10 + \
    ["enroll_info"] * 10 + \
    ["course_content"] * 10 + \
    ["access_help"] * 10 + \
    ["certificate"] * 10 + \
    ["payment"] * 11 + \
    ["study_methods"]*10 +\
    ["goodbye"] * 10


# Create pipeline and train model
model = make_pipeline(TfidfVectorizer(lowercase=True,ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
model.fit(X, y)

# Evaluate on training data
print("\n📊 Evaluation Report:")
print(classification_report(y, model.predict(X)))

# Save the trained model
joblib.dump(model, 'ml_chatbot_model_with_greetings.pkl')
print("\n✅ Model trained and saved as 'ml_chatbot_model_with_greetings.pkl'")

# Test prediction
user_input = "enroll"
predicted = model.predict([user_input])[0]
print(f"\n🧠 Predicted category: {predicted}")

# Show lengths
print(f"Length of X: {len(X)}")
print(f"Length of y: {len(y)}")
