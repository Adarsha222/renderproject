<<<<<<< HEAD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib

# Training data
X = [
    # Greetings (formal, casual, slang)
    "Hello", "Hi there", "Hey!", "Good morning", "Good evening", "Howdy partner", "Yo!", "What's going on?",
    "Hey, how are you?", "Greetings!", "Hi, nice to meet you", "Hey buddy", "Morning, what's new?",
    "Hi, hope you are well", "Yo, what's up?", "Heya!",

    # Enroll Info (direct questions, indirect, casual, confused)
    "How do I sign up for the course?", "Where can I register?", "I'm interested in joining, how?", 
    "What's the enrollment procedure?", "Tell me how to get on board", "Is enrollment still open?",
    "I want to register but don't know how", "How can I apply?", "Who do I contact to enroll?",
    "Is there a deadline for joining?", "Can I join the course anytime?", "How does registration work?",
    "I'm confused about enrollment", "Need help with signing up", "What's the process for admission?",

    # Course Content (curious, detailed, broad, skeptical)
    "What exactly do you teach in this course?", "Does the course include AI?", "Will I learn about neural networks?",
    "What are the main topics covered?", "Can you give me a syllabus?", "Does the course touch on data science?",
    "Are there hands-on projects?", "Is computer vision part of it?", "What skills will I gain?",
    "How advanced is the material?", "Will this course help me get a job?", "Is Python used in the course?",
    "What can I expect to learn?", "Does it include statistics for ML?", "Are there quizzes or exams?",

    # Access Help (frustrated, technical, confused)
    "I can't log into my account", "The course page won’t load", "Password reset isn't working",
    "I'm stuck on the login screen", "Why am I not able to access my lessons?", "Access denied when I try to enter",
    "Help! I forgot my password", "My login keeps failing", "Page shows error when opening course",
    "I'm locked out of the course", "Can’t get past the sign-in", "My account won’t authenticate",
    "Having issues accessing content", "I think I’m blocked from the course", "Need assistance logging in",

    # Certificate (formal, casual, uncertain)
    "Do you provide a certificate after course completion?", "Is there a certificate?", "How do I get my diploma?",
    "Will I receive proof that I completed the course?", "What’s the process to claim my certificate?", 
    "Do you send certificates by email?", "Is there a fee for the certificate?", "When will I get the certificate?",
    "Can I download my certificate anytime?", "How do I verify my completion?", "Is the certificate recognized?",
    "Are certificates digital or physical?", "I completed the course but haven’t received a certificate",
    "Is certification included?", "What kind of credential will I get?",

    # Payment (formal, casual, doubts, problem)
    "How much does this course cost?", "Is there a way to pay in installments?", "Can I get a refund?",
    "Do you accept PayPal or credit cards?", "What's the price?", "Are there any discounts available?",
    "Is the payment process secure?", "I paid but didn’t get access", "Can I pay with a debit card?",
    "How do I pay for the course?", "Are there hidden fees?", "Is there a money-back guarantee?",
    "Can I get a receipt?", "Do you have scholarship options?", "Is payment required upfront?",

    # Duration (curious, uncertain, detailed)
    "How long will it take to complete the course?", "Is this a 4-week program?", "What’s the total study time?",
    "Can I finish it in a month?", "How many hours per week should I dedicate?", "Is it self-paced or scheduled?",
    "When does the course end?", "Is there a time limit?", "How flexible is the schedule?",
    "Do I need to be online at fixed times?", "How many modules are there?", "Is it possible to take breaks?",
    "Will it take more than 10 hours per week?", "Is the course intensive?", "How fast can I finish?",

    # Goodbye (casual, polite, friendly)
    "Thanks a lot!", "Thank you very much", "Goodbye", "See you later", "Bye!", "Catch you later",
    "Appreciate your help", "That’s all for now", "Have a good one", "Take care", "Later!",
    "Thanks, bye!", "Cheers", "Talk to you soon", "Bye bye",

    # Find Presentation (direct, indirect, casual, polite)
    "Can you find the presentation?", "Show me the slides please", "Where is the PPT?",
    "Do you have the presentation link?", "Send me the presentation file", "I want to see the deck",
    "Could you share the slide link?", "Presentation please", "Is the presentation available?",
    "How can I access the PPT?", "Please provide the slides", "Can I get the link to the presentation?",
    "Where can I find the lecture presentation?", "I'd like to review the slides", "Can you share the PowerPoint?"
]

y = [
    "greeting"] * 16 + \
    ["enroll_info"] * 16 + \
    ["course_content"] * 16 + \
    ["access_help"] * 16 + \
    ["certificate"] * 16 + \
    ["payment"] * 16 + \
    ["study_methods"] * 10 + \
    ["goodbye"] * 15 + \
    ["find_presentation"] * 15


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
=======
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
joblib.dump(model, 'ml_chatbot_model_diverse.pkl')
print("\n✅ Model trained and saved as 'ml_chatbot_model_diverse.pkl'")

# Test prediction
user_input = "enroll"
predicted = model.predict([user_input])[0]
print(f"\n🧠 Predicted category: {predicted}")

# Show lengths
print(f"Length of X: {len(X)}")
print(f"Length of y: {len(y)}")
>>>>>>> 7d7dd324e298a62abef5923901317e80c8e6428a
