import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing function
def clean_text(text):
    text = text.lower()

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]

    return ' '.join(words)

# Apply preprocessing
df['message'] = df['message'].apply(clean_text)

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])

y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test with new email
sample_email = [
    "Congratulations! You have won a free iPhone. Click here to claim now."
]

sample_vector = vectorizer.transform(sample_email)
prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("\nPrediction: SPAM")
else:
    print("\nPrediction: NOT SPAM")
