import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Step 1: Load the CSV file
df = pd.read_csv('/content/head.csv')

# Step 2: Check for missing values in the dataset (optional but good practice)
print("Missing values in the dataset:")
print(df.isnull().sum())

# Step 3: Preprocess the dataset
# Extracting relevant columns: 'text' for the headlines, 'sentiment' for labels
X = df['text']  # The news headlines
y = df['sentiment']  # The sentiment labels (e.g., Positive, Negative, Neutral)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Convert text data to numerical vectors using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 6: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

# Step 8: Run the chatbot
def chatbot():
    print("Hello! I can analyze the sentiment of news headlines. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Check if the headline exists in the dataset
        match = df[df['text'].str.contains(user_input, case=False)]

        if not match.empty:
            # If a match is found, print the sentiment
            predicted_sentiment = match.iloc[0]['sentiment']
            print(f"Chatbot: Predicted Sentiment: {predicted_sentiment}")
        else:
            print("Chatbot: No matching headline found in the dataset.")

# Run the chatbot
chatbot()
