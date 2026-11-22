
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv(r"C:\Users\omprakash sah\Downloads\spam (1).csv")

# Remove duplicate data
data = data.drop_duplicates()

# Replace category names
data['Category'] = data['Category'].replace(
    ['ham', 'spam'],
    ['Not Spam', 'spam']
)

# Separate messages and categories
mess = data['Message']
cat = data['Category']

# Split dataset
mess_train, mess_test, cat_train, cat_test = train_test_split(
    mess,
    cat,
    test_size=0.2,
    random_state=42
)

# Convert text into numerical features
cv = CountVectorizer(stop_words='english')

features = cv.fit_transform(mess_train)

features_test = cv.transform(mess_test)

# Create Model
model = MultinomialNB()

# Train Model
model.fit(features, cat_train)

# Test Model
prediction = model.predict(features_test)

accuracy = accuracy_score(cat_test, prediction)

# Prediction Function
def predict(message):
    input_message = cv.transform([message])
    result = model.predict(input_message)
    return result[0]

# Streamlit Web Application
st.title("Spam Detection System")

st.write("Enter a message to check whether it is Spam or Not Spam.")

input_mess = st.text_input("Enter Message Here")

if st.button("Validate"):

    if input_mess.strip():

        output = predict(input_mess)

        if output == "spam":
            st.error("This Message is Spam!")

        else:
            st.success("This Message is Not Spam!")

    else:
        st.warning("Please Enter a Message!")

# Display Accuracy
st.write("Model Accuracy:", round(accuracy * 100, 2), "%")





