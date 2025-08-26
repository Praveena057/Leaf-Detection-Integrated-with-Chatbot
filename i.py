import os
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import random
import string
import nltk
import base64 # Add base64 module for image encoding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from PIL import Image

# Define CSS for styling
css = """
<style>
.stApp {
    font-family: 'Dancing Script', cursive;/* Specify Dancing Script as the font */
    background-image: url("https://github.com/Shashipenjarla/leaf_background_photo/blob/main/bg_leaf.jpg?raw=true");
    color: #fff;
    margin: 0;
    padding: 0;
    width: 100vw;
    background-size: cover;
}

.main-header {
  font-size: 48px;
  text-align: center;
  animation: heartbeat 1.5s;
}

#leaf-chatbot-recognition{
  color:#000;
}

/* Section header */
.section-header {
  font-size: 28px;
  margin-bottom: 20px;
  color: #4B0082;
  text-decoration: underline;
  font-family: 'Arial', sans-serif;
}

/* Uploaded image */
.uploaded-image {
  border-radius: 15px;
  box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
}
.st-emotion-cache-qoz3f2 p{
  color:darkgreen;
}

.st-emotion-cache-1wda3go #image-recognition{
    height:4rem;
    width:20rem;
    background-color:dimgrey;
    border-radius:1rem;
    text-align:center;
    display:flex;
    justify-content:center;
    align-items:center;
    padding-left:0.6rem;
    text-decoration:none;
}

.st-emotion-cache-1wda3go #leafbot-chat-with-me{
    height:4rem;
    width:30rem;
    background-color:dimgrey;
    border-radius:1rem;
    text-align:center;
    display:flex;
    justify-content:center;
    align-items:center;
    padding-left:0.6rem;
}

.st-emotion-cache-12xsiil{
    height:2rem;
    width:20rem;
    background-color:dimgrey;
    border-radius:1rem;
}
/* Bot message */
.bot-message {
  background-color: #ffe4b5;/* Cornflower blue */
  color: #2e8b57;
  padding: 15px;
  margin: 15px;
  border-radius: 10px;
  font-family: 'Verdana', sans-serif;
  animation: slideInRight 0.5s ease;/* Slide in from right animation*/
}

/* Leaf information */
.leaf-info {
  background-color:rgba(0, 0, 0, 0.2); /* Wheat background */

  padding: 15px;
  margin: 15px;
  border: 2px solid #CD853F; /* Peru border */
  border-radius: 15px;
  animation: fadeInLeft 0.5s ease; /* Fade in from left animation */
}

/* Error message */
.error-message {
  color: #FF0000; /* Red error text */
  font-weight: bold;
  text-align: center;
  margin: 20px;
  animation: shake 0.5s; /* Shake animation */
}

/* Loading animation */
@keyframes pulse {
  0% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.1);
  }
  100% {
    transform: scale(1);
  }
}

/* File uploader */
div[role="button"] {
  padding: 12px 24px; /* Larger padding */
  background-color: #20B2AA; /* Light sea green background */
  color: #fff; /* White text */
  border-radius: 8px; /* Rounded corners */
  cursor: pointer;
  transition: background-color 0.3s ease;
  max-width: 150px; /* Larger button width */
  margin: 0 auto;
}

div[role="button"]:hover {
  background-color: #008080; /* Teal on hover */
}

/* Chat container */
.chat-container {
  max-width: 700px;
  margin: 0 auto;
  padding: 30px;
  background-color: rgba(0, 0, 0, 0.3); /* Alice blue background */
  border-radius: 20px;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.1); /* Soft shadow */
}

/* Chat input */
.chat-input {
  display: flex;
  align-items: center;
  margin-top: 30px;
}

.chat-input input[type="text"] {
  flex: 1;
  padding: 10px;
  border: 2px solid #4682B4; /* Steel blue border */
  border-radius: 10px;
  outline: none;
  transition: border-color 0.3s ease;
}

.chat-input input[type="text"]:focus {
  border-color: #1E90FF; /* Dodger blue on focus */
}

/* Button */
button {
  padding: 12px 24px;
  background-color: #7CFC00; /* Lawn green */
  color: #fff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

button:hover {
  background-color: #32CD32; /* Lime green on hover */
}

/* Leaf container */
.leaf-container {
  background-color: rgb(255, 228, 181); /* Blanched almond background with transparency */
  padding: 20px;
  border-radius: 20px;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.2); /* Soft shadow */
}

/* Outcome text */
.outcome-text {
  font-family: 'Arial', sans-serif;
  color: #800080; /* Purple text */
  font-size: 20px;
  margin-bottom: 20px;
}

/* Description subheader */
.description-subheader {
  font-family: 'Arial', sans-serif;
  color: #006400; /* Dark green text */
  font-size: 24px;
  margin-bottom: 10px;
}

/* Description */
.description {
  font-family: 'Arial', sans-serif;
  color: #2E8B57; /* Sea green text */
  font-size: 16px;
  line-height: 1.5;
}

/* Keyframes for slideInRight animation */
@keyframes slideInRight {
  0% {
    transform: translateX(100%); /* Start from right */
    opacity: 0; /* Start invisible */
  }
  100% {
    transform: translateX(0); /* Slide in to original position */
    opacity: 1; /* Fade in fully visible */
  }
}

/* Keyframes for fadeInLeft animation */
@keyframes fadeInLeft {
  0% {
    transform: translateX(-100%); /* Start from left */
    opacity: 0; /* Start invisible */
  }
  100% {
    transform: translateX(0); /* Slide in to original position */
    opacity: 1; /* Fade in fully visible */
  }
}

/* Keyframes for shake animation */
@keyframes shake {
  0%, 100% {
    transform: translateX(0); /* Start and end position */
  }
  10%, 30%, 50%, 70%, 90% {
    transform: translateX(-10px); /* Shake left */
  }
  20%, 40%, 60%, 80% {
    transform: translateX(10px); /* Shake right */
  }
}

@keyframes glow {
    0% {
        text-shadow: 0 0 10px #FFD700, 0 0 20px #FFD700, 0 0 30px #FFD700;
    }
    100% {
        text-shadow: 0 0 20px #FFD700, 0 0 30px #FFD700, 0 0 40px #FFD700;
    }
}

/* Add a bouncing effect for the chatbot header */
.leafbot-header {
    animation: bounce 1s ;
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% {
        transform: translateY(0);
    }
    40% {
        transform: translateY(-20px);
    }
    60% {
        transform: translateY(-10px);
    }
}

/* Add a spinning animation for the chat input box */
.chat-input input[type="text"]:focus {
    animation: spin 1s;
}

/* Add a floating effect for the chat button */
.chat-input button {
    animation: float 2s ;
}

@keyframes float {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-10px);
    }
}

/* Add a glowing effect for the submit button */
button:hover {
    animation: glow-shadow 0.5s ease-in-out infinite alternate;
}

@keyframes glow-shadow {
    0% {
        box-shadow: 0 0 5px rgba(255, 215, 0, 0.7);
    }
    100% {
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.7);
    }
}

.leaf-info {
  background-color: rgba(0, 0, 0, 0.8);
  padding: 15px;
  margin: 15px;
  border: 2px solid #CD853F; /* Peru border */
  border-radius: 15px;
}

/* Add a bouncing effect for the error messages */
.error-message {
    animation: bounce 0.5s ease-in-out infinite alternate;
}

/* Add a shimmering effect for the outcome text */
.outcome-text {
    position: relative;
}


@keyframes heartbeat {
    0%, 100% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.1);
    }
}

/* Add a rotating animation for the image uploader */
div[role="button"] {
    transition: transform 0.5s ease-in-out;
}

div[role="button"]:hover {
    transform: rotate(360deg);
}

/* Add a bouncing effect for the uploaded image */
.uploaded-image {
    animation: bounce 1s ease-in-out infinite;
}

/* Add a rotating animation for the bot messages */
.bot-message {
    animation: rotate 2s ;
}
.describe {

    color: white !important;
}

</style>
"""
# Render CSS
st.markdown(css, unsafe_allow_html=True)

# Render text using st.markdown with classes
st.markdown("<h1 class='main-header'>Leaf Chatbot & Recognition</h1>", unsafe_allow_html=True)

# Image Recognition Section
st.markdown("<h2 class='section-header'>Image Recognition</h2>", unsafe_allow_html=True)


# Initialize NLTK for text processing
nltk.download('punkt', quiet=True)

nltk.download('wordnet')

# Load leaf descriptions from a JSON file
leaf_descriptions = {}
try:
    with open('leaves.txt', 'r') as file:
        data = file.read()
        if data:
            leaf_descriptions = json.loads(data)
        else:
            st.error("The 'leaves.txt' file is empty.")
except FileNotFoundError:
    st.error("The 'leaves.txt' file was not found.")
except json.decoder.JSONDecodeError as err:
    st.error(f"Error decoding JSON data: {err}")

# Load the pre-trained leaf recognition model
model = tf.keras.models.load_model('leaf_model.keras')
leaf_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy ', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


# Function to classify leaf image

def classify_image(image_path):
    # Resize to what the model expects: 224x224
    input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    input_image_array = tf.keras.utils.img_to_array(input_image)

    # Normalize if required (optional, but often models need this)
    input_image_array = input_image_array / 255.0

    # Expand dims to get batch size of 1
    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)  # shape (1, 224, 224, 3)

    # Predict
    predictions = model.predict(input_image_exp_dim)
    temperature = 0.1
    result = tf.nn.softmax(predictions[0] /  temperature ).numpy()
    predicted_class_index = np.argmax(result)
    predicted_leaf = leaf_names[predicted_class_index]
    confidence_score = np.max(result) * 100

    outcome = f"The image belongs to {predicted_leaf} with a confidence score of {confidence_score:.2f}%"
    print(predicted_leaf, confidence_score, outcome)
    return predicted_leaf, confidence_score, outcome



lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "hello", "how is your day", "hi there", "hey there", "greetings"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def get_leaf_attribute_response(leaf_name, attribute, unique_id):
    if leaf_name in leaf_descriptions and attribute in leaf_descriptions[leaf_name]:
        print(print(leaf_descriptions.keys()))
        response = f"<div class='leaf-info' id='{unique_id}'>"
        response += f"<h3 class='describe'>{leaf_name.capitalize()}: {attribute.capitalize()}</h3>"
        response += f"<p>{leaf_descriptions[leaf_name][attribute]}</p>"
        response += "</div>"
        return response
    else:
        return f"<div class='error-message' id='{unique_id}'>Sorry, I don't have information on {attribute} for {leaf_name}.</div>"

# Update the code where you call get_leaf_attribute_response
# Function to update leaf information

def update_leaf_info(leaf_name, attributes):
    if attributes:
        for attribute in attributes:
            unique_id = f"{leaf_name.lower()}-{attribute.lower()}-{random.randint(0, 10000)}" # Generate a unique ID
            response_text = get_leaf_attribute_response(leaf_name, attribute, unique_id)
            st.markdown(f'<div class="spin-animation">{response_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown("<div class='bot-message'>I didn't understand that. Please mention a valid attribute or enter the leaf name followed by an asterisk (*) in any position to display all attributes.</div>", unsafe_allow_html=True)

attribute_keywords = {
    "color": ["color", "hue", "tone", "shade", "pigment", "tint", "chroma", "saturation"],
    "evolution": ["evolution", "development", "history", "adaptation", "genetic variation"],
    "medicinal": ["medicines", "medicine","medicinal", "therapeutic", "healing", "curative", "pharmaceutical"],
    "characteristic": ["characteristic", "traits", "feature", "attribute", "quality"],
    "use cases": ["use cases", "usages", "uses","use", "applications", "utilization"],
    "general appearance": ["general appearance", "appearance", "overall look", "visual aspect"],
    "rare appearances": ["rare appearances", "rare", "uncommon", "unusual", "scarce"],
    "availability": ["availability", "available", "presence", "existence", "accessibility"],
    "description": ["describe", "description", "detail", "elaborate", "explain"]
}

def extract_attributes(query):
    tokens = nltk.word_tokenize(query.lower())
    attributes = set()
    for token in tokens:
        for attr, keywords in attribute_keywords.items():
            if token in keywords:
                attributes.add(attr)
                break
    return attributes

# Convert image to Base64 string (optional)
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    uploaded_file = st.file_uploader('Upload a Leaf Image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None and 'leaf_class' not in st.session_state:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        image_path = os.path.join('uploads', uploaded_file.name)
        with open(image_path, 'wb') as f:
            f.write(uploaded_file.read())
        leaf_class, confidence, outcome = classify_image(image_path)
        st.session_state.leaf_class = leaf_class
        st.session_state.confidence = confidence
        st.session_state.outcome = outcome
        st.session_state.image_path = image_path

    if 'leaf_class' in st.session_state:
        st.image(st.session_state.image_path, use_container_width = True)
        st.markdown(f"<div class='outcome-text'>{st.session_state.outcome}</div>", unsafe_allow_html=True)
        description = leaf_descriptions.get(st.session_state.leaf_class.lower(), {}).get("description", "No description available.")
        st.markdown(f"""
        <div class='leaf-container'>
            <h2 class='description-subheader'>Description</h2>
            <div class='description'>{description}</div>
        </div>""", unsafe_allow_html=True)

    # Rest of your code for the chatbot section...
    st.markdown("<h1 class='leafbot-header'>LeafBot: Chat with Me!</h1>", unsafe_allow_html=True)

    user_input = st.text_input("You:", key='user_input')

    if st.button('Submit'):
        user_response = user_input.lower()
        if user_response != 'bye':
            greeting_response = greeting(user_response)
            if greeting_response:
                st.markdown(f"<div class='bot-message'>{greeting_response}</div>", unsafe_allow_html=True)
            else:
                leaf_names_in_query = [name.lower() for name in leaf_names if name.lower() in user_response]
                if len(leaf_names_in_query) > 1:
                    st.markdown("<div class='bot-message'>Sorry, only one leaf's information can be shown at a time.</div>", unsafe_allow_html=True)
                elif len(leaf_names_in_query) == 1:
                    leaf_name = leaf_names_in_query[0]
                    if f"*{leaf_name}" in user_response or f"{leaf_name}*" in user_response or f"{leaf_name} *" in user_response:
                        attributes = leaf_descriptions.get(leaf_name, {}).keys()
                        update_leaf_info(leaf_name, attributes)
                    else:
                        attributes = extract_attributes(user_response)
                        update_leaf_info(leaf_name, attributes)
                else:
                    st.markdown("<div class='bot-message'>I didn't understand that. Please mention a leaf name.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='bot-message'>Bye! Take care.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
