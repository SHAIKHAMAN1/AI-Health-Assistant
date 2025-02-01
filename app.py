import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set the page configuration (MUST BE FIRST Streamlit COMMAND)
st.set_page_config(
    page_title="Healthcare Assistant Chatbot",
    page_icon="🩺",
    layout="centered",
)

# Download necessary NLTK data once
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load the pre-trained Hugging Face model for Question Answering once
@st.cache_resource
def load_chatbot_model():
    return pipeline("question-answering", model="deepset/bert-base-cased-squad2", framework="pt")  # Force PyTorch

chatbot = load_chatbot_model()

# Define healthcare-specific response logic
def healthcare_chatbot(user_input):
    """
    Provide healthcare-related responses based on user input with detailed explanations.
    """
    context = """
    You are a healthcare assistant capable of answering questions about symptoms, medications, appointments, and general
    health advice. Common topics include colds, flu, allergies, and medical guidance. Always provide detailed and clear 
    information to help users understand the topic.
    """
    if chatbot:
        response = chatbot(question=user_input, context=context)
        detailed_response = (
            f"**Answer:** {response['answer']}\n\n"
            "Here are some additional details that might help:\n"
            "- If this relates to symptoms, ensure you monitor changes and consult a healthcare provider if necessary.\n"
            "- For medication questions, always follow the prescribed dosage and reach out to your doctor for specific advice.\n"
            "- If you're unsure about any health-related concerns, it's better to consult a professional in person."
        )
        return detailed_response
    else:
        return "The chatbot model is not available at the moment. Please try again later."

# Streamlit web app interface
def main():
    """
    Main function to run the Streamlit web app.
    """
    st.markdown("""
    <style>
    .chat-box {
        background-color: #f9f9f9;
        padding: 10px;
        margin: 10px 0;
        border-radius: 10px;
        font-size: 1.1em;
    }
    .user-message {
        text-align: right;
        background-color: #cfe2ff;
        color: #004085;
    }
    .bot-message {
        text-align: left;
        background-color: #e2f0d9;
        color: #155724;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🩺 Healthcare Assistant Chatbot")
    st.write("Hi! I'm here to help you with healthcare-related questions. Ask me about symptoms, medications, appointments, and more.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Enter your query:", placeholder="Type your question here...")
    if st.button("Send"):
        if user_input.strip():
            st.session_state["chat_history"].append(("user", user_input))
            response = healthcare_chatbot(user_input)
            st.session_state["chat_history"].append(("bot", response))
        else:
            st.warning("Please enter a valid query.")

    for sender, message in st.session_state["chat_history"]:
        css_class = "user-message" if sender == "user" else "bot-message"
        st.markdown(f'<div class="chat-box {css_class}">{message}</div>', unsafe_allow_html=True)

    st.markdown("## 🤝 Disclaimer")
    st.write("""
    This chatbot is for informational purposes only and does not replace professional medical advice.
    Always consult a doctor for medical concerns.
    """)

if __name__ == "__main__":
    main()

