import streamlit as st
from utils import walkthrough,sample_questions,intro_to_data,banner
import time
import yaml


with open('conf_telchurn.yml', 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

# Function to map roles to Streamlit roles
def role_to_streamlit(role):
    return "assistant" if role == "model" else role

# Function to add sidebar elements
def add_sidebar_elements():
    linkedin_url = "https://www.linkedin.com/in/david-babu-15047096/"
    ko_fi_url = "https://ko-fi.com/Q5Q0V3AJA"

    icons_html = f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-left: auto; margin-right: auto; max-width: fit-content;">
        <a href="{ko_fi_url}" target="_blank">
            <img height="36" style="border:0px;height:36px;" src="https://storage.ko-fi.com/cdn/kofi2.png?v=3" border="0" alt="Buy Me a Coffee at ko-fi.com" />
        </a>
        <a href="{linkedin_url}" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Linkedin_icon.svg" alt="LinkedIn" style="width: 30px; height: 30px;">
        </a>
    </div>
    """
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    with st.sidebar.expander("Click here for a short introduction to know what I can do for you"):
        st.markdown(walkthrough())
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    with st.sidebar.expander("Click here to see some sample questions I can help you with"):
        st.markdown(sample_questions())
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    with st.sidebar.expander("Click here to get an Introduction to Data and Model behind the wraps"):
        st.markdown(intro_to_data(model_config))
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    banner()
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(icons_html, unsafe_allow_html=True)

def display_chat_history(chat_history):
    for message in chat_history:
        role = role_to_streamlit(message.role)
        parts = message.parts
        
        for part in parts:
            if "text" in part:
                if part.text and part.text.strip():
                    with st.chat_message(role):
                        st.write(part.text)




# Directly inject custom CSS to make expander header text darker
st.markdown(
    """
    <style>
    /* Make sidebar expander header text darker */
    .stSidebar .st-expander .st-expanderHeader {
        color: #1a1a1a !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Function to handle new user input and get response from the model
def handle_user_input(prompt):
    # Display user's message
    st.chat_message("user").markdown(prompt)

    with st.spinner("Processing..."):
        # Send user entry to Gemini and get the response
        response = st.session_state.chat.send_message(prompt)
        # Assuming response.candidates[0].content.parts is a list of text parts
        parts = response.candidates[0].content.parts[0].text

    # Placeholder for the assistant's response
    response_container = st.chat_message("assistant")
    response_placeholder = response_container.empty()
    response_text = ""

    # Simulate streaming each character
    for char in parts:
        response_text += char
        response_placeholder.markdown(response_text)
        time.sleep(0.005)  # Simulate streaming delay for demonstration purposes