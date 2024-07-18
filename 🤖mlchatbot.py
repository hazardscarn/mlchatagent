from google.cloud import bigquery
from google.cloud import bigquery_connection_v1 as bq_connection
from google.cloud import bigquery_storage
import google.auth
import pandas as pd
from google.cloud.exceptions import NotFound
from google.cloud import aiplatform
from vertexai.generative_models import GenerationConfig
import vertexai
import google.generativeai as genai
import google.ai.generativelanguage as glm
from google.generativeai import caching
import streamlit as st

st.set_page_config(
    page_title="MLy - ML Model Whisperer",
    page_icon="🤖",
)


from utils import walkthrough,access_secret_version,sample_questions,normalize_string,remove_sql_and_backticks,agent_prompt
from toolbox import generate_sql,execute_sql,subset_churn_contribution_analysis
from toolbox import subset_clv_analysis,generate_visualizations,subset_shap_summary,question_reformer,customer_recommendations,model_stat
from streamlit_utils import add_sidebar_elements,display_chat_history,handle_user_input

#GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
project_id = "mlchatagent-429005"
secret_id = "GOOGLE_API_KEY"

# Access the secret
GOOGLE_API_KEY = access_secret_version(project_id, secret_id)

##Create the Main Agent
gen_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-001",system_instruction=agent_prompt(),
    tools=[generate_sql,execute_sql,subset_churn_contribution_analysis,subset_clv_analysis,generate_visualizations
           ,subset_shap_summary,question_reformer,customer_recommendations,model_stat],
    generation_config={"temperature":0.3})







# Initialize chat session in session state
if "chat" not in st.session_state:
    st.session_state.chat = gen_model.start_chat(enable_automatic_function_calling=True)
if "intermediate_results" not in st.session_state:
    st.session_state.intermediate_results = {}





##Add Title
st.markdown("<h2 style='text-align: center;'>Meet MLy 🤖: Your ML Model Whisperer</h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)



# Display sidebar elements
add_sidebar_elements()

display_chat_history(st.session_state.chat.history)

if prompt := st.chat_input("I possess a well of knowledge. What would you like to know?"):
    handle_user_input(prompt)
    
