import re
from google.cloud import secretmanager
import yaml
import streamlit as st

# Load configuration files
with open('llm_configs.yml', 'r') as f:
    llm_config = yaml.load(f, Loader=yaml.FullLoader)

with open('conf_telchurn.yml', 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

def remove_sql_and_backticks(input_text):
    modified_text = re.sub(r'```|sql', '', input_text)
    modified_text = re.sub(r'\\\\', '', modified_text)
    return modified_text

def normalize_string(s):
    s = s.lower()
    #s = re.sub(r'\s+', ' ', s)  # Replace multiple spaces with a single space
    s = re.sub(r'[ \t]+', ' ', s)  # Replace multiple spaces with a single space
    #s = re.sub(r'[^a-zA-Z0-9\s]', '', s)  # Remove all non-alphanumeric characters except spaces
    return s.strip()



def access_secret_version(project_id, secret_id, version_id="latest"):
    """
    Access the payload for the given secret version if one exists. The version
    can be a version number as a string (e.g. "5") or an alias (e.g. "latest").
    """
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version.
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version.
    response = client.access_secret_version(name=name)

    # Return the secret payload.
    # WARNING: Do not print the secret in production.
    payload = response.payload.data.decode("UTF-8")
    return payload



def sample_questions():

    response = """ You can ask me about anything about the process you are looking for.\n

In this scenario, I'm provided with dataset and ML model of a telecom company. I can help answer any question you have based on your role or generic.

Here are some sample questions you can ask me if you are a :-

1. **Retention/Pricing/Marketing Analyst**:\n
    a. What are the main reasons for churn for customers?\n
    b. Create buckets for current equipment age with this logic 0-90,91-180,181-365,366-720,720-1000,1000+. Then give me average churn rate and count for these buckets.\n
    c. What is the net effect on CLV if we change the currentequipment age to 30 for those customers with currentequipment age greater than 900 days? Assume the cost of treatment is $150 per customer.\n
    e. What is the net effect on CLV if we change the currentequipment age to 30 for those customers with currentequipment age greater than 900 days and have churn prediction more than 0.5? Assume the cost of treatment is $150 per customer
    f. What is the average age of customers who have higher churn because of revenue_per_minute?\n

2. **Customer Service Rep**:\n
    a. What are the recommendations to reduce churn probability for customer with customer_id 3334558?\n
    b. Who is customer_id 3334558?\n 

3. **Data Analyst**:\n
    a. What are the top 10 customers with highest churn probability?\n
    b. What is the churn probability distribution for customers with revenue_per_minute more than 0.5?\n
    c. Create a vizualization of average churn rate and average predicted churn across different occupation\n
    d. How many customers with children and aged under 50 have current equipment age more than 600 days?\n

4. **Anyone**:\n
    a. What are the stats of the model?\n
    b. What is the AUC and F1 score of the model?\n
    c. Display the average churn vs average predicted churn across vigintiles distribution of model with test data\n

These are some sample questions you can ask me. Feel free to ask me anything you want to know irrepsctive of your role.\n
I am here to help you talk to the ML model in English and get the best informed answers for your questions.\n

                """
    return response

def walkthrough():
    """
    Provides a short introduction to all the tools available in the chatbot.
    Use this function to provide a short introduction to all the tools available in the chatbot.
    """
    response = """
    Hello! I'm MLy 🤖 your friendly AI assistant

    I'm here to help you interact effortlessly with our powerful machine learning models.
    Just ask me what you need in plain English, and I'll take care of the rest using my wide array of tools. Whether it's data insights, predictions, or recommendations.
    I'm here to make your experience as smooth and simple as possible.

    Below are some key processes I am trained to help you with:-

     ** Explain the main reasons for churn for a subset of customers of your choice**
        
        - Based on the subset you want to analyze, I will go and ask the ML model how each feature contributes to the churn probability of the customers in the subset.
        - I will then provide you with a excecutive report on the main reasons for churn along with some potential actions you can implement in the short and long term to reduce churn.
        - For a background on how this is done, I use SHAP analysis on the model predictions. This is a way to make the model explain individual contribution of each features to the prediction.
          I will then aggregate this information, transform, process and interpret and extract the required infomration you want to know.
        - Now you need to note that this is not a causal analysis. But with business knowledge and domain expertise you can use this information to make informed decisions.

     ** CLV impact analysis **

        - After churn reason report now you understand main reasons for churn and want to perform a campaign based on the information
        - But before you do this, how great would it be if you could know the net change in CLV if you apply the treatment on the subset of customers.
        - I can help you with this. I will calculate the net effect on CLV if you apply the treatment on the subset of customers.
        - All you have to do tell me:-
            a. Subset of customers you want to analyze
            b. The treatment you want to apply on the subset of customers (eg: decrease revenue_per_minute by 10%)
            c. The cost of the treatment per customer (Every treatment has a cost associated with it. Provide expected cost per customer for a year due to this treatment)
        - I will then calculate the difference in churn due to treatment applied from the model predictions and derive Discounted Cash Flow method to calculate the Customer Lifetime Value (CLV) for 1 year for the customers in the subset.          

    ** Churn Impact Analysis **

        - This is similar to CLV impact analysis but here I will calculate the average churn prediction before and after the treatment.
    
    ** Individual Customer Recommendations **

        - You want to know for a individual customer what are the recommendations to reduce churn probability.
        - For customers with high churn probabilty I can perform counterfactual analysis with the model and provide you with the recommendations to reduce churn probability for individual customers.
        - All you have to do is ask for recommendation with the customer ID
        - How I do this is as below:-
            a. I will get the customer data for the customer ID you provide
            b. I will counterfactual analysis performed on action features for the same customer ID
            c. If you are a customer service rep, I will also provide you with a questionnaire to check with customer on implementing the recommendations.
        - If you are keen on knowing what counterfatual analysis is, is a method used to estimate the causal effect of a treatment or intervention on an individual or group by comparing the observed outcome with a hypothetical scenario where the treatment did not occur. In simpler terms, it involves imagining what would have happened to the same individual or group if they had not received the treatment or intervention.
          I would optimize and identify based on the model what treatment is the best for the customer to reduce churn probability.

    ** Model Stats **

        - If you have any questions about model stats and accuracy, I can provide you with the model stats across test data validation and train data validation.
        - The stats include AUC, F1 Score, Precision, Recall, Lift etc.
    
    ** Visualizations **

        - If you want to see the data in a visual format, I can generate different types of visualizations on the subset of data retrieved from the SQL query.
        - I am still learning this skill but I will try the best I can        

    ** Explorartory Analysis **

        - You want to perform exploratory analysis on the data, I can help you with that.
        - Just tell me what you want to do and I will create SQL query for the same and provide you with the results.

    ** And many more **

    If you want to have an idea about the data and model, you can ask me for an introduction to the data and model.
    It might be ideal to familiarize yourself with the data and model before you start asking me questions.

        - There are many other ways you can make use of me. But this is a short introduction to what I can do for you.
        - For example you can ask what are the top 10 customers with highest churn probability, I can provide you with the list. 
        - Basically you can consider me as your data scientist assistant who can help you with any data related queries you have.
          I will take your question, ask around (to data, models, tools, internet) and provide you with the answer you are looking for as best as I can.
    """
    return response



def intro_to_data(model_config):
    """
    Generates a summary about what the dataset is,what are the models behind,what are the features available for analysis etc.
    Returns
    -------
    str
        A summary on what the dataset is, what are the models behind, what are the features available for analysis etc.
    """

    data_info = model_config['data']['description']
    model_info = model_config['model']['description']
    cat_features = model_config['model']['features']['cat_features']
    num_features = model_config['model']['features']['num_features']
    target = model_config['model']['features']['target']
    prediction = model_config['model']['features']['prediction_column']

    response = (
        f"**Dataset Description:**\n\n"
        f"{data_info}\n\n"
        f"**Dataset Features:**\n"
        f"- **Categorical Features:**\n  - " + "\n  - ".join(cat_features) + "\n"
        f"- **Numerical Features:**\n  - " + "\n  - ".join(num_features) + "\n"
        f"- **Target Feature:** {target}\n"
        f"- **Prediction Feature:** {prediction}\n\n"
        f"**Model Description:**\n\n"
        f"{model_info}"
    )

    return response



def agent_prompt():
    prompt=llm_config['main_agent']['prompt']
    return prompt