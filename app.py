from google.cloud import bigquery
from google.cloud import bigquery_connection_v1 as bq_connection
from google.cloud import bigquery_storage
from abc import ABC
from datetime import datetime
import google.auth
import pandas as pd
from google.cloud.exceptions import NotFound
from google.cloud import aiplatform
from vertexai.generative_models import GenerationConfig
import vertexai
import yaml
import asyncio
from agent import sqlagents
import tabulate
import google.generativeai as genai
import google.ai.generativelanguage as glm
from google.generativeai import caching
import re
import pickle
import json
import xgboost as xgb
from core.modelscorer import ModelScorer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
from agent import taskscheduler,oracle,VisualizeAgent
import numpy as np
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import uuid
# Load configuration files
with open('sql_config.yml', 'r') as f:
    sql_config = yaml.load(f, Loader=yaml.FullLoader)

with open('conf_telchurn.yml', 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

# Initialize Vertex AI and BigQuery
vertexai.init(project=sql_config['bigquery']['project_id'], location=sql_config['bigquery']['region'])
aiplatform.init(project=sql_config['bigquery']['project_id'], location=sql_config['bigquery']['region'])

# Initialize Agents and other components
Agent = sqlagents.Agent
embedder = sqlagents.EmbedderAgent('vertex')
SQLBuilder = sqlagents.BuildSQLAgent('gemini-1.5-flash-001')
SQLValidator = sqlagents.ValidateSQLAgent('gemini-1.5-flash-001')
ResponseAgent = sqlagents.ResponseAgent('gemini-1.5-flash-001')
SQLDebugger = sqlagents.DebugSQLAgent('gemini-1.5-flash-001')
QueryRefiller=sqlagents.QueryRefiller('gemini-1.5-flash-001')



bq_connector = sqlagents.BQConnector(
    project_id=sql_config['bigquery']['project_id'],
    dataset_name=sql_config['bigquery']['dataset_id'],
    region=sql_config['bigquery']['region']
)

task_master = taskscheduler.TaskMaster()
churn_explainer = oracle.ShapOracle()
xgb_scorer = ModelScorer()
visualize_agent = VisualizeAgent.VisualizeAgent()

# Constants from the configuration file
USER_DATABASE = sql_config['bigquery']['dataset_id']
call_await = sql_config['sql_run']['call_await']
num_table_matches = sql_config['sql_run']['num_table_matches']
num_column_matches = sql_config['sql_run']['num_column_matches']
table_similarity_threshold = sql_config['sql_run']['table_similarity_threshold']
column_similarity_threshold = sql_config['sql_run']['column_similarity_threshold']
example_similarity_threshold = sql_config['sql_run']['example_similarity_threshold']
num_sql_matches = sql_config['sql_run']['num_sql_matches']
DEBUGGING_ROUNDS = sql_config['sql_run']['DEBUGGING_ROUNDS']
RUN_DEBUGGER = sql_config['sql_run']['RUN_DEBUGGER']
LLM_VALIDATION = sql_config['sql_run']['LLM_VALIDATION']
EXECUTE_FINAL_SQL = sql_config['sql_run']['EXECUTE_FINAL_SQL']
VECTOR_STORE = sql_config['sql_run']['VECTOR_STORE']
DATA_SOURCE = sql_config['sql_run']['DATA_SOURCE']
KGQ_ENABLED = sql_config['sql_run']['KGQ_ENABLED']

def remove_sql_and_backticks(input_text):
    modified_text = re.sub(r'```|sql', '', input_text)
    modified_text = re.sub(r'\\\\', '', modified_text)
    return modified_text

def generate_sql(user_question: str):
    """
    Generates the SQL query based on the user question
    Use this function to create a SQL query to retrive any data user have asked for or to create an answer to user question.

    Parameters
    ----------
        user_question : str
            the user question
    Returns
    -------
        str
            the result sql query generated
    """
    st.markdown("---------------------------------------------")
    st.info("Generating SQL to answer the user question...")
    embedded_question = embedder.create(user_question)
    AUDIT_TEXT = f"\nUser Question: {user_question}\nUser Database: {USER_DATABASE}"
    process_step = "\n\nGet Exact Match: "

    if KGQ_ENABLED:
        exact_sql_history = bq_connector.getExactMatches(user_question)
    else:
        exact_sql_history = None

    if exact_sql_history is not None:
        final_sql = exact_sql_history
        invalid_response = False
        AUDIT_TEXT += "\nExact match found! Retrieving SQL query from cache."
    else:
        AUDIT_TEXT += process_step + "\nNo exact match found, retrieving schema and known good queries..."
        process_step = "\n\nGet Similar Match: "
        similar_sql = bq_connector.getSimilarMatches('example', USER_DATABASE, embedded_question, num_sql_matches, example_similarity_threshold) if KGQ_ENABLED else "No similar SQLs provided..."
        process_step = "\n\nGet Table and Column Schema: "
        table_matches = bq_connector.getSimilarMatches('table', USER_DATABASE, embedded_question, num_table_matches, table_similarity_threshold)
        column_matches = bq_connector.getSimilarMatches('column', USER_DATABASE, embedded_question, num_column_matches, column_similarity_threshold)
        AUDIT_TEXT += process_step + f"\nRetrieved Tables: \n{table_matches}\n\nRetrieved Columns: \n{column_matches}\n\nRetrieved Known Good Queries: \n{similar_sql}"

        if table_matches or column_matches:
            process_step = "\n\nBuild SQL: "
            generated_sql = SQLBuilder.build_sql(DATA_SOURCE, user_question, table_matches, column_matches, similar_sql)
            final_sql = generated_sql
            AUDIT_TEXT += process_step + f"\nGenerated SQL: {generated_sql}"

            if 'unrelated_answer' in generated_sql:
                invalid_response = True
                AUDIT_TEXT += "\nInvalid Response: "
            else:
                invalid_response = False
                if RUN_DEBUGGER:
                    generated_sql, invalid_response, AUDIT_TEXT = SQLDebugger.start_debugger(
                        DATA_SOURCE, generated_sql, user_question, SQLValidator,
                        table_matches, column_matches, AUDIT_TEXT, sql_config['bigquery']['project_id'],
                        similar_sql, DEBUGGING_ROUNDS, LLM_VALIDATION
                    )
                final_sql = generated_sql
                AUDIT_TEXT += f"\nFinal SQL after Debugger: \n{final_sql}"
        else:
            invalid_response = True
            AUDIT_TEXT += "\nNo tables found in the Vector DB. The question cannot be answered with the provided data source!"
    
    final_sql = final_sql.replace("\n", " ")
    final_sql = re.sub(r'```|sql', '', final_sql)
    final_sql = re.sub(r'\\\\', '', final_sql)
    st.write(f"Generated SQL: {final_sql}")
    return final_sql

def execute_sql(user_question: str, sql_generated: str, output_mode: str = 'json'):
    """
    Executes the provided SQL query using the sql_agent and returns the result as a markdown table or json object.
    Use this only for cases where the answer to user question is directly available and further tools or processing is not required.

    Parameters
    ----------
    user_question : str
        The user's question that the SQL query is intended to answer.

    sql_generated : str
        The SQL query to be executed.

    output_mode : str
        The format in which to return the result. Can be 'json' for a json object or 'table' for a markdown table.
        Default is 'json'.
        Choose 'table' ONLY if user have asked for a table, 'json' for all other cases.
        For eg: If the user question was about main reasons for churn overall or return a chunk of data as table use table mode
                But if user asks list of top 10 customers with highest churn probability, use json mode
    Returns
    -------
    dict or str
        The result of the SQL query. If output_mode is 'json', the result is a dictionary. If output_mode is 'table', 
        the result is a string formatted as a markdown table.
    """
    st.markdown("---------------------------------------------")
    st.info(f"Executing SQL in {output_mode} mode...")
    sql_generated = sql_generated.replace("\n", " ").replace("\\", "")
    bq_df = bq_connector.retrieve_df(sql_generated)
    if output_mode == 'json':
        response = bq_df.to_json(orient='records')
    else:
        bq_df=pd.DataFrame(bq_df)
        st.dataframe(bq_df)

        if bq_df.shape[0] <20:
            response = tabulate.tabulate(bq_df, headers='keys', tablefmt='pipe', showindex='never')
            response += "\n\nAbove table answers user question. Please provide a textual summary of this data to answer users question."
        else:
            response = "Data is too large to pass as text or create textual summary. Please explain to the user that the answer to their question is displayed as a table above."
    return response

def subset_churn_contribution_analysis(user_question: str, sql_generated: str):
    """
    Performs a churn contribution analysis on a subset of data.

    This function executes SQL passed to retrieve a subset of data, then uses a model to predict on this data.
    It then calculates the average prediction before and after a treatment, and returns this information in a string.
    Use this function to explain the impact of a treatment on a subset of data.
    Some use cases would be:-
        1. To understand impact of churn by changing feature value to a new value
        2. To understand impact of churn by decreasing revenue of a subset of customers
    In order to use this tool generate_sql tool must be ran first and sql query should be generated. 

    Parameters
    ----------
    user_question : str
        The user's question that the SQL query is intended to answer.
    sql_generated : str
        The SQL query to be executed.

    NOTE:
    ----
    SQL Query generated should return all the columns from dataset after required adjustements. USE Select * always

    Returns
    -------
    str
        A string containing the average churn prediction before and after the treatment.
    Notes
    -----
        - The output from this is the report. You have to display this report to the user as it is. DO NOT MODIFY THE OUTPUT.                
    
    """
    st.markdown("---------------------------------------------")
    st.info("Performing subset Churn analysis...")
    sql_generated = remove_sql_and_backticks(sql_generated).replace("\n", " ").replace("\\", "")
    df = bq_connector.retrieve_df(sql_generated)
    df2 = xgb_scorer.model_predictor(df.copy())
    response = f"The average churn prediction after the treatment changed from {round(100 * df2['prediction'].mean(), 2)}% to {round(100 * df2['new_prediction'].mean())}%."
    return response


def subset_clv_analysis(user_question:str, sql_generated:str,treatment_cost:float=0.0):

    """
    Performs net effect on CLV analysis on a subset of data.

    This function executes SQL passed to retrieve a subset of data, then net CLV analysis on this data.
    It then calculates the CLV impact made by the changes and returns this information in a string.
    Use this function to explain the impact or change in CLV if a treatment is applied on a subset of data.
    Some use cases would be:-
        1. To understand impact of lifetime value by changing feature value to a new value
        2. To understand impact in terms of $ value by decreasing revenue of a subset of customers
    In order to use this tool generate_sql tool must be ran first and sql query should be generated. 

    Parameters
    ----------
    user_question : str
        The user's question that the SQL query is intended to answer.
    sql_generated : str
        The SQL query to be executed.
    treatment_cost : float
        The cost of the treatment per customer applied to the subset of data. Default is 0.0.

    Returns
    -------
    str
        A string containing the the CLV impact made by the changes/treatment.

    Notes
    -----
        - The output from this is the report. You have to display this report to the user as it is. DO NOT MODIFY THE OUTPUT.                
    """
    st.markdown("---------------------------------------------")
    st.info("CLV Analysis Tool is running")
    # Execute the SQL query
    sql_generated=remove_sql_and_backticks(sql_generated)
    sql_generated=sql_generated.replace("\n", " ")
    sql_generated=sql_generated.replace("\\", "")

    ##Get the subset data from bigquery
    df = bq_connector.retrieve_df(sql_generated)
    df['current_revenue']=df['monthlyrevenue']*12
    #df['current_clv']=(df['monthlyrevenue']*12*df['prediction'])/(1+0.09-df['prediction'])
    df['current_clv'] = (df['monthlyrevenue'] * 12 * (1 - df['prediction'])) / (0.09 + df['prediction'])

    #print(df.shape)
    ##Get the model prediction on this data
    df2=xgb_scorer.model_predictor(df)
    #print(df.shape)

    df2['treatment_clv']=((df['monthlyrevenue']*12-treatment_cost)*(1-df['new_prediction']))/(0.09+df['new_prediction'])
    
    response = (
        "CLV Impact Analysis Report:\n"
        "I have used the Discounted Cash Flow method to calculate the Customer Lifetime Value (CLV) for 1 year for the customers in the subset.\n\n"
        "Assumptions:\n"
        "- Discount rate: 9%\n"
        "- Model churn prediction is the probability of churn in 1 year\n"
        "- Treatment cost per customer is: ${}\n\n"
        "Results:\n"
        "- The average CLV before the treatment is ${}.\n"
        "- The average CLV after the treatment is ${}.\n"
        "- The average churn predicted before the treatment is {}%.\n"
        "- The average churn predicted after the treatment is {}%.\n"
        "- The average CLV impact made by the treatment is ${} per customer.\n"
        "- The number of customers in the subset is {}.\n"
        "- Hence, according to the model, the treatment would generate ${} in total revenue.\n\n"
        "Note that the above results are based on the model predictions and assumptions made. This is a simplified version of the actual CLV calculation.\n"
        "You can use this information to understand the impact of the treatment on the subset of customers and make informed decisions with more detailed analysis."
    ).format(
        treatment_cost,
        round(df2['current_clv'].mean(), 2),
        round(df2['treatment_clv'].mean(), 2),
        round(100 * df2['prediction'].mean(), 2),
        round(100 * df2['new_prediction'].mean(), 2),
        round(df2['treatment_clv'].mean() - df2['current_clv'].mean(), 2),
        df2.shape[0],
        round(((round(df2['treatment_clv'].mean() - df2['current_clv'].mean(), 2)) * df2.shape[0]), 2)
    )
    print(response)
    #st.markdown(response)
    return response

def model_stat(user_question:str):

    """
    Returns the Model Stats to user.
    On any question related to mdoel accuracy this tool can be used to retrive the answer.

    Parameters:
    - user_question (str): The user's question about the data.

    Returns:
    - model_stats: str
        - The model stats across test data validation and train data validation
        - The stats include AUC, F1 Score, Precision, Recall, Lift etc.

    Note:
    - Unless specified by the user always use test data validation stats for model stats explanation
    """
    st.markdown("---------------------------------------------")
    st.info("Model Stats Tool is running")
    note= "\nNote:\n- Unless specified by the user always use test data validation stats for model stats explanation"
    with open("results\\tel_churn\\model_stats.txt", 'r') as file:
        model_stats = file.read()
    model_stats+=note
    return model_stats


def generate_visualizations(user_question: str, generated_sql: str):
    """
    Creates different types of visualizations on the subset of data retrieved from the SQL query.
    Use this tool if the customer asks for a plot or visualization.
    The tool generates Google Charts code for displaying charts on a web application by returning the HTML code for embedding in Streamlit.
    Generates two charts with elements "chart-div" and "chart-div-1".

    Parameters:
    - user_question (str): The user's question about the data.
    - generated_sql (str): The SQL query corresponding to the user's question.

    Returns:
    - Tuple containing HTML strings for embedding the visualizations.
    - Generates two charts with elements "chart-div" and "chart-div-1".
    """

    st.markdown("---------------------------------------------")
    st.info("Generating Visualizations...")
    generated_sql = generated_sql.replace("\n", " ").replace("\\", "")
    sql_results_json = bq_connector.retrieve_df(generated_sql).to_json(orient='records')
    
    # Generate unique element IDs
    chart_div_1_id = "chart_div_" + str(uuid.uuid4()).replace("-", "")

    # Generate the visualizations using VisualizeAgent
    charts_js = visualize_agent.generate_charts(user_question, generated_sql, sql_results_json)
    if charts_js is not None:
        # Ensure the JavaScript code does not have nested calls
        chart_js_1 = charts_js["chart_div"].replace("chart_div", chart_div_1_id).replace("new google.charts.BarChart", "new google.visualization.BarChart")

        # Create the full HTML content for the first chart
        chart_html_1 = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Google Chart 1</title>
            <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
            <script type="text/javascript">
                google.charts.load('current', {{packages: ['corechart']}});
                google.charts.setOnLoadCallback(drawChart);
                function drawChart() {{
                    {chart_js_1}
                }}
            </script>
        </head>
        <body>
            <div id="{chart_div_1_id}" style="width: 600px; height: 300px;"></div>
        </body>
        </html>
        '''

        # Use Streamlit's components to embed raw HTML
        st.markdown("---------------------------------------------")
        st.markdown("Here is the visualization requested:")
        st.components.v1.html(chart_html_1, height=350)
        return chart_html_1
    else:
        return f"Sorry. Unexpected error due to invalid sql query on data retrieval"


def question_reformer(user_question:str):
    """
    Reformulates the user question to make it more understandable and answerable.
    Use this tool to reformulate the user question to make it more clear and concise.
    It is important to use this tool especially before using subset shap summary tool.

    Parameters
    ----------
    user_question : str
        The user's question that the SQL query is intended to answer.

    Returns
    -------
    str
        A string containing the reformulated user questions
    """
    st.markdown("---------------------------------------------")
    st.info("Question Reformer is running")
    reformed_question=task_master.ask_taskmaster(user_question)
    #reformed_question=response.candidates[0].content.parts[0].text
    print(f"Reformed Question: {reformed_question}")

    return reformed_question

def subset_shap_summary(customer_data_sql_query:str,shap_data_sql_query:str,user_question:str):
    """
        Calculates the SHAP summary of customers from the customer data query and SHAP feature contribution data for same subset.
        This can be used to identify patterns, top churn contributors and feature importance for the subset of data.
        Use this tool always for understanding the main reasons for churn for a subset of customers.
        Use this tool for identifying top churn contributors customers for any subset of data.
        Use this tool to identify main reasons for churn for a subset of customers user wants to analyze.


        Parameters:
        ----------
        customer_data_sql_query : str
            The SQL query to retrieve the customer data for a subset of customers as asked by user.
            Should return all columns from customer data
            Use select * in the query. Don't limit the columns in the query. 
        shap_data_sql_query : str
            The SQL query to retrieve the SHAP feature contribution data for the same subset of customers as asked by user. 
            Should return all columns from shap value data
            Use select * in the query. Don't limit the columns in the query. 
        user_question:str
            The user's question that the SQL query is intended to answer.

        Returns:
        -------
        str
            A report on reasons for churn for the user question. You have to display this report to the user.
        
        Notes
        -----
        - It is important to have both the customer data query and SHAP data query for the same subset of customers to be passed as input.
        - The output from this is the report. You have to display this report to the user as it is. DO NOT MODIFY THE OUTPUT.
        - Add a final note after the report of how nd why the recommended actions should be tested with churn adn clv impact analysis.
"""
    st.markdown("---------------------------------------------")
    st.info("Subset Churn Summary Tool is running")
    customer_data_sql_query=remove_sql_and_backticks(customer_data_sql_query)
    customer_data_sql_query=customer_data_sql_query.replace("\\", "")
    shap_data_sql_query=remove_sql_and_backticks(shap_data_sql_query)
    shap_data_sql_query=shap_data_sql_query.replace("\\", "")

    ##Validate the query have select * enabled
    customer_data_sql_query_updated=QueryRefiller.check(generated_sql=customer_data_sql_query)
    print(f"Updated custommer sql query:{customer_data_sql_query_updated}")

    shap_data_sql_query_updated=QueryRefiller.check(generated_sql=shap_data_sql_query)
    print(f"Updated shap sql query:{shap_data_sql_query_updated}")


    ##Get the subset data from bigquery
    df_data=bq_connector.retrieve_df(customer_data_sql_query_updated)
    df_shap_data=bq_connector.retrieve_df(shap_data_sql_query_updated)

    ##Order both dataframes by customerid
    df_data=df_data.sort_values(by='customerid')
    df_shap_data=df_shap_data.sort_values(by='customerid')

    def sigmoid(x):
        """ Sigmoid function to convert log-odds to probabilities. """
        return 1 / (1 + np.exp(-x))
    
    base_value=1.0    
    base_probability = sigmoid(base_value)
    results = []
    feature_importances = {}
    # Process each feature
    ##Remove the SHAP prefix from SHAP data columns
    df_shap_data.columns = df_shap_data.columns.str.replace('shapvalue_', '')
    common_columns = df_data.columns.intersection(df_shap_data.columns)
    #Remove if column customerid exists
    columns_to_drop = ['customerid', 'churn']

    for col in columns_to_drop:
        if col in common_columns:
            common_columns = common_columns.drop(col)

    print(f"data shape:{df_data.shape}")
    print(f"shap data shape:{df_shap_data.shape}")
    # Calculate feature importances
    for feature in common_columns:
        feature_shap_values = df_shap_data[feature]
        feature_importances[feature] = np.mean(np.abs(feature_shap_values))


    importance_df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
    print(importance_df.head())
    importance_df.sort_values('Importance', ascending=False, inplace=True)
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    importance_ranks = importance_df.set_index('Feature')['Rank'].to_dict()
    #print(2)

    for feature in common_columns:
        feature_values = df_data[feature]
        feature_shap_values = df_shap_data[feature]
        df = pd.DataFrame({feature: feature_values, 'SHAP Value': feature_shap_values})
        numeric_features = df_data.select_dtypes(include=['number']).columns

        if feature in numeric_features:
            df['Group'] = pd.qcut(df[feature], 5, duplicates='drop')
        else:
            df['Group'] = df[feature]

        group_avg = df.groupby('Group', observed=True).agg({
            'SHAP Value': 'mean',
            feature: 'count'
        }).reset_index()

        group_avg.rename(columns={feature: 'Count'}, inplace=True)
        group_avg['Adjusted Probability'] = sigmoid(base_value + group_avg['SHAP Value'])
        group_avg['Probability Change (%)'] = (group_avg['Adjusted Probability'] - base_probability) * 100
        group_avg['Feature'] = feature
        group_avg['Feature Importance'] = feature_importances[feature]
        group_avg['Importance Rank'] = importance_ranks[feature]
        results.append(group_avg)
    
    result_df = pd.concat(results, ignore_index=True)
    ##Count of groups should be atleast 50 records - If not remove the group
    result_df = result_df[result_df['Count'] >= 50]
    result_df.sort_values(['Importance Rank', 'Probability Change (%)'], ascending=[True, False], inplace=True)
    result_df=result_df[result_df['Importance Rank']<=10]
    #print(tabulate.tabulate(result_df[['Feature','Group','Probability Change (%)','SHAP Value','Importance Rank']].head(30), headers='keys', tablefmt='pipe', showindex='never'))

    report=churn_explainer.ask_churnoracle(shap_summary=f"""The SHAP summary from the model is: 
    {tabulate.tabulate(result_df[['Feature','Group','Probability Change (%)','SHAP Value','Importance Rank']], headers='keys', tablefmt='pipe', showindex='never')}""",
    user_question=user_question)
    #print(report)
    return report

def customer_recommendations(user_question:str, customer_data_query:str,counterfatual_data_query:str):
    """
    Generates recommendations to reduce churn probability for individual customers.
    Use this tool to generate customer recommendations for individual customers with high churn probability.
    Need to use questions_reformer tool before using this tool to generate separate sql queries for customer data and counterfactual data.

    Parameters
    ----------
    user_question : str
        The user's question that the SQL query is intended to answer.
    customer_data_query : str
        The query to get the data for the customer for whom the recommendations are to be generated.
    counterfatual_data_query : str
        The query to get the counterfactual recommendations for the customer.

    Returns
    -------
    str
        A report on recommended actions to reduce the customer churn. You have to display this report to the user.
    """
    st.markdown("---------------------------------------------")
    st.info("Customer Recommendations Tool is running")
    # Execute the SQL query
    customer_data_query=remove_sql_and_backticks(customer_data_query)
    customer_data_query=customer_data_query.replace("\\", "")

    counterfatual_data_query=remove_sql_and_backticks(counterfatual_data_query)
    counterfatual_data_query=counterfatual_data_query.replace("\\", "")

    ##Get the subset data from bigquery
    counterfactuals = bq_connector.retrieve_df(counterfatual_data_query)
    customer_data = bq_connector.retrieve_df(customer_data_query)

    #print(counterfactuals.shape[0])
    #print(len(counterfactuals.to_json(orient='records')))

    if customer_data.shape[0]==0:
        response=f"Invalid customer ID"
    else:
        response=churn_explainer.ask_recommendation(user_question=user_question,
                                        customer=customer_data.to_json(orient='records'),
                                        counterfactual=counterfactuals.to_json(orient='records'))
    return response

def sample_questions():

    response = """ You can ask me about anything about the process you are looking for.\n

In this scenario, I'm provided with dataset and ML model of a telecom company. I can help answer any question you have based on your role or generic.

Here are some sample questions you can ask me if you are a :-

1. **Retention Manager**:\n
    a. What are the main reasons for churn for customers with equipment age more than 600 days?\n
    b. What is the net effect on CLV if we decrease the revenue_per_minute by 10% for customers with churn probability more than 0.5? Assume the cost of treatment is $10 per customer.\n
    c. What is the average age of customers who have higher churn because of revenue_per_minute?\n

2. **Customer Service Rep**:\n
    a. What are the recommendations to reduce churn probability for customer with customer_id 3334558?\n
    b. Who is customer_id 3334558?\n 

3. **Data Analyst**:\n
    a. What are the top 10 customers with highest churn probability?\n
    b. What is the churn probability distribution for customers with revenue_per_minute more than 0.5?\n
    c. Create a vizualization of churn probability distribution for customers with revenue_per_minute more than 0.5.\n
    d. How many customers with children and aged under 50 have current equipment age more than 600 days?\n

4. **Anyone**:\n
    a. What are the stats of the model?\n
    b. What is the AUC and F1 score of the model?\n
    c. Display the vigintile distribution of model\n

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

        - There are many other ways you can make use of me. But this is a short introduction to what I can do for you.
        - For example you can ask what are the top 10 customers with highest churn probability, I can provide you with the list. 
        - Basically you can consider me as your data scientist assistant who can help you with any data related queries you have.
          I will take your question, ask around (to data, models, tools, internet) and provide you with the answer you are looking for as best as I can.
    """
    return response


cot_prompts = """
You are an intelligent agent named MLi that answers user questions related to telecom churn analysis.
You have access to multitude of tools like:

  - question_reformer: To reformulate the user question to make it more clear and concise or to split into different tasks
  - generate_sql: To generate SQL query to answer user question 
  - execute_sql: To execute the SQL query and provide a textual summary of the data for simple tasks
  - subset_churn_contribution_analysis: To perform subset churn contribution analysis on the subset of data retrieved using the SQL query generated
  - subset_clv_analysis: To perform net effect on CLV or CLV impact analysis based on treatments applied on the subset of data
  - subset_shap_summary: To calculate the SHAP summary of customers from the customer data query and SHAP feature contribution data for same subset
  - customer_recommendations: To generate recommendations to reduce churn probability for individual customers
  - model_stat: To answer any question user have about model stats and accuracy
  - generate_visualizations: To create different types of visualizations on the subset of data retrieved from the SQL query


When you use the following tools, you should display the response exactly as from the tool. No modification:
    - subset_churn_contribution_analysis
    - subset_shap_summary
    - subset_clv_analysis
    - customer_recommendations

**Guidelines:**
- Explain to user what all you can do. Do not mention what tools you have but mention what all you can do and what all user can ask you.
- Always understand the user question and it's contents clearly. If it is not clear, ask for more details.
- Always use question_reformer tool first for any task
- Question Reformer have to ran before sql generation tool. Failure to do this will result in fatal error
- Reformed question should be used to generate SQL query
- If the reponse from tool states to display the message exactly as it is to the user, then display the message as it is to the user.
- If the user question is about main reasons for churn, always use subset shap summary tool to get the top churn contributors.
- When using subset shap summary tool you should return the ouput from it exactly as it is to the user.
- If the user question is about recommended actions to reduce churn for a single specific customer, always use customer_recommendations tool to get the recommendations.
- When using customer_recommendations tool you should return the ouput from it exactly as it is to the user.
- If multiple tools are needed to answer the user query, use task segmentation to split the tasks and execute them in order.
- If mutiple tools are needed to answer user query and it takes same sql generated, reuse the same sql generated for all tools instead of generating new sql for each tool.
- When using execute_sql tool, you should not display the data. You should provide a textual summary of the data.
- If you have answered the user query, always ask the user if they have any more questions or if they need any more help.
- If you already have answer to the user query, you can use the same answer to answer the user query again if the user asks the same question again.
- If repsonse from excecute_sql is long, please provide an answer to the user in textual format instead of displaying the data based on their query.


The welcome message should be as below:

    Hello! I'm MLy 🤖 your friendly AI assistant

    I'm here to help you interact effortlessly with our powerful machine learning models.
    In this scenario, I'm provided with dataset and ML model of a telecom company. I can help answer any question you have based on your role or generic.
    Just ask me what you need in plain English, and I'll take care of the rest using my wide array of tools. Whether it's data insights, predictions, or recommendations.
    I'm here to make your experience as smooth and simple as possible.

    PS: If you want a walkthrough on how I may be of service to you, please let me know.

If the user asks for a walkthrough:-
    - Provide a short introduction to all the tools you have and what infomration it helps with
    - It should be short and concise
    
    """

gen_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-001",system_instruction=cot_prompts,
    tools=[generate_sql,execute_sql,subset_churn_contribution_analysis,subset_clv_analysis,generate_visualizations
           ,subset_shap_summary,question_reformer,customer_recommendations,model_stat],
    generation_config={"temperature":0.3})
#chat = gen_model.start_chat(enable_automatic_function_calling=True)

# Gemini uses 'model' for assistant; Streamlit uses 'assistant'
def role_to_streamlit(role):
  if role == "model":
    return "assistant"
  else:
    return role



def add_sidebar_elements():

    
    
    # linkedin_url = "https://www.linkedin.com/in/david-babu-15047096/"
    # #buy_me_a_coffee_url = "https://www.buymeacoffee.com/yourusername"

    # linkedin_icon_html = f"""
    #     <a href="{linkedin_url}" target="_blank">
    #         <img src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Linkedin_icon.svg" alt="LinkedIn" style="width: 30px; height: 30px; margin: 10px 0;">
    #     </a>
    # """

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

    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)  # Adding space at the top if needed
        st.markdown(icons_html, unsafe_allow_html=True)

        

# Function to display chat history
def display_chat_history(chat_history):
    for message in chat_history:
        role = role_to_streamlit(message.role)
        parts = message.parts
        
        # Display message parts based on their type
        for part in parts:
            if "text" in part:
                if part.text and part.text.strip():
                    with st.chat_message(role):
                        st.write(part.text)

st.markdown("<h2 style='text-align: center;'>Meet MLi 🤖: Your ML Model Whisperer</h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
with st.sidebar.expander("Click here for a short introduction to know what I can do for you if you are new to me"):
    st.markdown(walkthrough())
st.sidebar.markdown("<br>", unsafe_allow_html=True)
with st.sidebar.expander("Here are some sample questions if you want to know what you can ask me"):
    st.markdown(sample_questions())
add_sidebar_elements()

# Initialize chat session in session state
if "chat" not in st.session_state:
    st.session_state.chat = gen_model.start_chat(enable_automatic_function_calling=True)

#tab1, tab2, tab3 = st.tabs(["Chatbot", "QA Chatbot", "chk"])


# Display chat messages from history above the current input box
display_chat_history(st.session_state.chat.history)

# Accept user's next message, add to context, resubmit context to Gemini
if prompt := st.chat_input("I possess a well of knowledge. What would you like to know?"):
    # Display user's last message
    st.chat_message("user").markdown(prompt)
    
    with st.spinner("Processing..."):
        # Send user entry to Gemini and get the response
        response = st.session_state.chat.send_message(prompt)
    
    # Add model's response to chat history and display it
    with st.chat_message("assistant"):
        st.write(response.candidates[0].content.parts[0].text)
# with tab2:
#   st.write(st.session_state.chat.history)
  
# with tab3:
#   for message in st.session_state.chat.history:
#         st.write(message.role)
#         st.write(message.parts[0].text)
# # conversation
# for message in st.session_state.chat_history:
#     if isinstance(message, AIMessage):
#         with st.chat_message("AI"):
#             st.write(message)
#     elif isinstance(message, HumanMessage):
#         with st.chat_message("Human"):
#             st.write(message)


