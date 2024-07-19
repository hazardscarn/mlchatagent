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
from google.cloud import secretmanager
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import Dict, Text
import time
import ast
from utils import walkthrough,access_secret_version,sample_questions,normalize_string,remove_sql_and_backticks


#GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
project_id = "mlchatagent-429005"
secret_id = "GOOGLE_API_KEY"

# Access the secret
GOOGLE_API_KEY = access_secret_version(project_id, secret_id)



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
with open(model_config['model']['shap_base_value'], "r") as file:
    base_value = float(file.read().strip())


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
    
    st.markdown("--------------------------------------📥 *Generating Query* 📥--------------------------------------")
    intermediate_steps = []
    normalized_question = normalize_string(user_question)
    try:
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
            intermediate_steps.append({
                'step': 'Exact Match',
                'details': f"Exact Match Found: {final_sql}"
            })
        else:
            AUDIT_TEXT += process_step + "\nNo exact match found, retrieving schema and known good queries..."
            process_step = "\n\nGet Similar Match: "
            similar_sql = bq_connector.getSimilarMatches('example', USER_DATABASE, embedded_question, num_sql_matches, example_similarity_threshold) if KGQ_ENABLED else "No similar SQLs provided..."
            process_step = "\n\nGet Table and Column Schema: "
            table_matches = bq_connector.getSimilarMatches('table', USER_DATABASE, embedded_question, num_table_matches, table_similarity_threshold)
            column_matches = bq_connector.getSimilarMatches('column', USER_DATABASE, embedded_question, num_column_matches, column_similarity_threshold)
            AUDIT_TEXT += process_step + f"\nRetrieved Tables: \n{table_matches}\n\nRetrieved Columns: \n{column_matches}\n\nRetrieved Known Good Queries: \n{similar_sql}"
            # Format the details for better readability
            formatted_details = (
                f"**Similar SQL:**\n```\n{similar_sql}\n```\n\n"
                f"**Tables:**\n{table_matches}\n\n"
                f"**Columns:**\n{column_matches}"
            )
            intermediate_steps.append({
                'step': 'Similar Match',
                'details': formatted_details
            })

            if table_matches or column_matches:
                process_step = "\n\nBuild SQL: "
                generated_sql = SQLBuilder.build_sql(DATA_SOURCE, user_question, table_matches, column_matches, similar_sql)
                final_sql = generated_sql
                AUDIT_TEXT += process_step + f"\nGenerated SQL: {generated_sql}"
                intermediate_steps.append({
                    'step': 'Build SQL',
                    'details': f"Generated SQL: {generated_sql}"
                })

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
                    intermediate_steps.append({
                        'step': 'Debugger',
                        'details': f"Final SQL after Debugger: {final_sql}"
                    })
            else:
                invalid_response = True
                AUDIT_TEXT += "\nNo tables found in the Vector DB. The question cannot be answered with the provided data source!"
                intermediate_steps.append({
                    'step': 'No Tables Found',
                    'details': "No tables found in the Vector DB."
                })
        final_sql = final_sql.replace("\n", " ")
        final_sql = re.sub(r'```|sql', '', final_sql)
        final_sql = re.sub(r'\\\\', '', final_sql)
        #st.write(f"Generated SQL: {final_sql}")

        # Save intermediate result
        if 'intermediate_results' not in st.session_state:
            st.session_state.intermediate_results = {}
        if normalized_question not in st.session_state.intermediate_results:
            st.session_state.intermediate_results[normalized_question] = []
        st.session_state.intermediate_results[normalized_question].append({
            "tool": "generate_sql",
            "sql_generated": final_sql,
            "intermediate_steps": intermediate_steps
        })

        return final_sql
    except Exception as e:
        return str(e)

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
    st.markdown("--------------------------------------⚙️ *Executing Query* ⚙️--------------------------------------")

    try:
        normalized_question = normalize_string(user_question)
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
                response = f"""Explain to the user that the answer to their question is displayed as a table above.
                            Data is too large to create a textual summary though. If user needs more insights, please ask for more specific question."""
            
        # Save intermediate result
        if 'intermediate_results' not in st.session_state:
            st.session_state.intermediate_results = {}
        if normalized_question not in st.session_state.intermediate_results:
            st.session_state.intermediate_results[normalized_question] = []
        st.session_state.intermediate_results[normalized_question].append({
            "tool": "execute_sql",
            "sql_generated": sql_generated,
            "result": bq_df
        })
    except Exception as e:
        return str(e)


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
    st.markdown("--------------------------------------📊 *Subset Churn Impact Tool* 📊--------------------------------------")

    try:
        sql_generated = remove_sql_and_backticks(sql_generated).replace("\n", " ").replace("\\", "")
        df = bq_connector.retrieve_df(sql_generated)
        df = df.reset_index(drop=True)
        df2 = xgb_scorer.model_predictor(df.copy())
        response = f"The average churn prediction after the treatment changed from {round(100 * df2['prediction'].mean(), 2)}% to {round(100 * df2['new_prediction'].mean())}%."
        return response
    except Exception as e:
        return str(e)

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
    st.markdown("--------------------------------------💡 *CLV Impact Analysis Tool* 💡--------------------------------------")

    try:
        # Execute the SQL query
        sql_generated=remove_sql_and_backticks(sql_generated)
        sql_generated=sql_generated.replace("\n", " ")
        sql_generated=sql_generated.replace("\\", "")

        ##Get the subset data from bigquery
        df = bq_connector.retrieve_df(sql_generated)
        df=df.reset_index(drop=True)
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
    except Exception as e:
        return str(e)

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
    st.markdown("--------------------------------------📈 *Model Stats Tool* 📈--------------------------------------")

    try:
        note= "\nNote:\n- Unless specified by the user always use test data validation stats for model stats explanation"
        with open(model_config['model']['model_stats'], 'r') as file:
            model_stats = file.read()
        model_stats+=note
        return model_stats
    except Exception as e:
        return str(e)

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

    st.markdown("--------------------------------------📉 *Visualization Tool* 📉--------------------------------------")

    try:
        normalized_question = normalize_string(user_question)
        generated_sql = generated_sql.replace("\n", " ").replace("\\", "")
        sql_results = bq_connector.retrieve_df(generated_sql)
        sql_results_json=sql_results.to_json(orient='records')

        ###Adding conditions to prevent full dataset going into visualization agent
        if sql_results.shape[0] == 0:
            return f"Sorry. Unexpected error due to invalid sql query on data retrieval"
        elif sql_results.shape[0] > 1000:
            return f"""Sorry. Unexpected error due to large data size. Please try with a smaller subset of data.
            If you find the query to be incorrect, please rephrase the question and try again."""
        else:
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

                # Save intermediate result
                if 'intermediate_results' not in st.session_state:
                    st.session_state.intermediate_results = {}
                if normalized_question not in st.session_state.intermediate_results:
                    st.session_state.intermediate_results[normalized_question] = []
                st.session_state.intermediate_results[normalized_question].append({
                    "tool": "generate_visualizations",
                    "sql_generated": generated_sql,
                    "result": chart_html_1
                })

                return chart_html_1
            else:
                return f"Sorry. Unexpected error due to invalid sql query on data retrieval"
    except Exception as e:
        return str(e)


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
    st.markdown("--------------------------------------🔄 *Processing Input* 🔄--------------------------------------")
    try:
        reformed_question=task_master.ask_taskmaster(user_question)
        #reformed_question=response.candidates[0].content.parts[0].text
        print(f"Reformed Question: {reformed_question}")
        return reformed_question
    except Exception as e:
        return str(e)

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
    st.markdown("--------------------------------------🔍 *Subset Churn Analysis Tool* 🔍--------------------------------------")

    try:
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

        ##Addtional check to ensure both dataframes have same number of records
        df_shap_data=pd.merge(df_shap_data,df_data[['customerid']],on='customerid',how='inner')

        def sigmoid(x):
            """ Sigmoid function to convert log-odds to probabilities. """
            return 1 / (1 + np.exp(-x))
        
        print(base_value)
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
        importance_df.sort_values('Importance', ascending=False, inplace=True)
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        importance_ranks = importance_df.set_index('Feature')['Rank'].to_dict()
        #print(2)

        if df_data.shape[0]>10000:
            subset_levels=10
        elif df_data.shape[0]>5000:
            subset_levels=5
        elif df_data.shape[0]>1000:
            subset_levels=3
        else:
            subset_levels=2


        ##Making sure index is reset for both dataframes
        df_shap_data=df_shap_data.reset_index(drop=True)
        df_data=df_data.reset_index(drop=True)
        
        for feature in common_columns:
            feature_values = df_data[feature]
            feature_shap_values = df_shap_data[feature]
            df = pd.DataFrame({feature: feature_values, 'SHAP Value': feature_shap_values})
            numeric_features = df_data.select_dtypes(include=['number']).columns

            
            if feature in numeric_features:
                # Equal sized bins take out SHAP patterns as we get in SHAP PDP
                #df['Group'] = pd.qcut(df[feature], subset_levels, duplicates='drop')
                # Equal length buckets are used instead
                min_val = df[feature].min()
                max_val = df[feature].max()
                range_width = max_val - min_val
                bin_edges = np.linspace(min_val - 0.01 * range_width, max_val + 0.01 * range_width, subset_levels + 1)
                #print(feature)
                df['Group'] = pd.cut(df[feature], subset_levels)
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
        print(result_df.shape)

        if df_data.shape[0] > 1000:
            ##Count of groups should be atleast 50 records - If not remove the group
            result_df = result_df[result_df['Count'] >= 50]
            result_df.sort_values(['Importance Rank','SHAP Value', 'Probability Change (%)'], ascending=[True,False, False], inplace=True)
               
        copy=result_df[['Feature','Group','SHAP Value','Adjusted Probability','Probability Change (%)','Feature Importance','Importance Rank','Count']].copy()
        # result_df=result_df[result_df['Importance Rank']<=10]
        #result_df = result_df.head(100)
        #print(tabulate.tabulate(result_df[['Feature','Group','Probability Change (%)','SHAP Value','Importance Rank']].head(30), headers='keys', tablefmt='pipe', showindex='never'))
        ##Limit Data with positive SHAP contributions only
        result_df = result_df[result_df['SHAP Value'] > 0]
        result_df=result_df[result_df['Importance Rank']<=25]
        print(result_df.shape)


        report=churn_explainer.ask_churnoracle(shap_summary=f"""The total count of customers in this analysis is {df_data.shape[0]}.The SHAP summary from the model is: 
        {tabulate.tabulate(result_df[['Feature','Group','SHAP Value','Probability Change (%)','Importance Rank']], headers='keys', tablefmt='pipe', showindex='never')}""",
        user_question=user_question)

        # Save intermediate result
        normalized_question=normalize_string(user_question)
        if 'intermediate_results' not in st.session_state:
            st.session_state.intermediate_results = {}
        if normalized_question not in st.session_state.intermediate_results:
            st.session_state.intermediate_results[normalized_question] = []
        st.session_state.intermediate_results[normalized_question].append({
            "tool": "subset_shap_explanation",
            "result_df": copy
        })


        return report
    except Exception as e:
        return str(e)

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
    st.markdown("--------------------------------------💬 *Customer Recommendations Tool* 💬--------------------------------------")

    try:
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
    except Exception as e:
        return str(e)
