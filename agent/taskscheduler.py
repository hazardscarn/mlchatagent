import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
from abc import ABC
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.language_models import CodeGenerationModel
from vertexai.language_models import CodeChatModel
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import HarmCategory,HarmBlockThreshold
from vertexai.generative_models import GenerationConfig
from vertexai.language_models import TextEmbeddingModel
import time
import json
from google.cloud import bigquery
from google.cloud import bigquery_connection_v1 as bq_connection
import pandas as pd
from datetime import datetime
import google.auth
import pandas as pd
import yaml
from google.cloud.exceptions import NotFound
import google.generativeai as genai


class TaskMaster(ABC):
    """ 
    This agent creates splits the task to subtasks if needed.

    Attributes
    ----------
    agentType : str
        The type of agent.
    task_splitter : genai.GenerativeModel
        The generative model used to splits the task to subtasks.

    Methods
    -------
    ask_taskmaster(user_question):
        Generates subtasks based on the user's question.
    """

    agentType = "TaskMaster"

    def __init__(self):
        """
        Constructs all the necessary attributes for the TaskMaster object.
        """


        self.task_model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")

    def ask_taskmaster(self, user_question: str) -> dict:
        """
        Question Reformer.

        Parameters
        ----------
        user_question : str
            The user's question.
        generated_sql : str
            The SQL query generated to answer the user's question.
        features : list
            The features of the data.

        Returns
        -------
        dict
            A dictionary containing the plot type, the column names for the X and Y axes, and any additional arguments.
        """
        prompt = f"""

            You are an intelligent task-splitting and question-rewording agent within a chatbot.
            Your primary role is to understand user queries and reword them to ensure clarity, precision, and completeness for SQL generation. 
            
            Your key responsibilities include:

            * Understanding User Queries: Carefully interpret and comprehend the user's questions, identifying key components and intent.
                - Is there any filtering user is asking for? If so list those filtering conditions
                - Is there any new columns user wants to create? If so detail what those are
                - Is there any aggregation user is asking for? If so detail what those are
                - Is there any modification/adjustemnt/treatment to a column user is asking for? If so detail what those are
                - Is the question about a list of customers or a few selected columns? If so detail what those are
                - Is the question about a single customer? If so detail what those are 

            * Rewording for Clarity and Detail: Reformulate the user's questions if needed to ensure they are clear, detailed, and structured in a way that facilitates accurate SQL generation.
            * Maintaining Original Intent: Preserve the original intent and scope of the user's query while enhancing its clarity and precision.


            Below are some guidelines to help you reword user questions effectively:

            1. If the user question is about reasons for churn or contributing factors of churn, then split the question into sub-questions:-
                     a. Question to get the data of specific customers user have asked to identify churn reasons for
                     b. Question to get the SHAP data of specific customers user have asked to identify churn reasons for
                In both questions ensure the query generated will return all columns i.e. use select * from table
            
            2. If the user question is about CLV or Churn impact analysis:-
                    a. Understand what subset of customers user is asking for
                    b. Understand what changes or treatment the user is asking for
                    c. Reword the question to make these two points clear and precise to generate SQL query
                    d. Add wording to reworded question to ensure the query generated will return all columns i.e. use select * from table along with the treatment or changes user is asking for

            3. If the user question is about general data analysis:
                    a. Ensure the question is clear and precise
                    b. Make sure filtering conditions are clear and precise
                    c. Make sure aggregation conditions are clear and precise
                    d. Make sure any new columns if user is asking for are clear and precise
                    e. Make sure instruction is passed if the question needs all columns to be returned i.e. use select * from table

            4. If the user question is about a few selected columns or list of customers:
                    a. Ensure the question is clear and precise
                    b. Make sure filtering conditions are clear and precise
                    c. Ask for the columns user is interested in only

            5. If the user question involves a plot or visualization:
                    a. Ensure only the columns asked for or needed to answer question is selected
                    b. Never select * from table unless it's clearly specified
                    c. Make sure the query returns only the columns needed

            **Below are some Examples:**

            *Example 1:*
            **User Question:** "What are the main reasons for churn of customers in service area hou?"
            **Reformed Question:**
            1. "Get all the data of customers in service area hou
                - filter by service area hou
                - select all columns"
            2. "Get all the SHAP data of customers in service area hou
                - filter by service area hou
                - select all columns"

            **Example 2:**
            **User Question:** "What are the main reasons for churn for customers with children aged more than 50?"
            **Reformed Question:**
            1. "Get all the data of customers with children and are aged more than 50
                - filter by customers with children, customers aged more than 50
                - select all columns""
            2. "Get the SHAP data of customers with children and are aged more than 50
                - filter by customers with children, customers aged more than 50
                - select all columns""

            **Example 6:**
            **User Question:** "What are the main reasons for churn"
            **Reformed Question:**
            1. "Get all the data of all customers
                - select all columns""
            2. "Get the SHAP data of all customers
                - select all columns""

            **Example 3:**
            **User Question:** "What are some recommended actions to reduce churn for customer 3334558?"
            **Reformed Question:**
            1. "Get all the data of customer 3334558"
            2. "Get all the counterfactual data of customer 3334558"

            **Example 4:**
            **User Question:** "What would be impact of churn if monthlyrevenue is cut by 5 percent and currentequipmentage is changed to 30
              for customers with churn probability more than 0.5 and currentequipmentage more than 500?"
            **Reformed Question:**
            1. "Get all the data of customers with churn probability more than 0.5 and and currentequipmentage more than 500, after monthlyrevenue is cut by 5 percent and currentequipmentage is changed to 30
                - filter by: churn probability more than 0.5, currentequipmentage more than 500
                - change/modify: monthlyrevenue by decreasing it by 5 percent, modify currentequipmentage to 30
                - select all columns"
            2. Use Churn effect Tool to identify the impact of churn if monthlyrevenue is cut by 5 percent and currentequipmentage is changed to 30

            **Example 5:**
            **User Question:** "What would be impact on CLV if one more activesubs was added and totalreccurring charges was reduced by 10$
            for customers in service are hou and nyc, and having incomegroup higher than 5? Assume treatment cost is 100$"
            **Reformed Question:**
            1. "Get all the data of customers in service are hou and nyc, and having incomegroup higher than 5, after activesubs was increased by 1 and totalreccuring charges was reduced by 10$
                - filter by:  service are hou and nyc,incomegroup higher than 5
                - change/modify: activesubs by increasing it by 1 percent, totalreccurring by decreasing it by 10$
                - select all columns"
            2. Use CLV Analysis Tool to identify impact on CLV if one more activesubs was added and totalreccurring charges was reduced by 10$
            for customers in service are hou and nyc, and having incomegroup higher than 5. Assume treatment cost is 100$

            **Example 6**
            **User Question:** Please provide a plot of distribution of customers across different age
            **Reformed Question:**
            1. "Get the distribution of customers across different age by bucketing age into 10 buckets"
                - select:  age_bucket, age bucket min, age_bucket max, count
                - group by : age_bucket
                - Note : Never use Select * for any plot or visualization

            **Example 7**
            **User Question:** "What is the  age distribution of customers who have higher churn because of revenue_per_minute?"
            **Reformed Question:**
            1. "Get the age distribution by bucketing age into 10 buckets for customers  who have higher churn because of revenue_per_minute
                - filter by: churn because of revenue_per_minute
                - select: age_bucket, age bucket min, age_bucket max, count,avg churn
                - group by: age_bucket
                - Note : To identify customers who have higher churn because of revenue_per_minute, use the customer_shap_data and filter by revenue_per_minute>0.5"
            Below is the user question:
            {user_question}

            **Notes:**
            - Create subquestion only if necessary
            - Never use Select * for any plot or visualization
            - Ensure clarity and precision in sub-questions.
            - Each sub-question should lead to a specific and actionable SQL query.
            - Maintain the context of the original user question while decomposing it into sub-questions.
            - Output should be just the Reformed Question or SubQuestions
            - Use Counterfactual Analysis tool only when the question is about a single customers churn recommendation
            - You should only return the reformed question or subquestions. DO NOT return any additional content or explanation.
            """
        taks_result = self.task_model.generate_content(prompt, stream=False)
        return taks_result.candidates[0].content.parts[0].text