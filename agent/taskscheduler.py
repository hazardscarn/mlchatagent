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
with open('./llm_configs.yml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

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
        prompt = conf['question_reformer']['prompt1']+f"""{user_question}"""+conf['question_reformer']['prompt2']
            
        taks_result = self.task_model.generate_content(prompt, stream=False)
        return taks_result.candidates[0].content.parts[0].text