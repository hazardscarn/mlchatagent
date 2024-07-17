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


class ShapOracle(ABC):
    """ 
    This agent creates report from SHAP contribution data

    Attributes
    ----------
    agentType : str
        The type of agent.
    churnexplainer : genai.GenerativeModel
        The generative model used to splits the task to subtasks.

    Methods
    -------
    ask_churnoracle(shap_summary):
        Generates subtasks based on the user's question.
    """

    agentType = "ShapOracle"

    def __init__(self,config_file="./conf_telchurn.yml"):
        """
        Constructs all the necessary attributes for the TaskMaster object.
        """


        self.churnexplainer = genai.GenerativeModel(model_name="gemini-1.5-flash-001")
        with open(config_file) as file:
            self.conf = yaml.load(file, Loader=yaml.FullLoader)
    def ask_churnoracle(self,user_question, shap_summary: str) -> dict:
        """
        Generates a report on reasons for churn from shap summary.

        Parameters
        ----------
        shap_summary : str
            shap summary.
        user_question : str
            user's question.
        features : list
            The features of the data.

        Returns
        -------
        dict
            A report on reasons for churn.
        """
        prompt = f"""

            You are an intelligent agent that analyzes SHAP summary data to identify reasons for customer churn and suggest possible next actions to reduce churn. 
            Your goal is to create a detailed report that highlights key insights on mian reasons for churn and provides actionable recommendations based on the SHAP data.

            *Higher the SHAP value (more positive it is), the more the feature pushes prediction towards churn and vice-versa*

            Your task is as below:-

                1. Read and Analyze the Data:
                    - Understand the structure of the provided SHAP data.
                    - Focus on the features, groups,SHAP values.
                    - *Higher the SHAP value (more positive it is), the more the feature pushes prediction towards churn*
                    - Probability change % shows the % increase in churn by this group compared to baseline model
                    - Understand clearly which features and which groups within the features have the highest SHAP values.
                2. Identify Key Insights:
                    - Determine which features have the highest positive impact on churn.Use SHAP values and probability changes for this.
                    - *Higher the SHAP value (more positive it is), the more the feature pushes prediction towards churn*
                    - Provide extra attention to the sign of the SHAP value. Positive SHAP value indicates higher churn probability and vice-versa.
                    - DO NOT take negative SHAP valued groups as higher churn probability groups. That is a fatal error.
                    - Idenitifying wrong trends/insights will lead to wrong recommendations. This is a fatal error and should be avoided at all costs.
                    - Identify as much insights as possible from the SHAP summary data.
                    - If you identify a group within a feature have high SHAP value, identify the trend of churn comparing to other groups in the feature. This is important to understand the impact of the group in the feature.
                    - Make sure the trend you are reporting is accurate.
                    - When explaining a numeric feature with its ranges in Group, explain which range have higher churn and which range have lower churn
                    - DO NOT randmoly say a group in a feature has higher churn. You should explain how much higher churn it has compared to other groups in the feature.
                    - Always check if there is another group in the feature that has higher churn than the group you are reporting.
                    - If you report a group have higher churn contribution than another group in the feature, but the SHAP data shows otherwise it is a FATAL error
                3. Provide Reasons for Churn:
                    - Clearly articulate the reasons for churn based on the data analysis.
                    - *Higher the SHAP value (more positive it is), the more the feature pushes prediction towards churn*
                    - Provide as much reasons with justifiable evidence from the SHAP summary data.
                    - Explain how different groups within each feature contribute to the overall churn probability.
                4. Suggest Next Actions:
                    - Based on the insights, recommend specific actions to reduce churn.
                    - Consider both immediate and long-term strategies.
                    - Highlight which customer segments should be targeted for each action.
                5. Generate a Detailed Report:
                    - Summarize your findings in a clear and concise report.
                    - Reinforce the importance of targeted actions to reduce churn.

            **NOTES:
             - Use 'Probability Change (%)'to get a churn contribution in probability scale which is more readable to user 
                - Higher the value (more positive it is), the more the feature pushes the model output towards churn
                - Lower the value (more negative it is), the more the feature prevents churn
                    eg: If a feature has a probability change of 3.2, it means that the churn probability increases by 3.2% due to that feature compared to base model.
                    eg: If a feature has a probability change of -2.5, it means that the churn probability decreases by 2.5% due to that feature compared to base model..
            - **SHAP Value : 
                - *Higher the SHAP value (more positive it is), the more the feature pushes prediction towards churn*
                - Lower the SHAP value (more negative it is), the more the feature prevents churn
            - If you report a group have higher churn contribution than another within feature, but the SHAP data shows otherwise it is a FATAL error
            - Always make the report grounded with SHAP summary data. If you identify a trend that is not supported by SHAP data, it is a FATAL error
                For example if SHAP value for x1,x2,x3,x4,x5 is 0.6,-0.1,-0.2,0.1,0.2 the report should state:-
                  - x1 is highest contributor of churn as by model 
                  - There is also trend in increasing churn with x4 and x5 but it is lower than than x1
            - 'Group' refers to different sub groups within a feature.  
            - *Insights generated for each groups should be grounded with supporting facts from summary*
            - DO NOT using words like The negative impact, negative SHAP Value etc. Use words like higher churn probability, higher churn contribution etc.
            - When explaining a numeric feature with its ranges in Group, explain which range have higher churn and which range have lower churn
            - THE STATS BEHIND CHURN REASONS SHOULD BE GROUNDED IN THE SHAP SUMMARY DATA. DO NOT MAKE UP REASONS FOR CHURN.
            - Probability Change (%) is already in percentage scale. No need to convert it to percentage scale.                  
            - 'Importance Rank' shows the importance of the feature in model predictions; a lower rank means higher importance.
            - 'Importance Rank' has no place in Churn analysis. Use SHAP values and Probability Change (%) for analysis.
            -  Importance Rank is not a measure of churn contribution. It specifies which features are more important for the model to make predictions.
            -  You may use Importance rank to answer any questions related to feature importance in the model. However, it is not relevant for churn analysis. 
            - When you make the final report try to make it non technical in wording as possible

            Below is the SHAP summary data you need to analyze:
            {shap_summary}

            Below is the original user question. Use it to identify the context of the report:
            {user_question}

            Output format of report should be as below:

            1. Overview:
                - Brief summary of the report and what the report is for.
                - Mention the count of customers this report is based on.
            2. Key Insights:
                - List the top features contributing to churn and summarize their impact. Provide atleast 6 key insights.
            3. Reasons for Churn:
                - Detailed explanation of the reasons for churn based on the data analysis
            4. Next Actions:
                - Specific recommendations to reduce churn and target customer segments
            5. Conclusion:
                - Final thoughts and summary of the report 
                - Add a note to use Churn and CLV impact analysis first to understand approximate impact of potential actions you may take from this report
                - Add a note at the end of report that these findings are based on the model and may not be accurate in real world. You should do more detailed analysis on every insights. I can help you with that if you want, just tell me what to do.                         
            """
        shap_action = self.churnexplainer.generate_content(prompt, stream=False)
        return shap_action.candidates[0].content.parts[0].text
    
    def ask_recommendation(self,user_question:str, counterfactual: str,customer:str) -> dict:
        """
        Generates a report on recommended actions to reduce churn for a customer.

        Parameters
        ----------
        shap_summary : str
            shap summary.
        user_question : str
            user's question.
        features : list
            The features of the data.

        Returns
        -------
        str
            A report on recommended actions to reduce churn for a customer. 
        """
        if ((not counterfactual) or (len(counterfactual.strip()) > 100)):

            prompt = f"""

                You are an assistant to customer service manager who is looking for recommendations to reduce churn for a specific customer.
                You have access to recommended actions to prevent churn generated by counterfactual analysis of action features for a specific customer.
                Your goal is to provide clear and actionable insights on how to reduce customer churn based on counterfactual analysis and to develop effective questions to engage customers and implement the recommendations.
                Your task is as below:-

                    1. Read and Analyze the Data:
                        - Understand the counterfactual recommendations for the specific customer.
                        - Understand who the customer is. Focus on customer and contract features.
                            Below is the list of customer level feature names:
                            {self.conf['llm_subsets']['customer_features']}
                            Below is the list of customer contract feature names:
                            {self.conf['llm_subsets']['contract_features']}

                    2. Identify Key Insights:
                        - There might be multiple recommendations for the customer. 
                        - Use the rank of the recommendations to prioritize the actions order
                        - If multiple recommendations contains same actions with minor changes, consider only the top ranked action.
                        - If possible specify the churn probability reduced by each successful action. Keep the probability change in percentage scale and rounded to 2 decimal places.
                        - Understand what the current values of the customer are for the features in the recommendations.

                    3. Communicate Clearly:
                        - Use simple and clear language to explain the recommendations.
                        - Avoid technical jargon and ensure the explanation is understandable for a customer service agent.

                    4. Provide Specific Recommendations:
                        - Detail the actions that can be taken to reduce churn.
                        - Explain why these actions are effective based on the counterfactual analysis.
                        - Include examples or scenarios to illustrate the recommendations.

                    5. Create Questionnaires:
                        - Develop specific questions for customer service agents to ask customers in order to implement the recommended actions.
                        - Ensure the questions are relevant to the recommended actions and designed to gather necessary information or prompt customer engagement.
                        - Do not ask personal questions or questions about data already available to you.
                        - Questionnaire should be sequenced in a way that it helps to implement the recommendations.
                        - QUestionnaire should be designed in a way that it helps to understand the customer's current situation and how the recommendations can be implemented.
                        - Questionnaire should be sequential. The answer to one question should lead to the next question.
                        - Create multiple different segments of questions for each recommended action.


                Below is the counterfactual recommendations for the specific customer:
                {counterfactual}

                Below is the customer details:
                {customer}

                Below is the original user question. Use it to identify the context of the report:
                {user_question}


                ***OUTPUT FORMAT SHOULD BE AS BELOW***:

                1. Customer Persona::
                    - Describe who the customer is in 30 words or less, including a persona if possible.
                    - Also make a note of customer's churn probaility predicted
                2. Recommended Actions:
                    - List the recommended actions based on the counterfactual analysis
                3. **Questionnaire for Agent**:
                    - Develop a set of questions for the customer service agent to ask customers to implement the recommended actions.
                """
        else:
            prompt = f"""

                You are an assistant to customer service manager who is looking for recommendations to reduce churn for a specific customer.
                You have access to major contributing factors towards churn across portfolio.
                Your goal is to provide clear and actionable insights on how to reduce customer churn based on available information and to develop effective questions to engage customers and implement the recommendations.
                Only make recommendations based of action features for a specific customer.
                Your task is as below:-

                    1. Read and Analyze the Data:
                        - Understand who the customer is. Focus on customer and contract features.
                            Below is the list of customer level feature names:
                            {self.conf['llm_subsets']['customer_features']}
                            Below is the list of customer contract feature names:
                            {self.conf['llm_subsets']['contract_features']}

                    2. Identify Key Insights:
                        
                        - Look and understand at the major contributing features towards churn across portfolio below 
                        - Understand what the current values of the customer are for these features
                        - Only make recommendations based of action features for a specific customer.
                        - Make recommendations to customer based on action features grounded by major contributing factors cross portfolio to reduce churn

                    3. Communicate Clearly:
                        - Use simple and clear language to explain the recommendations.
                        - Avoid technical jargon and ensure the explanation is understandable for a customer service agent.
                        - Support your findings with evidence from the SHAP summary with probability changes.

                    4. Provide Specific Recommendations:
                        - Only make recommendations based of action features for a specific customer.
                        - Detail the actions that can be taken to reduce churn.
                        - Explain why these actions are effective.
                        - Include examples or scenarios to illustrate the recommendations.

                    5. Create Questionnaires:
                        - Develop specific questions for customer service agents to ask customers in order to implement the recommended actions.
                        - Ensure the questions are relevant to the recommended actions and designed to gather necessary information or prompt customer engagement.
                        - Do not ask personal questions or questions about data already available to you.
                        - Questionnaire should be sequenced in a way that it helps to implement the recommendations.
                        - QUestionnaire should be designed in a way that it helps to understand the customer's current situation and how the recommendations can be implemented.
                        - Questionnaire should be sequential. The answer to one question should lead to the next question.
                        - Create multiple different segments of questions for each recommended action.


                Below is the major contributing factors towards churn across portfolio:
                ***Service City:**
                        * **NMC, ATH, and SEW cities:**  These cities have the highest churn probability, with NMC experiencing a 1.15% increase in churn probability compared to the base model. This indicates that customers in these cities are significantly more likely to churn. 
                        * **LAU, MIL, OMA, VAH, NOR:** These cities also contribute to churn but have a lower churn probability than the top 3 cities. For instance, LAU has a churn probability increase of 0.59%.
                        * **Other cities:** Several other cities show a lower churn probability than the top 3, including 'LOU', 'OHH', 'SDA', 'NNY', 'IPM', and 'DET'.  Cities like 'STL', 'SFR', 'PIT', 'CHI', 'MIN', 'BOS', 'NOL', 'SHE', 'NSH', 'HWI', 'PHX', 'FLN', 'HAR', 'KCY', 'APC', 'DAL', 'OHI', 'SAN', 'NMX', 'PHI', 'SLC', 'DEN', 'HOU', 'IND', 'MIA', 'GCW', 'LAW', 'INH', 'BIR', 'SFU', 'UNKNOWN', and 'NVU' are associated with even lower churn probabilities. 
                    ***Current Equipment Days:**
                        * **(-3.001, 86.0], (720.2, 1794.0]**: These ranges demonstrate the highest contribution to churn, potentially indicating a dissatisfaction with equipment, leading to churn. The range (-3.001, 86.0] shows a -1.47% decrease in churn probability, and the range (720.2, 1794.0] shows a -1.35% decrease in churn probability. 
                        * **(86.0, 183.0], (224.0, 278.0]**: These ranges show a relatively lower contribution to churn compared to the higher ranges. 
                        * **(278.0, 324.0], (324.0, 376.0], (183.0, 224.0], (376.0, 447.0], (447.0, 566.0], (566.0, 720.2]:** These ranges show even lower contribution to churn compared to the previous ranges. 
                    ***Months in Service:**
                        * **(10.0, 12.0]**: This range shows the highest contribution to churn, with a 0.17% probability increase. This suggests that customers who have been with the company for 10 to 12 months are more likely to churn.
                        * **(14.0, 16.0]**: This range shows a lower but still positive contribution to churn. 
                        * **(26.0, 33.0]**: This range shows a lower contribution to churn than the previous ranges. 
                        * **(16.0, 19.0]**: This range shows a small contribution to churn.
                        * **(12.0, 14.0]**: This range shows a very low contribution to churn. 
                        * **(5.999, 8.0], (22.0, 26.0], (19.0, 22.0], (8.0, 10.0], (33.0, 59.0]:** These ranges show a negative contribution to churn, indicating that customers in these ranges are less likely to churn compared to other groups.
                    ***Total Recurring Charge:**
                        * **(30.0, 33.0]**: This range has the highest contribution to churn, with a 0.28% increase in churn probability. This suggests that customers with a recurring charge in this range are more likely to churn.
                        * **(-9.001, 17.0]**: This range has a lower but still positive contribution to churn.
                        * **(75.0, 400.0]**: This range shows a negative contribution to churn. 
                        * **(40.0, 45.0], (55.0, 60.0], (45.0, 50.0], (17.0, 30.0], (33.0, 40.0], (60.0, 75.0], (50.0, 55.0]:** These ranges show a negative contribution to churn, implying that customers within these ranges are less likely to churn.
                    ***Credit Rating:**
                        * **6-verylow:** This group exhibits the highest contribution to churn, with a 0.30% probability increase. This suggests financial constraints might be impacting the churn decision of these customers.
                        * **7-lowest:** This group has a negative contribution to churn.
                        * **1-highest:** This group also has a negative contribution to churn.
                        * **5-low, 2-high, 4-medium, 3-good:** These groups show a negative contribution to churn, implying that customers with these credit ratings are less likely to churn compared to other groups. 

                Below is the customer details:
                {customer}

                Below is the list of action feature names:
                {self.conf['llm_subsets']['action_features']}



                Below is the original user question. Use it to identify the context of the report:
                {user_question}


                ***OUTPUT FORMAT SHOULD BE AS BELOW***:

                1. Customer Persona::
                    - Describe who the customer is in 30 words or less, including a persona if possible.
                    - Also make a note of customer's churn probaility predicted
                2. Recommended Actions:
                    - List the recommended actions based on the counterfactual analysis
                3. **Questionnaire for Agent**:
                    - Develop a set of questions for the customer service agent to ask customers to implement the recommended actions.

                """
        #print(prompt)
        recommendation = self.churnexplainer.generate_content(prompt, stream=False)
        return recommendation.candidates[0].content.parts[0].text