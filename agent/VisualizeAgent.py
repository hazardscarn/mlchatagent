#Copyright 2024 Google LLC
#This agent was modified and built from the agent built by Google LLC in 2024
#This agent generates google charts code for displaying charts on web application
#Generates two charts with elements "chart-div" and "chart-div-1"
#Code is in javascript

from abc import ABC
from vertexai.language_models import CodeChatModel
from vertexai.generative_models import GenerativeModel,HarmCategory,HarmBlockThreshold
from .core import Agent 
import pandas as pd
import json
import yaml  


class VisualizeAgent(Agent, ABC):
    """
    An agent that generates JavaScript code for Google Charts based on user questions and SQL results.

    This agent analyzes the user's question and the corresponding SQL query results to determine suitable chart types. It then constructs JavaScript code that uses Google Charts to create visualizations based on the data.

    Attributes:
        agentType (str): Indicates the type of agent, fixed as "VisualizeAgent".
        model_id (str): The ID of the language model used for chart type suggestion and code generation.
        model: The language model instance.

    Methods:
        getChartType(user_question, generated_sql) -> str:
            Suggests the two most suitable chart types based on the user's question and the generated SQL query.

            Args:
                user_question (str): The natural language question asked by the user.
                generated_sql (str): The SQL query generated to answer the question.

            Returns:
                str: A JSON string containing two keys, "chart_1" and "chart_2", each representing a suggested chart type.

        getChartPrompt(user_question, generated_sql, chart_type, chart_div, sql_results) -> str:
            Creates a prompt for the language model to generate the JavaScript code for a specific chart.

            Args:
                user_question (str): The user's question.
                generated_sql (str): The generated SQL query.
                chart_type (str): The desired chart type (e.g., "Bar Chart", "Pie Chart").
                chart_div (str): The HTML element ID where the chart will be rendered.
                sql_results (str): The results of the SQL query in JSON format.

            Returns:
                str: The prompt for the language model to generate the chart code.

        generate_charts(user_question, generated_sql, sql_results) -> dict:
            Generates JavaScript code for two Google Charts based on the given inputs.

            Args:
                user_question (str): The user's question.
                generated_sql (str): The generated SQL query.
                sql_results (str): The results of the SQL query in JSON format.

            Returns:
                dict: A dictionary containing two keys, "chart_div" and "chart_div_1", each holding the generated JavaScript code for a chart.
    """


    agentType: str ="VisualizeAgent"

    def __init__(self):
        self.model_id = 'gemini-1.5-flash-001'
        self.model = GenerativeModel("gemini-1.5-flash-001")

    def getChartType(self,user_question, generated_sql):
        map_prompt=f'''
        You are expert in generating visualizations.

        Some commonly used charts and when do use them:

            - Text or Score card is best for showing single value answer

            - Table is best for Showing data in a tabular format.

            - Bullet Chart is best for Showing individual values across categories.

            - Bar Chart is best for Comparing individual values across categories, especially with many categories or long labels.

            - Column Chart is best for Comparing individual values across categories, best for smaller datasets.

            - Line Chart is best for Showing trends over time or continuous data sets with many data points.

            - Area Chart is best for Emphasizing cumulative totals over time, or the magnitude of change across multiple categories.

            - Pie Chart is best for Show proportions of a whole, but only for a few categories (ideally less than 8).

            - Scatter Plot	is best for Investigating relationships or correlations between two variables.

            - Bubble Chart	is best for Comparing and showing relationships between three variables.

            - Histogram	is best for Displaying the distribution and frequency of continuous data.

            - Map Chart	is best for Visualizing data with a geographic dimension (countries, states, regions, etc.).

            - Gantt Chart	is best for Managing projects, visualizing timelines, and task dependencies.

            - Heatmap is best for	Showing the density of data points across two dimensions, highlighting areas of concentration.


        Examples:

        Question: What is the average monthly revenue and churn rate for each service city??
        SQL: SELECT service_city,AVG(monthlyrevenue) AS avg_monthly_revenue,AVG(churn) AS churn_rate FROM `mlchatagent-429005.telecom_churn.customer_data`
            GROUP BY service_city;
        Answer: Bar Chart, Table Chart 

        Question: Create 10 buckets for current equipment age. Provide a plot of equipment age buckets and average churn rate?
        SQL: WITH data AS (
                            SELECT 
                                currentequipmentdays,
                                churn,
                                NTILE(10) OVER (ORDER BY currentequipmentdays) AS current_equipment_age_bucket,
                            FROM 
                                `mlchatagent-429005.telecom_churn.customer_data`
                            ),
                            buckets AS (
                            SELECT 
                                SAFE_CAST(current_equipment_age_bucket AS BIGNUMERIC) AS current_equipment_age_bucket,
                                SAFE_CAST(MIN(currentequipmentdays) OVER (PARTITION BY current_equipment_age_bucket) AS BIGNUMERIC) AS current_equipment_age_bucket_min,
                                SAFE_CAST(MAX(currentequipmentdays) OVER (PARTITION BY current_equipment_age_bucket) AS BIGNUMERIC) AS current_equipment_age_bucket_max,
                            churn
                            FROM 
                                data
                            )
                            SELECT 
                            current_equipment_age_bucket,
                            current_equipment_age_bucket_min,
                            current_equipment_age_bucket_max,
                            avg(churn) as avg_churn
                            FROM 
                            buckets
                            GROUP BY 
                            current_equipment_age_bucket,
                            current_equipment_age_bucket_min,
                            current_equipment_age_bucket_max;

        Answer: Table Chart, Bar Chart

        Question: Please provide a plot of distribution of customers across different credit group 
        SQL:  SELECT creditrating, COUNT(*) AS count FROM mlchatagent-429005.telecom_churn.customer_data GROUP BY 1
        Answer: Pie Chart, Bar Chart

        Guidelines:
        -Do not add any explanation to the response. Only stick to format Chart-1, Chart-2
        -Do not enclose the response with js or javascript or ```

        Below is the Question and corresponding SQL Generated, suggest best two of the chart types

        Question : {user_question}
        Corresponding SQL : {generated_sql}

        Note:
            -If generated sql query have select * without a GROUP BY condition, suggestion-1 and suggestion-1 should be None.
            -Faliure to do so will create Fatal errors
            -All series on a given axis must be of the same data type. For example, if you have a bar chart with a series of numbers and a series of strings, the chart will not render correctly.
            -Convert data types as necessary to ensure that all series on a given axis are of the same data type.
        Output format:
        Respond using a valid JSON format with two elements chart_1 and chart_2 as below

        {{"chart_1":suggestion-1,
         "chart_2":suggestion-2}}

      '''
        chart_type=self.model.generate_content(map_prompt, stream=False).candidates[0].text
        # print(chart_type)
        # chart_type = model.predict(map_prompt, max_output_tokens = 1024, temperature= 0.2).candidates[0].text
        return chart_type.replace("\n", "").replace("```", "").replace("json", "").replace("```html", "").replace("```", "").replace("js\n","").replace("json\n","").replace("python\n","").replace("javascript","")

    def getChartPrompt(self,user_question, generated_sql, chart_type, chart_div, sql_results):
        return f'''
        You are expert in generated visualizations.
        
    Guidelines:
    -Do not add any explanation to the response.
    -Do not enclose the response with js or javascript or ```


    You are asked to generate a visualization for the following question:
    {user_question}

    The SQL generated for the question is:
    {generated_sql}

    The results of the sql which should be used to generate the visualization are in json format as follows:
    {sql_results}

    Needed chart type is  : {chart_type}

Guidelines:

   - Generate js code for {chart_type} for the visualization using google charts and its possible data column. You do not need to use all the columns if not possible.
   - The generated js code should be able to be just evaluated as javascript so do not add any extra text to it.
   - ONLY USE the template below and STRICTLY USE ELEMENT ID {chart_div} TO CREATE THE CHART

    google.charts.load('current', <add packages>);
    google.charts.setOnLoadCallback(drawChart);
    drawchart function 
        var data = <Datatable>
        with options
    Title=<<Give appropiate title>>
    width=600,
    height=300,
    hAxis.textStyle.fontSize=5
    vAxis.textStyle.fontSize=5
    legend.textStyle.fontSize=10

    other necessary options for the chart type

        var chart = new google.charts.<chart name>(document.getElementById('{chart_div}'));

        chart.draw()

    Example Response: 

   google.charts.load('current', {{packages: ['corechart']}});
   google.charts.setOnLoadCallback(drawChart);
    function drawChart() 
   {{var data = google.visualization.arrayToDataTable([['Product SKU', 'Total Ordered Items'],
     ['GGOEGOAQ012899', 456],   ['GGOEGDHC074099', 334], 
      ['GGOEGOCB017499', 319],    ['GGOEGOCC077999', 290], 
         ['GGOEGFYQ016599', 253],  ]); 
         
         var options =
          {{ title: 'Top 5 Product SKUs Ordered',  
           width: 600,   height: 300,    hAxis: {{     
           textStyle: {{       fontSize: 12    }} }},  
            vAxis: {{     textStyle: {{      fontSize: 12     }}    }},
               legend: {{    textStyle: {{       fontSize: 12\n      }}   }},  
                bar: {{      groupWidth: '50%'    }}  }};
                 var chart = new google.visualization.BarChart(document.getElementById('{chart_div}')); 
                  chart.draw(data, options);}}
        '''

    def generate_charts(self,user_question,generated_sql,sql_results):
        chart_type = self.getChartType(user_question,generated_sql)
        # chart_type = chart_type.split(",")
        # chart_list = [x.strip() for x in chart_type]
        chart_json = json.loads(chart_type)
        chart_list =[chart_json['chart_1'],chart_json['chart_2']]
        print("Charts Suggested : " + str(chart_list))
        # Check if the first component is not None or 'None'
        if chart_list[0] is not None and chart_list[0] != 'None':
            context_prompt=self.getChartPrompt(user_question,generated_sql,chart_list[0],"chart_div",sql_results)
            context_prompt_1=self.getChartPrompt(user_question,generated_sql,chart_list[1],"chart_div_1",sql_results)
            context_query = self.model.generate_content(context_prompt, stream=False)
            context_query_1 = self.model.generate_content(context_prompt_1, stream=False)
            google_chart_js={"chart_div":context_query.candidates[0].text.replace("```json", "").replace("```", "").replace("json", "").replace("```html", "").replace("```", "").replace("js","").replace("json","").replace("python","").replace("javascript",""),
                            "chart_div_1":context_query_1.candidates[0].text.replace("```json", "").replace("```", "").replace("json", "").replace("```html", "").replace("```", "").replace("js","").replace("json","").replace("python","").replace("javascript","")}

            return google_chart_js
        else:
            return None