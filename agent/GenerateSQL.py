from utils import remove_sql_and_backticks
import google.generativeai as genai
import google.ai.generativelanguage as glm
from google.generativeai import caching
import pandas as pd
import numpy as np
import tabulate
import datetime



class BuildSQLAgent:
    """
    An agent specialized in generating SQL queries for BigQuery or Sqllite databases.

    This agent analyzes user questions, available table schemas, and column descriptions to construct syntactically and semantically correct SQL queries. It adapts the query generation process based on the target database type (BigQuery or Sqllite).


    Methods:
        build_sql(source_type,user_question, tables_schema, similar_sql, max_output_tokens=2048, temperature=0.4, top_p=1, top_k=32) -> str:
            Generates an SQL query based on the provided parameters.

            Args:
                source_type (str): The database type ("bigquery" or "sqllite").
                user_question (str): The question asked by the user.
                tables_schema (str): A description of the available tables and their columns.
                tables_detailed_schema (str): Detailed descriptions of the columns in the tables.
                similar_sql (str): Examples of similar SQL queries to guide the generation process.
                max_output_tokens (int, optional): Maximum number of tokens in the generated SQL. Defaults to 2048.
                temperature (float, optional): Controls the randomness of the generated output. Defaults to 0.4.
                top_p (float, optional): Nucleus sampling threshold for controlling diversity. Defaults to 1.
                top_k (int, optional): Consider the top k most probable tokens for sampling. Defaults to 32.

            Returns:
                str: The generated SQL query as a single-line string.
    """

        

    def generate_sql(user_question:str):
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

        context_prompt = f"""
                You are a Sqlite SQL guru. Write a SQL comformant query for Sqlite that answers the following question while using the provided context to correctly refer to the SQlite tables and the needed column names.

                Guidelines:
                - Join as minimal tables as possible.
                - When joining tables ensure all join columns are the same data_type.
                - Analyze the database and the table schema provided as parameters and undestand the relations (column and table relations).
                - Don't include any comments in code.
                - **Remove ```sql and ``` from the output and generate the SQL in single line.**
                - Tables should be refered to using a fully qualified name with enclosed in ticks (`) e.g. `project_id.owner.table_name`.
                - Use all the non-aggregated columns from the "SELECT" statement while framing "GROUP BY" block.
                - Use ONLY the column names (column_name) mentioned in Table Schema. DO NOT USE any other column names outside of this.
                - Associate column_name mentioned in Table Schema only to the table_name specified under Table Schema.
                - Table names are case sensitive. DO NOT uppercase or lowercase the table names.
                - Always enclose subqueries and union queries in brackets.
                - Refer to the examples provided below, if given.
                - If the question is not clear, answer with : {"Sorry. No information available for this question."}
                - If the question have task to create summary, create the appropritae summary logic
                - Only answer questions relevant to the tables or columns listed in the table schema If a non-related question comes, answer exactly with : {"Sorry. No information available for this question."}
                - When question is about average feature contribution to churn, use customer_shap_contributions columns only in the output unless it's necessary
                - ***When a subset data query is asked DONOT use LIMIT. Return query for whole data asked for***
                - When using a column make sure it's available in the table schema provided.
                - **When using column values to subset make sure the case matches. it is case sensitive.**

                Here are some examples of user-question and SQL queries:

                Q:Give me shap contribution of customers in service city hou
                A: SELECT T1.* FROM `customer_shap_contributions` T1 JOIN `customer_data` T2 ON T1.customerid = T2.customerid WHERE T2.service_city = 'hou'

                Q:Identify top churn contributors customers with age more than 60?
                A: SELECT * FROM `customer_data` WHERE agehh1 > 60
                A: SELECT T1.* FROM `customer_shap_contributions` T1 JOIN `customer_data` T2 ON T1.customerid = T2.customerid WHERE T2.agehh1 > 60
                            
                Q: What is the average monthly charges for customers who have churned?
                A: SELECT AVG(monthly_charges) from customer_data where churn=1;

                Q:Tell me about customerID 3114822?
                A: SELECT * from customer_data where customerID=3114822;

                Q:Tell me SHAP contribution of customerID 3114822?
                A: SELECT * from customer_shap_contributions where customerID=3114822;

                Q:Tell me some recommendations to reduce churn for customerID 3114822?
                A: SELECT * from customer_counterfactuals where customerID=3114822;



                Q:Give a summary of shap contribution of age and revenue customers in service city hou
                A:  SELECT AVG(`customer_shap_contributions`.agehh1),
                    AVG(`customer_shap_contributions`.revenue_per_minute)
                    FROM `customer_shap_contributions` 
                    JOIN `customer_data` ON
                    `customer_shap_contributions`.customerid = `customer_data`.customerid 
                    WHERE `customer_data`.service_city = 'hou'

                Q:Identify customer data in service city hou?
                A: SELECT * FROM `customer_data` WHERE service_city = 'hou'
                    
                Q:Give me shap contribution of customers in service city hou
                A: SELECT T1.* FROM `customer_shap_contributions` T1 JOIN `customer_data` T2 ON T1.customerid = T2.customerid WHERE T2.service_city = 'hou'

                Q:Identify top churn contributors customers with age more than 60?
                A: SELECT * FROM `customer_data` WHERE agehh1 > 60
                A: SELECT T1.* FROM `customer_shap_contributions` T1 JOIN `customer_data` T2 ON T1.customerid = T2.customerid WHERE T2.agehh1 > 60
                            


                Below are descriptions for the tables in the database:

                customer_data : 
                    - Contains all available customer data points and their churn prediction
                    - Donot use this data for SHAP contribution except for adding filters
                data_dictionary: 
                    - Contains the description of all the columns in the customer_data table
                customer_shap_contributions : 
                    - Contains the individual feature contribution towards every customers churn prediction in SHAP scale.
                    - High positive SHAP value indicates higher probability of churn and vice versa.
                    - Can be used to explain a churn prediction of a customer in combination with customer_data
                customer_counterfactuals : 
                    - Contains the counterfactuals generated for every customer. Can be used to provide recommendations to a customer to reduce churn



                Table Schema:
                Tables:

                    customer_data
                    customer_shap_contributions
                    customer_counterfactuals


                    Schemas:


                    customer_data:
                    
                    Name: childreninhh, Type: TEXT
                    Name: handsetrefurbished, Type: TEXT
                    Name: handsetwebcapable, Type: TEXT
                    Name: truckowner, Type: TEXT
                    Name: rvowner, Type: TEXT
                    Name: homeownership, Type: TEXT
                    Name: buysviamailorder, Type: TEXT
                    Name: respondstomailoffers, Type: TEXT
                    Name: optoutmailings, Type: TEXT
                    Name: nonustravel, Type: TEXT
                    Name: ownscomputer, Type: TEXT
                    Name: hascreditcard, Type: TEXT
                    Name: newcellphoneuser, Type: TEXT
                    Name: notnewcellphoneuser, Type: TEXT
                    Name: ownsmotorcycle, Type: TEXT
                    Name: madecalltoretentionteam, Type: TEXT
                    Name: creditrating, Type: TEXT
                    Name: prizmcode, Type: TEXT
                    Name: occupation, Type: TEXT
                    Name: maritalstatus, Type: TEXT
                    Name: service_city, Type: TEXT
                    Name: monthlyrevenue, Type: REAL
                    Name: monthlyminutes, Type: REAL
                    Name: totalrecurringcharge, Type: REAL
                    Name: directorassistedcalls, Type: REAL
                    Name: overageminutes, Type: REAL
                    Name: roamingcalls, Type: REAL
                    Name: percchangeminutes, Type: REAL
                    Name: percchangerevenues, Type: REAL
                    Name: droppedcalls, Type: REAL
                    Name: blockedcalls, Type: REAL
                    Name: unansweredcalls, Type: REAL
                    Name: customercarecalls, Type: REAL
                    Name: threewaycalls, Type: REAL
                    Name: receivedcalls, Type: REAL
                    Name: outboundcalls, Type: REAL
                    Name: inboundcalls, Type: REAL
                    Name: peakcallsinout, Type: REAL
                    Name: offpeakcallsinout, Type: REAL
                    Name: droppedblockedcalls, Type: REAL
                    Name: callforwardingcalls, Type: REAL
                    Name: callwaitingcalls, Type: REAL
                    Name: monthsinservice, Type: REAL
                    Name: uniquesubs, Type: REAL
                    Name: activesubs, Type: REAL
                    Name: handsets, Type: REAL
                    Name: handsetmodels, Type: REAL
                    Name: currentequipmentdays, Type: REAL
                    Name: agehh1, Type: REAL
                    Name: retentioncalls, Type: REAL
                    Name: retentionoffersaccepted, Type: REAL
                    Name: referralsmadebysubscriber, Type: REAL
                    Name: adjustmentstocreditrating, Type: REAL
                    Name: revenue_per_minute, Type: REAL
                    Name: total_calls, Type: REAL
                    Name: avg_call_duration, Type: REAL
                    Name: service_tenure, Type: REAL
                    Name: customer_support_interaction, Type: REAL
                    Name: handsetprice, Type: REAL
                    Name: incomegroup, Type: REAL
                    Name: customerid, Type: REAL
                    Name: churn, Type: REAL
                    Name: prediction, Type: REAL

                        Distinct Values for Categorical Variables:

                        childreninhh: ['no', 'yes']
                        handsetrefurbished: ['no', 'yes']
                        handsetwebcapable: ['no', 'yes']
                        truckowner: ['yes', 'no']
                        rvowner: ['yes', 'no']
                        homeownership: ['known', 'unknown']
                        buysviamailorder: ['no', 'yes']
                        respondstomailoffers: ['yes', 'no']
                        optoutmailings: ['yes', 'no']
                        nonustravel: ['no', 'yes']
                        ownscomputer: ['no', 'yes']
                        hascreditcard: ['no', 'yes']
                        newcellphoneuser: ['no', 'yes']
                        notnewcellphoneuser: ['no', 'yes']
                        ownsmotorcycle: ['no', 'yes']
                        madecalltoretentionteam: ['no', 'yes']
                        creditrating: ['2-high', '3-good', '4-medium', '5-low', '1-highest', '6-verylow', '7-lowest']
                        prizmcode: ['rural', 'other', 'suburban', 'town']
                        occupation: ['professional', 'other', 'homemaker', 'crafts', 'self', 'clerical', 'retired', 'student']
                        maritalstatus: ['yes', 'no', 'unknown']


                    customer_shap_contributions:
                    Name: childreninhh, Type: REAL
                    Name: handsetrefurbished, Type: REAL
                    Name: handsetwebcapable, Type: REAL
                    Name: truckowner, Type: REAL
                    Name: rvowner, Type: REAL
                    Name: homeownership, Type: REAL
                    Name: buysviamailorder, Type: REAL
                    Name: respondstomailoffers, Type: REAL
                    Name: optoutmailings, Type: REAL
                    Name: nonustravel, Type: REAL
                    Name: ownscomputer, Type: REAL
                    Name: hascreditcard, Type: REAL
                    Name: newcellphoneuser, Type: REAL
                    Name: notnewcellphoneuser, Type: REAL
                    Name: ownsmotorcycle, Type: REAL
                    Name: madecalltoretentionteam, Type: REAL
                    Name: creditrating, Type: REAL
                    Name: prizmcode, Type: REAL
                    Name: occupation, Type: REAL
                    Name: maritalstatus, Type: REAL
                    Name: service_city, Type: REAL
                    Name: monthlyrevenue, Type: REAL
                    Name: monthlyminutes, Type: REAL
                    Name: totalrecurringcharge, Type: REAL
                    Name: directorassistedcalls, Type: REAL
                    Name: overageminutes, Type: REAL
                    Name: roamingcalls, Type: REAL
                    Name: percchangeminutes, Type: REAL
                    Name: percchangerevenues, Type: REAL
                    Name: droppedcalls, Type: REAL
                    Name: blockedcalls, Type: REAL
                    Name: unansweredcalls, Type: REAL
                    Name: customercarecalls, Type: REAL
                    Name: threewaycalls, Type: REAL
                    Name: receivedcalls, Type: REAL
                    Name: outboundcalls, Type: REAL
                    Name: inboundcalls, Type: REAL
                    Name: peakcallsinout, Type: REAL
                    Name: offpeakcallsinout, Type: REAL
                    Name: droppedblockedcalls, Type: REAL
                    Name: callforwardingcalls, Type: REAL
                    Name: callwaitingcalls, Type: REAL
                    Name: monthsinservice, Type: REAL
                    Name: uniquesubs, Type: REAL
                    Name: activesubs, Type: REAL
                    Name: handsets, Type: REAL
                    Name: handsetmodels, Type: REAL
                    Name: currentequipmentdays, Type: REAL
                    Name: agehh1, Type: REAL
                    Name: retentioncalls, Type: REAL
                    Name: retentionoffersaccepted, Type: REAL
                    Name: referralsmadebysubscriber, Type: REAL
                    Name: adjustmentstocreditrating, Type: REAL
                    Name: revenue_per_minute, Type: REAL
                    Name: total_calls, Type: REAL
                    Name: avg_call_duration, Type: REAL
                    Name: service_tenure, Type: REAL
                    Name: customer_support_interaction, Type: REAL
                    Name: handsetprice, Type: REAL
                    Name: incomegroup, Type: REAL
                    Name: customerid, Type: REAL

                        Distinct Values for Categorical Variables:

                    customer_counterfactuals:
                    Name: customerid, Type: REAL
                    Name: changes, Type: TEXT

                        Distinct Values for Categorical Variables:



                    
                question:
                {user_question}

                """
        context_query=gmodel.generate_content(context_prompt, stream=False)
        print(remove_sql_and_backticks(str(context_query.candidates[0].content.parts[0].text)))

        return str(context_query.candidates[0].content.parts[0].text)
