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

with open('./llm_configs.yml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

##Load the Base Agent Structure
##This will act as base class for all the agents
class Agent(ABC):
    """
    The core class for all Agents
    """

    agentType: str = "Agent"

    def __init__(self,
                model_id:str):
        """
        Args:
            PROJECT_ID (str | None): GCP Project Id.
            dataset_name (str): 
            TODO
        """

        self.model_id = model_id 

        if model_id == 'code-bison-32k':
            self.model = CodeGenerationModel.from_pretrained('code-bison-32k')
        elif model_id == 'text-bison-32k':
            self.model = TextGenerationModel.from_pretrained('text-bison-32k')
        elif model_id == 'gemini-1.0-pro':
            self.model = GenerativeModel("gemini-1.0-pro")
        elif model_id == 'gemini-1.5-flash-001':
            self.model = GenerativeModel("gemini-1.5-flash-001")
        else:
            raise ValueError("Please specify a compatible model.")
        

##Embedding Agent is used to embed Known Good SQL,Table summarries and embed user question to retrive answers
class EmbedderAgent(Agent, ABC): 
    """ 
    This Agent generates embeddings 
    """ 

    agentType: str = "EmbedderAgent"

    def __init__(self, mode, embeddings_model='textembedding-gecko@002'): 
        if mode == 'vertex': 
            self.mode = mode 
            self.model = TextEmbeddingModel.from_pretrained(embeddings_model)

        else: raise ValueError('EmbedderAgent mode must be vertex')



    def create(self, question): 
        """Text embedding with a Large Language Model."""

        if self.mode == 'vertex': 
            if isinstance(question, str): 
                embeddings = self.model.get_embeddings([question])
                for embedding in embeddings:
                    vector = embedding.values
                return vector
            
            elif isinstance(question, list):  
                vector = list() 
                for q in question: 
                    embeddings = self.model.get_embeddings([q])

                    for embedding in embeddings:
                        vector.append(embedding.values) 
                return vector
            
            else: raise ValueError('Input must be either str or list')



class BQConnector(ABC):
    """
    Instantiates a BigQuery Connector.
    """
    connectorType: str = "Base"

    def __init__(self,
                 project_id:str,
                 region:str,
                 dataset_name:str):


        self.project_id = project_id
        self.region = region
        self.dataset_name = dataset_name
        self.opendataqna_dataset = dataset_name
        self.client=self.getconn()

    def getconn(self):
        client = bigquery.Client(project=self.project_id)
        return client
    
    def retrieve_df(self,query):
        query_cleaned=query.replace('```sql', '').replace('```', '').strip()
        return self.client.query_and_wait(query_cleaned).to_dataframe()
   
    
    def retrieve_matches(self, mode, schema, qe, similarity_threshold, limit): 
        """
        This function retrieves the most similar table_schema and column_schema.
        Modes can be either 'table', 'column', or 'example' 
        """
        matches = []

        if mode == 'table':
            sql = '''select base.content as tables_content from vector_search(TABLE `{}.table_details_embeddings`, "embedding", 
            (SELECT {} as qe), top_k=> {},distance_type=>"COSINE") where 1-distance > {} '''
        
        elif mode == 'column':
            sql='''select base.content as columns_content from vector_search(TABLE `{}.tablecolumn_details_embeddings`, "embedding",
            (SELECT {} as qe), top_k=> {}, distance_type=>"COSINE") where 1-distance > {} '''

        elif mode == 'example': 
            sql='''select base.example_user_question, base.example_generated_sql from vector_search ( TABLE `{}.example_prompt_sql_embeddings`, "embedding",
            (select {} as qe), top_k=> {}, distance_type=>"COSINE") where 1-distance > {} '''
    
        else: 
            ValueError("No valid mode. Must be either table, column, or example")
            name_txt = ''


        results=self.client.query_and_wait(sql.format('{}.{}'.format(self.project_id,self.opendataqna_dataset),qe,limit,similarity_threshold)).to_dataframe()


        # CHECK RESULTS 
        if len(results) == 0:
            print("Did not find any results. Adjust the query parameters.")

        if mode == 'table': 
            name_txt = ''
            for _ , r in results.iterrows():
                name_txt=name_txt+r["tables_content"]+"\n"

        elif mode == 'column': 
            name_txt = '' 
            for _ ,r in results.iterrows():
                name_txt=name_txt+r["columns_content"]+"\n"

        elif mode == 'example': 
            name_txt = ''
            for _ , r in results.iterrows():
                example_user_question=r["example_user_question"]
                example_sql=r["example_generated_sql"]
                name_txt = name_txt + "\n Example_question: "+example_user_question+ "; Example_SQL: "+example_sql

        else: 
            ValueError("No valid mode. Must be either table, column, or example")
            name_txt = ''

        matches.append(name_txt)
        

        return matches

    def getSimilarMatches(self, mode, schema, qe, num_matches, similarity_threshold):

        if mode == 'table': 
            #print(schema)
            match_result= self.retrieve_matches(mode, schema, qe, similarity_threshold, num_matches)
            match_result = match_result[0]
            # print(match_result)

        elif mode == 'column': 
            match_result= self.retrieve_matches(mode, schema, qe, similarity_threshold, num_matches)
            match_result = match_result[0]
        
        elif mode == 'example': 
            match_result= self.retrieve_matches(mode, schema, qe, similarity_threshold, num_matches)
            if len(match_result) == 0:
                match_result = None
            else:
                match_result = match_result[0]

        return match_result

    def getExactMatches(self, query):
        """Checks if the exact question is already present in the example SQL set"""
        check_history_sql=f"""SELECT example_user_question,example_generated_sql FROM {self.project_id}.{self.opendataqna_dataset}.example_prompt_sql_embeddings
                          WHERE lower(example_user_question) = lower('{query}') LIMIT 1; """

        exact_sql_history = self.client.query_and_wait(check_history_sql).to_dataframe()


        if exact_sql_history[exact_sql_history.columns[0]].count() != 0:
            sql_example_txt = ''
            exact_sql = ''
            for index, row in exact_sql_history.iterrows():
                example_user_question=row["example_user_question"]
                example_sql=row["example_generated_sql"]
                exact_sql=example_sql
                sql_example_txt = sql_example_txt + "\n Example_question: "+example_user_question+ "; Example_SQL: "+example_sql

            print("Found a matching question from the history!" + str(sql_example_txt))
            final_sql=exact_sql

        else: 
            print("No exact match found for the user prompt")
            final_sql = None

        return final_sql

    def test_sql_plan_execution(self, generated_sql):
        try:

            job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            query_job = self.client.query(generated_sql,job_config=job_config)
            # print(query_job)
            exec_result_df=("This query will process {} bytes.".format(query_job.total_bytes_processed))
            correct_sql = True
            print(exec_result_df)
            return correct_sql, exec_result_df
        except Exception as e:
            return False,str(e)


class BuildSQLAgent(Agent, ABC):
    """
    This Agent produces the SQL query 
    """

    agentType: str = "BuildSQLAgent"

    def build_sql(self,source_type, user_question,tables_schema,tables_detailed_schema, similar_sql,
                max_output_tokens=conf['build_sql']['max_output_tokens'],
                temperature=conf['build_sql']['temperature'],
                top_p=conf['build_sql']['top_p'],
                top_k=conf['build_sql']['top_k']):
         
        context_prompt = conf['build_sql']['prompt']
        context_prompt = context_prompt+f"""

        - Refer to the examples provided i.e. {similar_sql}


        Here are some examples of user-question and SQL queries:
        {similar_sql}


        question:
        {user_question}

        Table Schema:
        {tables_schema}

        Column Description:
        {tables_detailed_schema}


        NOTE: 
        ** Use customer_counterfactual_recommendations only if the question is about a specific individual customer recommendation. Using it for any other purpose will result in fatal errors.
        ** Never use counterfactuals table for churn reasoning for a subset. Doing so will result in fatal errors
        ** Counterfactuals table should only be used for individual customer recommendations or individual customer actions.
        ** For churn impact and CLV analysis always return all columns in dataset. Use select *, followed by modified columns.
        """
        
        if 'gemini' in self.model_id:
            # Generation Config
            config = GenerationConfig(
                max_output_tokens=max_output_tokens, temperature=temperature, top_p=top_p, top_k=top_k
            )

            # Generate text
            #print(f"prompt:{context_prompt}")
            context_query = self.model.generate_content(context_prompt, generation_config=config, stream=False)
            generated_sql = str(context_query.candidates[0].text)

        else:
            context_query = self.model.predict(context_prompt, max_output_tokens = max_output_tokens, temperature=temperature)
            generated_sql = str(context_query.candidates[0])


        return generated_sql
    

class ValidateSQLAgent(Agent, ABC): 
    """ 
    This Chat Agent checks the SQL for vailidity
    """ 

    agentType: str = "ValidateSQLAgent"


    def check(self, user_question, tables_schema, columns_schema, generated_sql):

        context_prompt = f"""

            Classify the SQL query: {generated_sql} as valid or invalid?

            Your job is to only check syntactic and semantic validity of the SQL query
            i.e to make sure the query is able to answer the user question in whole.

            Guidelines to be valid:
            - **DO NOT add entire schema with column names in the query. Use ONLY column names (e.g. select a.column from `project_id.owner.table_name` a)**
            - **When using column values to subset make sure the case matches. it is case sensitive and is lowercase**
            - Make sure the generated sql is able to answer the user question in whole. 
                For example if the question is to get churn impact analysis for a subset of customers,the query should return all the columns (select *) needed to perform the analysis.
            - If possible use subqueries to reduce the complexity of the query.
            - In case when all columns have to be returned, use * instead of column names.
            - If query returned some columns and not all, but user question is asking for all columns, then the query is invalid.
            - all join columns must be the same data_type.
            - Use table_alias.column_name when referring to columns. Example: dept_id=hr.dept_id
            - Capitalize the table names on SQL "where" condition.
            - Use the columns from the "SELECT" statement while framing "GROUP BY" block.
            - Always the table should be refered as schema.table_name.
            - Use all the non-aggregated columns from the "SELECT" statement while framing "GROUP BY" block.
            - Use counterfactual table only for individual customer recommendations and actions.
            - Never use counterfactuals for churn reasoning for a subset. Doing so will result in fat errors


        **NOTE:  
        - It is allowed when query is attempting to create a new column which already exists in the table. DO NOT consider this as an error.BIGQUERY will add new columns with _1,_2 etc.

        Parameters:
        - SQL query: {generated_sql}
        - table schema: {tables_schema}
        - column description: {columns_schema}


        Respond using a valid JSON format with two elements valid and errors. Remove ```json and ``` from the output:
        {{ "valid": true or false, "errors":errors }}

        Initial user question:
        {user_question}

        """        

        if self.model_id =='gemini-1.5-flash-001' or self.model_id == 'gemini-1.0-pro':
            context_query = self.model.generate_content(context_prompt, stream=False)
            generated_sql = str(context_query.candidates[0].text)

        else:
            context_query = self.model.predict(context_prompt, max_output_tokens = 8000, temperature=0)
            generated_sql = str(context_query.candidates[0])


        json_syntax_result = json.loads(str(generated_sql).replace("```json","").replace("```",""))

        # print('\n SQL Syntax Validity:' + str(json_syntax_result['valid']))
        # print('\n SQL Syntax Error Description:' +str(json_syntax_result['errors']) + '\n')
        
        return json_syntax_result
    

class ResponseAgent(Agent, ABC): 
    """
    A specialized Chat Agent designed to provide natural language responses to user questions based on SQL query results.

    This agent acts as a bridge between structured data returned from SQL queries and the user's natural language input. It leverages a language model (e.g., Gemini Pro or others) to interpret the query results and craft informative, human-readable answers.

    Key Features:

    * **Natural Language Generation:**  Transforms SQL results into user-friendly responses.
    * **Model Flexibility:** Supports multiple language models (currently handles Gemini Pro and others with slight adjustments).
    * **Contextual Understanding:**  Incorporates the user's original question and the SQL results to provide accurate and relevant answers. 

    Attributes:
        agentType (str): Identifies this agent as a "ResponseAgent".
        model_id (str): Indicates the specific language model being used.

    Methods:
        run(user_question, sql_result):
            Generates a natural language response based on the user's question and the SQL results.
            
    Example:

        response_agent = ResponseAgent(model_id='gemini-1.0-pro')
        response = response_agent.run("How many customers are in California?", sql_result) 
        # response might be: "There are 153 customers in California based on the data."
    """

    agentType: str = "ResponseAgent"

    # TODO: Make the LLM Validator optional
    def run(self, user_question, sql_result):

        context_prompt = f"""

            You are a Data Assistant that helps to answer users' questions on their data within their databases.
            The user has provided the following question in natural language: "{str(user_question)}"

            The system has returned the following result after running the SQL query: "{str(sql_result)}".

            Provide a natural sounding response to the user to answer the question with the SQL result provided to you. 
        """

        
        if self.model_id =='gemini-1.5-flash-001' or self.model_id == 'gemini-1.0-pro':
            context_query = self.model.generate_content(context_prompt, stream=False)
            generated_sql = str(context_query.candidates[0].text)

        else:
            context_query = self.model.predict(context_prompt, max_output_tokens = 8000, temperature=0)
            generated_sql = str(context_query.candidates[0])
        
        return generated_sql


class DebugSQLAgent(Agent, ABC): 
    """ 
    This Chat Agent runs the debugging loop.
    """ 

    agentType: str = "DebugSQLAgent"

    def __init__(self, chat_model_id = 'gemini-1.5-flash-001'): 
        self.chat_model_id = chat_model_id
        # self.model = CodeChatModel.from_pretrained("codechat-bison-32k")


    def init_chat(self, tables_schema,tables_detailed_schema,sql_example="-No examples provided..-"):
        context_prompt = f"""
        You are an BigQuery SQL guru. This session is trying to troubleshoot an BigQuery SQL query.  As the user provides versions of the query and the errors returned by BigQuery,
        return a new alternative SQL query that fixes the errors. It is important that the query still answer the original question.


        Guidelines:
        - Join as minimal tables as possible.
        - When joining tables ensure all join columns are the same data_type.
        - Analyze the database and the table schema provided as parameters and undestand the relations (column and table relations).
        - Use always SAFE_CAST. If performing a SAFE_CAST, use only Bigquery supported datatypes.
        - Always SAFE_CAST and then use aggregate functions
        - Don't include any comments in code.
        - Remove ```sql and ``` from the output and generate the SQL in single line.
        - Tables should be refered to using a fully qualified name with enclosed in ticks (`) e.g. `project_id.owner.table_name`.
        - Use all the non-aggregated columns from the "SELECT" statement while framing "GROUP BY" block.
        - Return syntactically and symantically correct SQL for BigQuery with proper relation mapping i.e project_id, owner, table and column relation.
        - Use ONLY the column names (column_name) mentioned in Table Schema. DO NOT USE any other column names outside of this.
        - Associate column_name mentioned in Table Schema only to the table_name specified under Table Schema.
        - Use SQL 'AS' statement to assign a new name temporarily to a table column or even a table wherever needed.
        - Table names are case sensitive. DO NOT uppercase or lowercase the table names.
        - Always enclose subqueries and union queries in brackets.
            - Never limit rows in the query unless specifically asked to do so.
        - Return all columns (select *) whenever possible except in vizualization/plot tasks.
        - Unless it's specifically asked to return only certain columns or if it is  vizualization/plot task always use select *
        - Never use select * for a plot or distribution task. Failure to do this will result in fatal error
        - It is a fatal mistake not to return all columns when asked to do so.
        - Make sure filters, if requested are applied to the correct columns and in the correct order.
        - When you are asked to modify a column, make sure you do it as requested. Failing to do so can result in critical errors.
        - When modifying or changing a column, ensure you do so within the SELECT clause of the query. Failing to do this can result in critical errors.
        - When modifying or changing a column,DO NOT rename them. We are just trying to update it's value.Changing the column name can result in critical errors.
        - NEVER GROUP BY a INT or FLOAT column unless it is specified you can group the column
        - Never group a query by more than 5 columns. Doing so will create Fatal errors
        - Always bucket a conitnous variable into 5 or 10 levels and then group by buckets if necessary
        - **When using column values to subset make sure the case matches. it is case sensitive and is lowercase**
        - Join as few tables as possible and ensure all join columns have the same data type.
        - Analyze the database and table schema provided as parameters to understand the relations.
        - Always use SAFE_CAST and only BigQuery-supported data types.
        - Use table aliases and reference columns like t1.column without the full path
        - Do not cast a column of string datatype to integer datatype unless necessary
        - Do not include any comments in the code.
        - Generate the SQL in a single line without ` ```sql ` and ` ``` `.
        - Refer to tables using fully qualified names enclosed in ticks (e.g., `project_id.owner.table_name`).
        - Use only column names mentioned in the Table Schema. Do not use any other column names.
        - Associate column names mentioned in the Table Schema only with the specified table_name.
        - Use the SQL 'AS' statement to assign a new name temporarily to a table column or table wherever needed.
        - Table names are case-sensitive. Do not change the case of table names.
        - Enclose subqueries and union queries in brackets.
        - Use column values in a case-sensitive manner for subsetting.
        - Create new columns or summaries with appropriate names (e.g., SUM(column) AS total_column, COUNT(column) AS total_count).
        - Use subqueries for bucketing numeric columns.
        - Use NTILE and window functions for bucketing and ranking (e.g., NTILE(10) OVER (ORDER BY column_name) AS bucket).
        - Create meaningful bucket ranges with min and max values for easier interpretation. For example:
        - First, use a subquery to create buckets using NTILE and include the column for bucketing (e.g., totalrecurringcharge).
        - Then, create another subquery to calculate the min and max values for each bucket.
        - Finally, join these subqueries and format the bucket ranges as strings (e.g., CONCAT('$', FORMAT('%f', min_value), ' - $', FORMAT('%f', max_value))).
        - Use window functions to calculate running totals, averages, and other aggregations.
        - Use counterfactual table only for individual customer reccomendations.                  
        - Refer to the examples provided i.e. {sql_example}

        Parameters:
        - table metadata: {tables_schema}
        - column metadata: {tables_detailed_schema}
        - SQL example: {sql_example}

        """

        
        if self.chat_model_id == 'codechat-bison-32k':
            chat_model = CodeChatModel.from_pretrained("codechat-bison-32k")
            chat_session = chat_model.start_chat(context=context_prompt)
        elif self.chat_model_id == 'gemini-1.0-pro':
            chat_model = GenerativeModel("gemini-1.0-pro-001")
            chat_session = chat_model.start_chat(response_validation=False)
            chat_session.send_message(context_prompt)
        elif self.chat_model_id == 'gemini-1.5-flash-001':
            chat_model = GenerativeModel("gemini-1.5-flash-001")
            chat_session = chat_model.start_chat(response_validation=False)
            chat_session.send_message(context_prompt)
        elif self.chat_model_id == 'gemini-ultra':
            chat_model = GenerativeModel("gemini-1.0-ultra-001")
            chat_session = chat_model.start_chat(response_validation=False)
            chat_session.send_message(context_prompt)
        else:
            raise ValueError('Invalid chat_model_id')
        
        return chat_session


    def rewrite_sql_chat(self, chat_session, question, error_df):


        context_prompt = f"""
            What is an alternative SQL statement to address the error mentioned below?
            Present a different SQL from previous ones. It is important that the query still answer the original question.
            All columns selected must be present on tables mentioned on the join section.
            Avoid repeating suggestions.

            Original SQL:
            {question}

            Error:
            {error_df}

            """

        if self.chat_model_id =='codechat-bison-32k':
            response = chat_session.send_message(context_prompt)
            resp_return = (str(response.candidates[0])).replace("```sql", "").replace("```", "")
        elif self.chat_model_id =='gemini-1.0-pro' or self.chat_model_id =='gemini-1.5-flash-001':
            response = chat_session.send_message(context_prompt, stream=False)
            resp_return = (str(response.text)).replace("```sql", "").replace("```", "")
        elif self.chat_model_id == 'gemini-ultra':
            response = chat_session.send_message(context_prompt, stream=False)
            resp_return = (str(response.text)).replace("```sql", "").replace("```", "")
        else:
            raise ValueError('Invalid chat_model_id')

        return resp_return


    def start_debugger  (self,
                        source_type,
                        query,
                        user_question, 
                        SQLChecker,
                        tables_schema, 
                        tables_detailed_schema,
                        AUDIT_TEXT,
                        project_id, 
                        similar_sql="-No examples provided..-", 
                        DEBUGGING_ROUNDS = 2,
                        LLM_VALIDATION=True):
        i = 0  
        STOP = False 
        invalid_response = False 
        chat_session = self.init_chat(tables_schema,tables_detailed_schema,similar_sql)
        sql = query.replace("```sql","").replace("```","").replace("EXPLAIN ANALYZE ","")
        json_syntax_result = {}
        json_syntax_result['valid'] = True 
        connector=BQConnector(project_id=project_id, region='us-central1', dataset_name='telecom_churn')

        AUDIT_TEXT=AUDIT_TEXT+"\n\nEntering the debugging steps!"
        while (not STOP):

            # Check if LLM Validation is enabled 
            if LLM_VALIDATION: 
                # sql = query.replace("```sql","").replace("```","").replace("EXPLAIN ANALYZE ","")
                json_syntax_result = SQLChecker.check(user_question,tables_schema,tables_detailed_schema, sql) 

            else: 
                json_syntax_result['valid'] = True 

            if json_syntax_result['valid'] is True:
                # Testing SQL Execution
                if LLM_VALIDATION: 
                    AUDIT_TEXT=AUDIT_TEXT+"\nGenerated SQL is syntactically correct as per LLM Validation!"
                
                else: 
                    AUDIT_TEXT=AUDIT_TEXT+"\nLLM Validation is deactivated. Jumping directly to dry run execution."

                    
                correct_sql, exec_result_df = connector.test_sql_plan_execution(sql)
                print("exec_result_df:" + exec_result_df)
                if not correct_sql:
                        AUDIT_TEXT=AUDIT_TEXT+"\nGenerated SQL failed on execution! Here is the feedback from bigquery dryrun/ explain plan:  \n" + str(exec_result_df)
                        rewrite_result = self.rewrite_sql_chat(chat_session, sql, exec_result_df)
                        print('\n Rewritten and Cleaned SQL: ' + str(rewrite_result))
                        AUDIT_TEXT=AUDIT_TEXT+"\nRewritten and Cleaned SQL: \n' + str({rewrite_result})"
                        sql = str(rewrite_result).replace("```sql","").replace("```","").replace("EXPLAIN ANALYZE ","")

                else: STOP = True
            else:
                print(f'\nGenerated qeury failed on syntax check as per LLM Validation!\nError Message from LLM:  {json_syntax_result} \nRewriting the query...')
                AUDIT_TEXT=AUDIT_TEXT+'\nGenerated qeury failed on syntax check as per LLM Validation! \nError Message from LLM:  '+ str(json_syntax_result) + '\nRewriting the query...'
                
                syntax_err_df = pd.read_json(json.dumps(json_syntax_result))
                rewrite_result=self.rewrite_sql_chat(chat_session, sql, syntax_err_df)
                print(rewrite_result)
                AUDIT_TEXT=AUDIT_TEXT+'\n Rewritten SQL: ' + str(rewrite_result)
                sql=str(rewrite_result).replace("```sql","").replace("```","").replace("EXPLAIN ANALYZE ","")
            i+=1
            if i > DEBUGGING_ROUNDS:
                AUDIT_TEXT=AUDIT_TEXT+ "Exceeded the number of iterations for correction!"
                AUDIT_TEXT=AUDIT_TEXT+ "The generated SQL can be invalid!"
                STOP = True
                invalid_response=True
            # After the while is completed
        if i > DEBUGGING_ROUNDS:
            invalid_response=True
        # print(AUDIT_TEXT)
        return sql, invalid_response, AUDIT_TEXT


class QueryRefiller(Agent, ABC): 
    """ 
    This Agent makes sure the query uses select * and not subset of columns for specific processes
    """ 

    agentType: str = "QueryFillerAgent"


    def check(self, generated_sql):

        context_prompt = f"""
        Your task is to check if the SQL query contains "SELECT *" and modify it if necessary:
        - If the query does not have "SELECT *" or "select alias.*", modify it to include "SELECT *".
        - If the query already has "SELECT *" or "select alias.*", return the query as is.
        - Include any modified or new columns in the query.
        - Ensure the query remains valid SQL.

        Examples:

        Example 1 - When "SELECT *" is missing:
        Original query: SELECT eco-sector-422622-b5.telecom_churn.customer_shap_data.shapvalue_agehh1, eco-sector-422622-b5.telecom_churn.customer_shap_data.shapvalue_childreninhh FROM eco-sector-422622-b5.telecom_churn.customer_shap_data INNER JOIN eco-sector-422622-b5.telecom_churn.customer_data ON eco-sector-422622-b5.telecom_churn.customer_shap_data.customerid = eco-sector-422622-b5.telecom_churn.customer_data.customerid WHERE eco-sector-422622-b5.telecom_churn.customer_data.agehh1 > 50 AND eco-sector-422622-b5.telecom_churn.customer_data.childreninhh = TRUE
        Reframed query: SELECT t1.* FROM eco-sector-422622-b5.telecom_churn.customer_shap_data t1 INNER JOIN eco-sector-422622-b5.telecom_churn.customer_data t2 ON t1.customerid = t2.customerid WHERE t2.agehh1 > 50 AND t2.childreninhh = TRUE  

        Example 2 - When "SELECT *" is already there:
        Original query: SELECT * FROM eco-sector-422622-b5.telecom_churn.customer_data
        Reframed query: SELECT * FROM eco-sector-422622-b5.telecom_churn.customer_data

        Example 3 - When "SELECT *" is already there along with a modified column:
        Original query: SELECT *,(revenue_per_minute-0.1) as revenue_per_minute FROM eco-sector-422622-b5.telecom_churn.customer_data
        Reframed query: SELECT *,(revenue_per_minute-0.1) as revenue_per_minute FROM eco-sector-422622-b5.telecom_churn.customer_data

        Example 4 - When "SELECT alias.*" is already there:
        Original query: SELECT t1.* FROM mlchatagent-429005.telecom_churn.customer_shap_data t1 JOIN mlchatagent-429005.telecom_churn.customer_data t2 ON t1.customerid = t2.customerid WHERE t2.ageinhh1 < 20
        Reframed query: SELECT t1.* FROM mlchatagent-429005.telecom_churn.customer_shap_data t1 JOIN mlchatagent-429005.telecom_churn.customer_data t2 ON t1.customerid = t2.customerid WHERE t2.ageinhh1 < 20

        Here is the SQL generated:
        {generated_sql}

        Output: reframed_query

        Note:
        - Output only the reframed query.
        - Do not add any extra text, SQL keywords, or symbols (e.g., "sql", "```", "output").
        """



        if self.model_id =='gemini-1.5-flash-001' or self.model_id == 'gemini-1.0-pro':
            context_query = self.model.generate_content(context_prompt, stream=False)
            reformed_sql = str(context_query.candidates[0].text)

        else:
            context_query = self.model.predict(context_prompt, max_output_tokens = 8000, temperature=0)
            reformed_sql = str(context_query.candidates[0])

        return reformed_sql