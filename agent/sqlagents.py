#Copyright 2024 Google LLC
#This agent was modified and built from the agent logic built by Google LLC in 2024
#Licensed under the Apache License, Version 2.0 (the "License");


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
            """
        context_prompt = context_prompt + conf['validate_sql']['prompt']
        context_prompt=context_prompt+f"""  
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
        
        return json_syntax_result
    
class DebugSQLAgent(Agent, ABC): 
    """ 
    This Chat Agent runs the debugging loop.
    """ 

    agentType: str = "DebugSQLAgent"

    def __init__(self, chat_model_id = 'gemini-1.5-flash-001'): 
        self.chat_model_id = chat_model_id
        # self.model = CodeChatModel.from_pretrained("codechat-bison-32k")


    def init_chat(self, tables_schema,tables_detailed_schema,sql_example="-No examples provided..-"):
        context_prompt = conf['debug_sql']['prompt']
        context_prompt = context_prompt+f"""

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

        context_prompt = conf['query_filler']['prompt']+f"""
        
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