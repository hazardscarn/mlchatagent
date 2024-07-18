# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#Copyright 2024 Google LLC
#This agent was modified and built from the agent built by Google LLC in 2024
#Licensed under the Apache License, Version 2.0 (the "License");

"""
Provides the base class for all Agents 
"""

from abc import ABC
import google.generativeai as genai
import google.ai.generativelanguage as glm
import os
from dotenv import load_dotenv
from google.cloud import secretmanager

#load_dotenv()


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


#GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
project_id = "mlchatagent-429005"
secret_id = "GOOGLE_API_KEY"

# Access the secret
GOOGLE_API_KEY = access_secret_version(project_id, secret_id)

genai.configure(api_key=GOOGLE_API_KEY)

class Agent(ABC):
    """
    The core class for all Agents
    """

    agentType: str = "Agent"

    def __init__(self):
        """
        Args:
            PROJECT_ID (str | None): GCP Project Id.
            dataset_name (str): 
            TODO
        """
        self.model=genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
        
    def generate_llm_response(self,prompt):
        context_query = self.model.generate_content(prompt,stream=False)
        return str(context_query.candidates[0]).replace("```sql", "").replace("```", "")