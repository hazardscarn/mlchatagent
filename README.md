# mlchatagent
Agent Infrastructure to let users talk to ML model in English and use the model as a product with Gemini

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Usage](#usage)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

ML Chat Agent provides an infrastructure that allows users to interact with machine learning models through natural language. Leveraging the Gemini framework, this project aims to make ML models accessible and usable as products through a conversational interface. This project leverages Cell2Cell Telecom Public churn dataset available from Kaggle. With the xgboost model build on this dataset a chatbot is created that will let users ask questions to Xgboost Model and data infra in English and get the answer they look for. Some of the tools users can leverage is asking for reasons for churn, impact analysis, individual recommendation, vizualization etc


## Features

    - Subset Churn Reason
    - Subset CLV Impact Analysis
    - Subset Churn Impact Anlysis
    - Individual Counterfactual Analysis
    - Query and Answer on Model
    - Query and Answer on Data
    - Text to SQL converter
    - Query Reframer
    - Query to Visual converter

## Usage

To use ML chat Agent in local env:
    **Install requirements**:
    ```
        pip install -r requirements.txt
    ```

    **Start the agent**:

    streamlit run .\0🤖mlchatbot.py

To use ML chat Agent deployed in cloud run:

    https://mlychatapp-zye2qkdmta-uc.a.run.app/

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project leverages/modified upon code from open-source projects of 2024 Google LLC, licensed under the Apache 2.0 License. We would like to thank the authors of these projects for  their contributions.
- This project uses cell2cell data provided by Kaggle