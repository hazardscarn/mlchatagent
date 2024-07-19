FROM python:3.11.7

EXPOSE 8080
WORKDIR /app

COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "0🤖mlchatbot.py", "--server.port=8080", "--server.address=0.0.0.0"]