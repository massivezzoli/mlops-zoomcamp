FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app
COPY [ "Pipfile", "Pipfile.lock" , "starter.py", "./"] 

RUN pip install pipenv 
RUN pipenv install --ignore--pipfile