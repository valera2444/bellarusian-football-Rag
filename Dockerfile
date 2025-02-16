FROM python:3.12-slim

COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade -r ./requirements.txt

ARG CACHEBUST=0
COPY ./app ./app

RUN pip install "psycopg[binary,pool]"

CMD ["fastapi", "run", "app/main.py", "--port", "5050"]