FROM continuumio/miniconda3

WORKDIR /app
COPY requirements.txt .

RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y

RUN curl -fsSL https://get.deta.dev/cli.sh | sh
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
EXPOSE 8002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]