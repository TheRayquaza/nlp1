FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV PYTHONPATH="/app:$PYTHONPATH"

RUN apt-get update --no-install-recommends \
    && apt-get install --no-install-recommends -y \
       python3 git make cron bzip2 tzdata locales \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 \
    && cp /usr/share/zoneinfo/Europe/Paris /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

ARG MODEL_NAME

WORKDIR /app

RUN python3 -m venv .venv \
    && . .venv/bin/activate \
    && pip install --upgrade pip

COPY models/model_${MODEL_NAME}/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY ./data/RAW_recipes.csv /app/data/RAW_recipes.csv
COPY models/exchange /app/exchange
COPY models/model_${MODEL_NAME} /app/model_${MODEL_NAME}

CMD ["python3", "src/__main__.py"]
