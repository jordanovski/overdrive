FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OVERDRIVE_HUB_ROOT=/models \
    OVERDRIVE_WEB_PORT=8080

WORKDIR /app

COPY LICENSE README.md pyproject.toml ./
COPY scr ./scr

RUN python -m pip install --upgrade pip \
    && python -m pip install .

EXPOSE 8080

CMD ["sh", "-c", "exec overdrive --hub-root \"$OVERDRIVE_HUB_ROOT\" web --host 0.0.0.0 --port \"$OVERDRIVE_WEB_PORT\""]