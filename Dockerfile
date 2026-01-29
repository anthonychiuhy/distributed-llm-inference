FROM python:3.13-slim

WORKDIR /app

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-cache

# data
COPY data/conversations.json data/trace1.csv /app/data/
# logs
RUN mkdir logs
# code
COPY traffic_generator/main.py traffic_generator/app.py traffic_generator/__init__.py /app/traffic_generator/

# # Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "traffic_generator/app.py", "--port", "8080", "--host", "0.0.0.0"]