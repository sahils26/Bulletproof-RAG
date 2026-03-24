FROM python:3.12-slim

# Install uv for fast dependency management
RUN pip install uv

WORKDIR /app

# Copy dependency files first (Docker layer caching)
COPY pyproject.toml uv.lock ./
COPY packages/ packages/

# Install all workspace packages
RUN uv sync

# Copy remaining project files
COPY . .

CMD ["uv", "run", "python", "-m", "deeprag"]
