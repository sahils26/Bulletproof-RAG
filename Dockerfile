FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies using uv
RUN uv sync --frozen --no-cache

# Expose ports (dashboard)
EXPOSE 8000

# Default command (overridden by docker-compose)
CMD ["uv", "run", "deeprag.mcp_server"]
