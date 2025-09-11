# Showroom AI Assistant Backend
FROM python:3.11-slim

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Install UV package manager for faster Python package installs
# RUN pip install uv

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt /app/
RUN pip install -r requirements.txt

# # Install MCP server for Kubernetes operations
# RUN pip install mcp-server-kubernetes

RUN pip install uv

# Copy application code
COPY backend/ /app/
COPY config/ /app/config/

# Copy workshop content for RAG processing
COPY content/ /app/content/

# Create directories for static files
RUN mkdir -p /app/www /app/pdfs

# # Create non-root user for security
# RUN useradd -r -u 1001 -m -c "showroom app user" -s /bin/bash appuser && \
#     chown -R appuser:appuser /app
# USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CONTENT_DIR=/app/content
ENV STATIC_DIR=/app/www
ENV PDF_DIR=/app/content/modules/ROOT/assets/techdocs
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]