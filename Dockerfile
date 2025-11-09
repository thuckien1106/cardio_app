# ========== STAGE 1: SQL Server ==========
FROM mcr.microsoft.com/mssql/server:2022-latest AS mssqlbase
ENV ACCEPT_EULA=Y
ENV SA_PASSWORD=123
ENV MSSQL_PID=Express
ENV MSSQL_TCP_PORT=1433

# ========== STAGE 2: Python + Flask App ==========
FROM python:3.11-slim

# ---- Thiết lập cơ bản ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---- Cài công cụ hệ thống và ODBC driver ----
RUN apt-get update && apt-get install -y \
    netcat-openbsd wget curl gnupg ca-certificates unixodbc unixodbc-dev \
    libssl3 libcurl4 libkrb5-3 && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/ubuntu/22.04/prod jammy main" > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 mssql-tools18 && \
    echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- Copy SQL Server từ stage 1 ----
COPY --from=mssqlbase /var/opt/mssql /var/opt/mssql
COPY --from=mssqlbase /opt/mssql /opt/mssql

# ---- Copy code dự án ----
COPY . /app

# ---- Cài Python packages ----
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install gunicorn

# ---- Cấu hình môi trường ----
ENV ACCEPT_EULA=Y \
    SA_PASSWORD=123

EXPOSE 8080

# ---- Entrypoint ----
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]
