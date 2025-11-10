# =========================================================
# 🫀 CARDIO-APP (Flask + SQL Server) - 2025 SAFE VERSION
# =========================================================
FROM mcr.microsoft.com/mssql/server:2022-latest AS mssqlbase
ENV SA_PASSWORD=123
ENV ACCEPT_EULA=Y
ENV MSSQL_PID=Express
COPY CVD.sql /tmp/CVD.sql

# =========================================================
FROM python:3.11-slim

WORKDIR /app

# --- Cài đặt thư viện hệ thống ---
RUN apt-get update && apt-get install -y \
    curl gnupg2 ca-certificates unixodbc unixodbc-dev libgssapi-krb5-2 \
    && rm -rf /var/lib/apt/lists/*

# --- Thêm Microsoft repo bằng keyrings (fix lỗi apt-key) ---
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | \
        gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 mssql-tools18 && \
    rm -rf /var/lib/apt/lists/*

# --- Cài thư viện Python ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# --- Copy source ---
COPY . .

# --- Cấp quyền script ---
RUN chmod +x /app/entrypoint.sh

EXPOSE 8080
CMD ["bash", "/app/entrypoint.sh"]
