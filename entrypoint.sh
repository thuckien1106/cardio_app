#!/usr/bin/env bash
set -e

echo "🔹 Khởi động SQL Server..."
/opt/mssql/bin/sqlservr > /tmp/sql.log 2>&1 &

echo "⏳ Chờ SQL Server khởi động (tối đa 90s)..."
for i in {1..90}; do
    if /opt/mssql-tools18/bin/sqlcmd -S localhost,1433 -U sa -P "123" -Q "SELECT 1" > /dev/null 2>&1; then
        echo "✅ SQL Server sẵn sàng (sau ${i}s)"
        break
    fi
    sleep 1
done

# --- Tạo DB nếu chưa có ---
echo "🛠️ Tạo cơ sở dữ liệu nếu chưa tồn tại..."
/opt/mssql-tools18/bin/sqlcmd -S localhost,1433 -U sa -P "123" -i /app/CVD.sql || echo "⚠️ Bỏ qua lỗi nếu DB đã tồn tại."

# --- Xác nhận DB ---
echo "📂 Danh sách DB hiện có:"
/opt/mssql-tools18/bin/sqlcmd -S localhost,1433 -U sa -P "123" -Q "SELECT name FROM sys.databases"

# --- Kiểm tra Flask file ---
if [ ! -f /app/app.py ]; then
  echo "❌ Không tìm thấy /app/app.py, dừng lại."
  exit 1
fi

echo "🚀 Khởi động Flask bằng Gunicorn..."
cd /app
exec gunicorn -w 2 -b 0.0.0.0:${PORT:-8080} app:app
