#!/usr/bin/env bash
set +e

# 1) Start SQL Server in background
echo "🔹 Đang khởi động SQL Server..."
/opt/mssql/bin/sqlservr > /dev/null 2>&1 &
sleep 5

# 2) Wait for SQL Server to be ready
echo "⏳ Chờ SQL Server khởi động trong 90 giây..."
for i in {1..90}; do
    if /opt/mssql-tools18/bin/sqlcmd -S localhost,1433 -U sa -P "${SA_PASSWORD}" -Q "SELECT 1" > /dev/null 2>&1; then
        echo "✅ SQL Server đã sẵn sàng (sau ${i}s)"
        break
    fi
    sleep 1
done

echo "⏳ Đợi thêm 10s cho SQL ổn định..."
sleep 10


# 3) Nạp CVD.sql nếu lần đầu
if [ ! -f "/var/opt/mssql/.db_inited" ]; then
  echo "🛠️ Nạp CVD.sql để tạo DB CVD_App"
  if command -v sqlcmd >/dev/null 2>&1; then
      echo "✅ Đã tìm thấy sqlcmd, bắt đầu import DB..."
      sqlcmd -S localhost -U sa -P "${SA_PASSWORD}" -i /app/CVD.sql
  else
      echo "❌ sqlcmd chưa tồn tại trong PATH — kiểm tra mssql-tools18."
  fi
  touch /var/opt/mssql/.db_inited
  echo "✅ Khởi tạo DB xong"
fi

# 4) Run Flask app
echo "🚀 Khởi động Flask bằng Gunicorn..."
exec gunicorn -w 2 -b 0.0.0.0:${PORT:-8080} app:app
