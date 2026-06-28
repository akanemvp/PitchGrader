web: mkdir -p /app/data; (test -s /app/data/statcast.db || wget -q "https://arlington-atlas-trustee-ali.trycloudflare.com/statcast.db" -O /app/data/statcast.db) & gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300
worker: python main.py live
