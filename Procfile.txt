web: gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:10000 --timeout 120
