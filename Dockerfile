FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python run_pipeline.py

EXPOSE 3000

CMD ["streamlit", "run", "app/dashboard.py", "--server.port=3000", "--server.address=0.0.0.0", "--server.headless=true"]