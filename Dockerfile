FROM python:3.11
WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

# Expose port (if needed for web traffic)
EXPOSE 8000

# Firebase credentials (important - do not commit credentials directly!)
ENV GOOGLE_APPLICATION_CREDENTIALS=/firebase-credentials.json 

CMD ["python", "app.py"] 