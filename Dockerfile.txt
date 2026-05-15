FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip setuptools==69.5.1

RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]