FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r req.txt
EXPOSE $PORT
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]