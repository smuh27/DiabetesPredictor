FROM python:3.9.13

ADD main.py .

COPY diabetes.csv .

RUN pip install numpy scikit-learn matplotlib pandas seaborn 

CMD ["python", "./main.py"]