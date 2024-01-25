FROM python:3.8.13

ADD requirements.txt .
RUN pip3 install -r ./requirements.txt

WORKDIR /home/data

ENTRYPOINT ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--port=8000", "--no-browser"]

