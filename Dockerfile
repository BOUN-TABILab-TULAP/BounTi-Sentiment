FROM docker.io/huggingface/transformers-pytorch-cpu:3.5.1

COPY . /workspace
WORKDIR /workspace

RUN apt-get update -y
RUN apt-get install wget -y

RUN pip3 install -r requirements.txt
RUN wget -q -r -nH --cut-dirs=1  --no-parent -e robots=off https://nlp.cmpe.boun.edu.tr/staticFiles/berturk/

# CMD tail -f /dev/null

EXPOSE 5000
ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]




