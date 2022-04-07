from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from flask import Flask, json, g, request, jsonify, json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "dbmdz/bert-base-turkish-128k-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

modelpath = "berturk/"
model = AutoModelForSequenceClassification.from_pretrained(modelpath, num_labels=3).to(device)

categories = ['negative', 'neutral', 'positive']
max_seq_length = 50

model.eval()

def to_id(text):
    return torch.tensor(tokenizer.encode(text))

def evaluater(sentences):
    with torch.no_grad():
        input_ids_all = []
        input_attentions = []
        for sentence in sentences.splitlines():
            input_ids_raw = to_id(sentence)[:max_seq_length]
            input_attention = torch.LongTensor([1]*len(input_ids_raw)+[0]*(max_seq_length-len(input_ids_raw)))
            input_attention = input_attention.to(device)
            input_ids = torch.cat((input_ids_raw, torch.tensor([0]*(max_seq_length-len(input_ids_raw)))), 0).to(device)
            input_ids_all.append(input_ids)
            input_attentions.append(input_attention)
        input_attentions = torch.stack(input_attentions)
        input_ids_all = torch.stack(input_ids_all)
        outputs = model(input_ids_all, input_attentions)
        class_ids = np.argmax(outputs[0].cpu(), axis=1)
        # print(class_ids)
        # for class_id in class_ids:
        #     print(categories[class_id])
        return [categories[class_id] for class_id in class_ids]


app = Flask(__name__)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    json_data = json.loads(request.data)

    result = {"text": "\n".join(evaluater(json_data['textarea']))}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0')
