import numpy as np
import torch
from flask import Flask
from flask import request

from recommend.Recommendation import GMF
from sentiment import SentimentAnalysis

app = Flask(__name__)

@app.get("/sentiment")
def sentiment_analysis():
    context = request.args['context']
    score = SentimentAnalysis.analyse(context)
    return str(score)
@app.get("/recommend")
def recommend():
    userId = request.args['userId']
    device = torch.device("cpu")
    new_model = GMF(6040, 3706, 8)
    new_model.load_state_dict(torch.load("recommend/NeuralCF/Pre_train/m1-1m_GMF.pkl"))
    items = [32, 34, 35, 30,50]
    u = userId
    users = np.full(len(items), u, dtype='int32')
    data = torch.tensor(np.vstack([users, np.array(items)]).T).to(device)
    output = new_model(data)
    result = output.detach().numpy()
    result = np.array(result).tolist()
    print(result)
    return result