# This is a sample Python script.
import numpy as np
import torch

from recommend.Recommendation import GMF
from sentiment import SentimentAnalysis

def recommend():
    device = torch.device("cpu")
    new_model = GMF(6040, 3706, 8)
    new_model.load_state_dict(torch.load("recommend/NeuralCF/Pre_train/m1-1m_GMF.pkl"))
    items = [32, 34, 35, 30,50]
    u = 0
    users = np.full(len(items), u, dtype='int32')
    data = torch.tensor(np.vstack([users, np.array(items)]).T).to(device)
    output = new_model(data)
    print(output)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    comments = u'这个很棒，我很喜欢！'
    score = SentimentAnalysis.analyse(comments)
    print(comments)
    print(score)
    recommend()