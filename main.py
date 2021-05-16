import pandas as pd
import numpy as np
import torch
from torch import nn

import datetime

# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv",
                        help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv",
                        help="input the generation data path")
    parser.add_argument(
        "--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv",
                        help="output the bids path")

    return parser.parse_args()


def output(path, data):

    df = pd.DataFrame(
        data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)


def getData(path1, path2):

    dateFormatter = "%Y-%m-%d %H:%M:%S"
    df_consumption = pd.read_csv(path1, encoding='utf-8')
    df_generation = pd.read_csv(path2, encoding='utf-8')
    h = len(df_consumption)
    gen, con = [], []
    for i in range(0, h):
        gen.append(df_generation["generation"][i])
        con.append(df_consumption["consumption"][i])

        last_date = datetime.datetime.strptime(
            df_consumption["time"][i], dateFormatter)
    gen = np.array(gen, dtype='float32')
    con = np.array(con, dtype='float32')
    gen = gen.reshape(-1, 1, 168)
    con = con.reshape(-1, 1, 168)

    gen = torch.from_numpy(gen)
    con = torch.from_numpy(con)
    return gen, con, last_date


class GRUNet(nn.Module):
    def __init__(self, input_size):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        self.out = nn.Sequential(
            nn.Linear(128, 24)
        )

    def forward(self, x):
        x, _ = self.rnn(x)
        s, b, h = x.shape
        x = x.view(s*b, h)
        x = self.out(x)
        x = x.view(s, b, -1)

        return x


def test_model(gen, con):
    model = GRUNet(168)
    model.load_state_dict(torch.load('./consumptionV1.pt'))
    model.eval()
    pd_con = model(con)
    model = GRUNet(168)
    model.load_state_dict(torch.load('./generationV1.pt'))
    model.eval()
    pd_gen = model(gen)
    print(pd_con.squeeze())
    print(pd_gen.squeeze())
    return pd_con.squeeze(), pd_gen.squeeze()


def rule(predict_consumption, predict_generation, last_date):

    ans = []
    for i in range(0, len(predict_consumption)):
        last_date = last_date + datetime.timedelta(hours=1)
        if predict_consumption[i] - predict_generation[i] > 0:
            ans.append([str(last_date), "buy", 2.3, 1])

        elif predict_consumption[i] - predict_generation[i] < 0:
            ans.append([last_date, "sell", 1.5, 1])

    return ans


if __name__ == "__main__":
    args = config()
    # try:
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen, con, last_date = getData(config().consumption, config().generation)
    pd_con, pd_gen = test_model(gen, con)
    data = rule(pd_con, pd_gen, last_date)
    # except:
    #     print('error')
    #     data = []
    output(args.output, data)
