import os
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch


def create_skill_graph(df):
    # Cache and load if exists.
    if os.path.exists("skill_graph.pickle"):
        print("Using existing skill graph...")
        return pickle.load(open("skill_graph.pickle", "rb")), pickle.load(
            open("skill_dict.pickle", "rb")
        )

    # Pre-processing steps to remove unwanted responses and group into buckets.
    print("Constructing skill graph...")
    df = df[~df["skill_name"].isna()]
    grouped = df.groupby("user_id")["skill_name"].agg(list)
    uniques = list(df["skill_name"].unique())

    # Count ordered co-occurrences in each student sequence.
    skill_cooccurs = {
        skill_name: np.zeros(df["skill_name"].nunique()) for skill_name in uniques
    }
    for seq in tqdm(grouped.values):
        cooccur = np.zeros(df["skill_name"].nunique())
        for s in reversed(seq):
            cooccur[uniques.index(s)] += 1
            skill_cooccurs[s] = skill_cooccurs[s] + cooccur

    # Normalize distribution and round to remove noise.
    skill_cooccurs = {k: (v / sum(v)).round(1) for k, v in skill_cooccurs.items()}
    dod = {}
    for i, (skill_name, edges) in enumerate(skill_cooccurs.items()):
        dod[i] = {}
        for j, e in enumerate(edges):
            if e > 0:
                dod[i][j] = {"weight": e}

    # Connect nodes in digraph with forward co-occurrence.
    skill_graph = nx.from_dict_of_dicts(dod)
    skill_dict = dict(zip(uniques, range(len(uniques))))

    # Save and cache graph for future usage.
    pickle.dump(skill_graph, open("skill_graph.pickle", "wb"))
    pickle.dump(skill_dict, open("skill_dict.pickle", "wb"))

    return skill_graph, skill_dict


def preprocess_data(data, block_size):
    """
    Pre-process data and pad to the maximum length.
    """
    features = ["skill_id", "correct"]
    seqs = data.groupby(["user_id"]).apply(lambda x: x[features].values.tolist())

    # ensure sequence is not too long
    length = min(max(seqs.str.len()), block_size)
    seqs = seqs.apply(
        lambda s: s[:length]
        + (length - min(len(s), length)) * [[-1000] * len(features)]
    )

    return seqs


def construct_batches(raw_data, batch_size, block_size, epoch=0, val=False):
    """
    Construct batches based on tabular KT data with user_id, skill_id, and
    correctness. Pads to the minimum of the maximum sequence length and the
    block size of the transformer.
    """
    np.random.seed(epoch)
    user_ids = raw_data["user_id"].unique()

    # Loop until one epoch of training.
    for _ in range(len(user_ids) // batch_size):
        user_idx = (
            raw_data["user_id"].sample(batch_size).unique()
            if not val
            else user_ids[_ * (batch_size // 2) : (_ + 1) * (batch_size // 2)]
        )

        filtered_data = raw_data[raw_data["user_id"].isin(user_idx)].sort_values(
            ["user_id", "order_id"]
        )

        batch_preprocessed = preprocess_data(filtered_data)

        batch = np.array(batch_preprocessed.to_list())

        # Next token prediction.
        X = torch.tensor(
            batch[:, :-1, ..., :], requires_grad=True, dtype=torch.float32
        ).cuda()

        y = torch.tensor(
            batch[:, 1:, ..., [0, 1]], requires_grad=True, dtype=torch.float32
        ).cuda()

        for i in range(X.shape[1] // block_size + 1):
            if X[:, i * block_size : (i + 1) * block_size].shape[1] > 0:
                yield [
                    X[:, i * block_size : (i + 1) * block_size],
                    y[:, i * block_size : (i + 1) * block_size],
                ]


def evaluate(model, skill_net, skill_graph, batches):
    ypred, ytrue = [], []
    for X, y in batches:
        mask = y[..., -1] != -1000
        all_skill_embd = skill_net(
            torch.arange(110).cuda(),
            skill_graph.edge_index.cuda(),
            skill_graph.weight.cuda().float(),
        )
        skill_embd = all_skill_embd[
            torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()
        ]
        ohe = torch.eye(110).cuda()[
            torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()
        ]
        X = torch.cat([X, skill_embd, ohe], dim=-1)
        corrects = model.forward(X, y[..., 0])[mask]
        y = y[..., -1].unsqueeze(-1)[mask]
        ypred.append(corrects.ravel().detach().cpu().numpy())
        ytrue.append(y.ravel().detach().cpu().numpy())
    ypred = np.concatenate(ypred)
    ytrue = np.concatenate(ytrue)

    return ypred, ytrue  # roc_auc_score(ytrue, ypred)


def train_test_split(data, train_split):
    """
    Performs a deterministic train-test split based on the tabular data provided.
    Note that this function needs to be called twice to perform a train-val-test
    split as desired.

    Arguments:
      - data: tabular KT dataset (pd.DataFrame)
      - train_spit

    Returns:
      - data_train: training dataset
      - data_val: validation/testing dataset
    """
    np.random.seed(42)
    data = data.set_index(["user_id", "skill_name"])
    idx = np.random.permutation(data.index.unique())
    train_idx, test_idx = (
        idx[: int(train_split * len(idx))],
        idx[int(train_split * len(idx)) :],
    )
    data_train = data.loc[train_idx].reset_index()
    data_val = data.loc[test_idx].reset_index()

    return data_train, data_val
