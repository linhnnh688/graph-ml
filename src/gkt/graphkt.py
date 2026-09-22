from torch_geometric.utils.convert import from_networkx

import pandas as pd
import itertools
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from .utils import *
from .preprocess import *
from .transformer import *
from .graph_net import *


def init_models(data, block_size, skill_embd_dim, graph_net=GAT, baseline=False):
    """
    Pre-process data, generate the skill graph, and initialize the models
    and optimizer.
    """
    # 1. Pre-process data by removing sequences of length 1 and undefined skills.
    # 2. Split data into training and validation set.
    # 3. Generate and load skill graph based on
    data_train, data_val, skill_graph = preprocess(data)
    n_skills = skill_graph.number_of_nodes()

    # Transformer configuration for KT model.
    config = GPTConfig(
        vocab_size=n_skills,
        block_size=block_size,
        n_layer=2,
        n_head=8,
        n_embd=128,
        input_size=2 + n_skills + skill_embd_dim * (1 - baseline),
        bkt=False,
    )
    model = GPT(config).cuda()

    # Convert skill_graph to torch_geometric graph and instantiate graph network.
    skill_graph = from_networkx(skill_graph)
    skill_net = graph_net(n_skills, skill_embd_dim).cuda()
    print(
        "Total Parameters:",
        sum(p.numel() for p in model.parameters())
        + sum(p.numel() for p in skill_net.parameters()),
    )

    # Optimize all parameters end-to-end.
    optimizer = optim.AdamW(
        itertools.chain(model.parameters(), skill_net.parameters()), lr=1e-4
    )

    return data_train, data_val, skill_graph, model, skill_net, optimizer


def init_graphkt(
    data,
    block_size,
    skill_embd_dim,
    train_split=0.8,
    graph_net=GAT,
    baseline=False,
):
    data_train, data_test = train_test_split(data, train_split=train_split)
    # torch.cuda.empty_cache()

    # Initialize and train models with GCN, GraphSAGE and GAT.
    data_train, data_val, skill_graph, model, skill_net, optimizer = init_models(
        data_train,
        block_size=block_size,
        skill_embd_dim=skill_embd_dim,
        graph_net=graph_net,
        baseline=baseline,
    )
    print(f"Training {skill_net.tag} model!")
    print(f"Training data: {len(data_train)}, Validation data: {len(data_val)}")

    return data_train, data_val, skill_graph, model, skill_net, optimizer


def train(
    model,
    optimizer,
    skill_net,
    skill_graph,
    data_train,
    data_val,
    batch_size,
    block_size,
    num_epochs,
    baseline=False,
):
    """
    Train the KT transformer and GNN end-to-end by optimizing the KT binary
    cross-entropy objective.Arguments:
      - model (transformer for KT)
      - skill_net (GNN for skill embeddings)
      - data_train (training data)
      - data_val (validation data)
      - num_epochs (number of training epochs)
    """
    for epoch in range(num_epochs):
        # Train model for num_epochs epochs.
        model.train()
        skill_net.train()
        batches_train = construct_batches(
            data_train, batch_size=batch_size, block_size=block_size, epoch=epoch
        )
        pbar = tqdm(batches_train)
        losses = []
        for X, y in pbar:
            optimizer.zero_grad()

            # Get node embeddings for all skills from skill_net (GNN).
            all_skill_embd = skill_net(
                torch.arange(110).cuda(),
                skill_graph.edge_index.cuda(),
                skill_graph.weight.cuda().float(),
            )

            # Select node embeddings corresponding to skill tagged with data.
            skill_embd = all_skill_embd[
                torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()
            ]
            ohe = torch.eye(110).cuda()[
                torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()
            ]

            # Concatenate data with skill embedding and one-hot encoding of skill.
            if baseline:
                feat = [X, ohe]
            else:
                feat = [X, skill_embd, ohe]
            output = model(
                torch.cat(feat, dim=-1), skill_idx=y[..., 0].detach()
            ).ravel()

            # Compute loss and mask padded values.
            mask = (y[..., -1] != -1000).ravel()
            loss = F.binary_cross_entropy(output[mask], y[..., -1:].ravel()[mask])

            # Backpropagate and take a gradient step.
            loss.backward()
            optimizer.step()

            # Report the training loss.
            losses.append(loss.item())
            pbar.set_description(f"Training Loss: {np.mean(losses)}")

        if epoch % 1 == 0:
            # Evaluate model using validation set.
            batches_val = construct_batches(
                data_val,
                batch_size=batch_size,
                block_size=block_size,
                epoch=epoch,
                val=True,
            )
            model.eval()
            skill_net.eval()

            # Construct predictions based on current model and compute error(s).
            ypred, ytrue = evaluate(
                model, skill_graph=skill_graph, skill_net=skill_net, batches=batches_val
            )
            auc = roc_auc_score(ytrue, ypred)
            acc = (ytrue == ypred.round()).mean()
            rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))

            # Report error metrics on validation set and save checkpoint.
            print(
                f"Epoch {epoch}/{num_epochs} - [VALIDATION AUC: {auc}] - [VALIDATION ACC: {acc}] - [VALIDATION RMSE: {rmse}]"
            )

            torch.save(
                model.state_dict(),
                f"ckpts/model-{skill_net.tag}-{epoch}-{auc}-{acc}-{rmse}.pth",
            )

            if not baseline:
                torch.save(
                    skill_net.state_dict(),
                    f"ckpts/skill_net-{skill_net.tag}-{epoch}-{auc}-{acc}-{rmse}.pth",
                )


def run():
    # Read KT dataset using Pandas.
    data = pd.read_csv(
        "https://github.com/CAHLR/pyBKT-examples/blob/master/data/as.csv?raw=true",
        encoding="latin",
    )

    # Global hyperparameters for KT model.
    num_epochs = 15
    batch_size = 8
    block_size = 2048

    # Graph network hyperparameters.
    skill_embd_dim = 128
    train_split = 0.8

    data_train, data_val, skill_graph, model, skill_net, optimizer = init_models(
        data, block_size, skill_embd_dim
    )

    train(
        model,
        optimizer,
        skill_net,
        skill_graph,
        data_train,
        data_val,
        batch_size=batch_size,
        block_size=block_size,
        num_epochs=num_epochs,
    )


if __name__ == "__main__":
    run()
