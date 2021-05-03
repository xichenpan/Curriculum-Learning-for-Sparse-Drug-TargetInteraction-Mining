from GraphModels import GraphNeuralNetwork
from utils import *
from Dataset import DrugTargetInteractionDataset
import torch
from torch.utils.data import DataLoader
import tqdm

def main():
    args = parse_args()

    DTIdataset = DrugTargetInteractionDataset(
        edge_weight=not args.no_edge_weight,
        use_hcount=not args.no_hcount
    )
    drug_dataloader = DataLoader(
        DTIdataset.drug_dataset,
        batch_size=args.bz,
        shuffle=True,
        num_workers=8
    )

    graph_model = GraphNeuralNetwork(
        in_dim=DTIdataset.drug_dataset.embedding_dim,
        out_dim=args.d_model,
        layer_type=args.graph_layer,
        num_pre=args.mlp_depth,
        num_graph_layer=args.graph_depth,
        head=args.GAT_head
    )

    graph_model = graph_model.cuda()

    optimizer = torch.optim.Adam(
        graph_model.parameters(),
        lr=args.lr
    )

    for batch in tqdm.tqdm(drug_dataloader):
        node, adj, padding_mask = [data.cuda().float() for data in batch]
        node = graph_model(node, adj)


if __name__ == '__main__':
    main()
