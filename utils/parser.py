import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    """
        Usually Use
    """
    parser.add_argument('--drugnet_lr_scale', type=float, default=0.5)
    parser.add_argument('--curriculum_weight', type=str, default=None)
    parser.add_argument('--neg_rate', type=int, default=3)
    parser.add_argument('--step_size', type=int, default=65536)
    parser.add_argument('--atten_type', type=str, default="wsam")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--add_transformer', type=bool, default=False)
    parser.add_argument('--focal_loss', type=bool, default=False)
    """
        Usually Freeze
    """
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--init_lr', type=float, default=3e-4)
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--graph_layer', type=str, default='GCN', choices=['GCN', 'GAT'])
    # "string describing convolutional feature extraction layers in form of a python list that contains [(dim, kernel_size, stride), ...]"
    parser.add_argument('--drug_conv', type=str, default="[(512, 1, 1)] * 3")
    parser.add_argument('--target_conv', type=str, default="[(512, 5, 2)] + [(512, 2, 2)] * 2")
    parser.add_argument('--conv_dropout', type=float, default=0.1)
    parser.add_argument('--freeze_protein_embedding', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='test')
    parser.add_argument('--code_dir', type=str, default='')
    parser.add_argument('--pretrain_dir', type=str, default='../3e_pretrain/model_weight.bin')
    parser.add_argument('--target_h5_dir', type=str, default='../3e_pretrain/targetfeature.h5')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=19260817)
    parser.add_argument('--final_lr', type=float, default=1e-7)
    parser.add_argument('--LR_SCHEDULER_FACTOR', type=float, default=0.5)
    parser.add_argument('--LR_SCHEDULER_WAIT', type=float, default=20)
    parser.add_argument('--LR_SCHEDULER_THRESH', type=float, default=0.001)
    parser.add_argument('--MOMENTUM1', type=float, default=0.9)
    parser.add_argument('--MOMENTUM2', type=float, default=0.999)
    parser.add_argument('--num_steps', type=int, default=400)
    parser.add_argument('--save_frequency', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--target_in_size', type=int, default=121)
    parser.add_argument('--GAT_head', type=int, default=2)
    parser.add_argument('--graph_depth', type=int, default=2)
    parser.add_argument('--mlp_depth', type=int, default=2)
    parser.add_argument('--no_edge_weight', default=False, action='store_true')
    parser.add_argument('--no_hcount', default=False, action='store_true')
    args = parser.parse_args()
    print('============Args==============')
    print(args)
    return args
