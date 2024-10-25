import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')

    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoint", help='number of epochs')

    # transformer config
    parser.add_argument('--transformer_dense_layer_dim', type=int, default=128, help='dropout rate')
    parser.add_argument('--transformer_qkv_dim', type=int, default=128, help='dropout rate')
    parser.add_argument('--transformer_num_head', type=int, default=4, help='dropout rate')

    return parser.parse_args()