import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--num_chunks', type=int, default=20, help='number of data chunks to use')

    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoint", help='path for checkpoint')
    parser.add_argument('--data_dir', type=str, default="./Data", help='path for data')

    # transformer config
    parser.add_argument('--transformer_dense_layer_dim', type=int, default=128)
    parser.add_argument('--transformer_qkv_dim', type=int, default=128)
    parser.add_argument('--transformer_num_head', type=int, default=4)

    return parser.parse_args()