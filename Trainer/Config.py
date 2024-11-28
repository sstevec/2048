import argparse


def get_bc_args():
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


def get_gail_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001, help='policy net learning rate')
    parser.add_argument('--disc_lr', type=float, default=0.0002, help='discriminator network learning rate')
    parser.add_argument('--entropy_weight', type=float, default=0.01, help='larger to encourage explore')

    parser.add_argument('--num_epochs', type=int, default=15000, help='number of full GAIL epochs')

    parser.add_argument('--self_play_steps', type=int, default=4000, help='num of steps of self-play')

    parser.add_argument('--disc_train_epochs', type=int, default=6, help='disc train epoches')
    parser.add_argument('--batch_size', type=int, default=128, help='disc train batch size')

    parser.add_argument('--policy_update_steps', type=int, default=15,
                        help='num of times to update policy network per GAIL epoch')

    parser.add_argument('--num_chunks', type=int, default=20, help='number of expert data chunks to use')
    parser.add_argument('--data_dir', type=str, default="./Data/processed_data", help='path for expert data')

    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoint", help='path for checkpoint')
    parser.add_argument('--save_freq', type=int, default=200, help='how often to save checkpoint')

    # transformer config
    parser.add_argument('--transformer_dense_layer_dim', type=int, default=128)
    parser.add_argument('--transformer_qkv_dim', type=int, default=128)
    parser.add_argument('--transformer_num_head', type=int, default=4)

    return parser.parse_args()
