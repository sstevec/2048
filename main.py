from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Trainer import get_bc_args, SequenceDataset
from Trainer.MyTrainer import Trainer

if __name__ == '__main__':
    # start training
    args = get_bc_args()

    train_set = SequenceDataset(directory=args.data_dir, num_chunks=args.num_chunks)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)

    writer = SummaryWriter()

    trainer = Trainer(args)

    # load model
    # last_epoch_num = trainer.load_latest_checkpoint()
    last_epoch_num = 0
    # trainer.load_checkpoint("./checkpoint/94.pth")
    trainer.train_model(train_loader, writer, last_epoch_num)

    writer.close()