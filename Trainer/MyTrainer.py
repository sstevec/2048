import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from Model.BaseTransformer import TransformerLayer
from Model.CustomFNN import CustomFNN
from Model.CustomResnet import ResNet, ResidualBlock
from Model.MyModel import ExpertLearningModel
from Trainer.Dataset import SequenceDataset
from Trainer.Config import get_bc_args
from Model.Util import precompute_2d_positional_encoding
import sys
import os


class Trainer:
    def __init__(self, args):
        self.args = args

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.init_components()

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.alpha = torch.tensor([1.0, 2.0, 10.0, 2.0]).to(
            self.device)  # weight for the action loss, harder penalty on rare moves
        self.gamma = 2.0

    def init_components(self):
        input_dim = 16  # 16 means observation only, 20 means with observation + action
        embedding_dim = 32  # embed so we don't input integers to the model
        resnet_output_dim = 256
        resnet_output_shape = 2
        output_dim = 4  # the probability for each action

        embedder = nn.Embedding(input_dim, embedding_dim)

        resnet = ResNet(ResidualBlock, [2, 2, 2], embedding_dim, 64, resnet_output_dim)

        positional_encoding = precompute_2d_positional_encoding(resnet_output_dim, resnet_output_shape,
                                                                resnet_output_shape).to(self.device)

        transformer1_dense_layer_dim = [resnet_output_dim, self.args.transformer_dense_layer_dim,
                                        resnet_output_dim]
        transformer1 = TransformerLayer(resnet_output_dim, self.device, self.args.transformer_qkv_dim,
                                        transformer1_dense_layer_dim,
                                        self.args.transformer_num_head)

        transformer2_dense_layer_dim = [resnet_output_dim, resnet_output_dim * 2, resnet_output_dim * 4,
                                        resnet_output_dim]
        transformer2 = TransformerLayer(resnet_output_dim, self.device, self.args.transformer_qkv_dim,
                                        transformer2_dense_layer_dim,
                                        self.args.transformer_num_head)

        fnn = CustomFNN(
            [resnet_output_dim * resnet_output_shape * resnet_output_shape, resnet_output_dim * 2,
             resnet_output_dim, 64], self.device, drop_rate=0.2)

        actor_head = nn.Linear(64, output_dim)
        critic_head = nn.Linear(64, 1)

        final_model = ExpertLearningModel(embedder, resnet, positional_encoding, transformer1, transformer2, fnn,
                                          actor_head, critic_head)

        return final_model

    def focal_loss(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none', weight=self.alpha, ignore_index=-1)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

    def train_model(self, dataloader, writer, start_epoch=0):
        # move model to GPU
        self.model.to(self.device)

        for epoch in range(start_epoch, self.args.num_epochs):
            self.model.train()  # Set model to training mode

            epoch_loss = 0.0
            total_correct = 0
            num_batches = len(dataloader)

            with tqdm(dataloader, unit="batch") as tepoch:

                tepoch.set_description(f"Epoch {epoch + 1}/{self.args.num_epochs}")

                for inputs, labels in tepoch:
                    # Move inputs and labels to the device
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()

                    outputs, _ = self.model(inputs[:, -1].to(torch.int32))  # outputs shape: [batch_size, 4]
                    # Shape: [batch_size, 4]

                    # since we are predicting the action for last step, we only need the label for last one
                    labels = labels[:, -1].long()  # Shape: [batch_size, 1]

                    # Compute the loss, ignoring padding (-1 labels)
                    loss = self.focal_loss(outputs, labels)

                    # Backward pass and optimization
                    loss.backward()
                    self.optimizer.step()

                    # Accumulate the batch loss to compute the epoch loss later
                    epoch_loss += loss.item()

                    # Update tqdm description with the current batch loss
                    tepoch.set_postfix(batch_loss=loss.item())

                    pred = outputs.argmax(dim=1)
                    total_correct += pred.eq(labels.view_as(pred)).sum().item()

                # Calculate average loss for the epoch
                avg_epoch_loss = epoch_loss / num_batches
                print(f"Epoch [{epoch + 1}/{self.args.num_epochs}], Loss: {avg_epoch_loss:.4f}")

                # Log the epoch loss to TensorBoard
                writer.add_scalar('Loss/train', avg_epoch_loss, epoch)

                total_acc = total_correct / num_batches / self.args.batch_size
                print(f"Epoch [{epoch + 1}/{self.args.num_epochs}], Acc: {total_acc:.4f}")

            # save the check point every epoch
            torch.save(self.model.state_dict(), f'{self.args.checkpoint_dir}/{epoch}.pth')

    def load_latest_checkpoint(self):
        # Get a list of all checkpoint files in the folder
        checkpoint_files = [f for f in os.listdir(self.args.checkpoint_dir) if f.endswith('.pth')]

        if not checkpoint_files:
            print("No checkpoints found in the folder.")
            return 0

        # Extract epoch numbers and find the largest one
        epoch_numbers = [int(f.split('.')[0]) for f in checkpoint_files]
        latest_epoch = max(epoch_numbers)

        # Construct the filename of the latest checkpoint
        latest_checkpoint = f"{latest_epoch}.pth"
        latest_checkpoint_path = os.path.join(self.args.checkpoint_dir, latest_checkpoint)

        # Load the checkpoint
        checkpoint = torch.load(latest_checkpoint_path)

        # Report the latest epoch number
        self.model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from epoch {latest_epoch}.")
        return latest_epoch

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {path}")


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
