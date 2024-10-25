import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Model.BaseTransformer import TransformerLayer
from Model.CustomFNN import CustomFNN
from Model.CustomLSTM import CustomLSTM
from Model.CustomResnet import ResNet, ResidualBlock
from Model.Model import ExpertLearningModel
from Dataset import SequenceDataset
from Config import get_args

from Data.ChunkData import process_csv_in_chunks
from Data.GenerateSeq import save_to_pt_file, process_csv_file

class Trainer:
    def __init__(self, args):
        self.args = args

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.init_components()

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padded labels (-1) during loss calculation
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def init_components(self):
        input_dim = 16  # 16 means observation only, 20 means with observation + action
        embedding_dim = 4  # embed so we don't input integers to the model
        transformer_input_dim = input_dim * embedding_dim
        resnet_output_dim = 128
        lstm_h_dim = 128
        output_dim = 4  # the probability for each action

        embedder = nn.Embedding(input_dim, embedding_dim)

        transformer_dense_layer_dim = [transformer_input_dim, self.args.transformer_dense_layer_dim,
                                       transformer_input_dim]
        transformer = TransformerLayer(transformer_input_dim, self.device, self.args.transformer_qkv_dim,
                                       transformer_dense_layer_dim,
                                       self.args.transformer_num_head)

        resnet = ResNet(ResidualBlock, [2, 2, 2], transformer_input_dim, 64, resnet_output_dim)

        lstm = CustomLSTM(resnet_output_dim, 192, 1, lstm_h_dim, self.device)

        fnn = CustomFNN([lstm_h_dim, 64, 16, output_dim], self.device)

        final_model = ExpertLearningModel(embedder, transformer, resnet, lstm, fnn, self.device, self.args.batch_size)

        return final_model

    def train_model(self, dataloader, writer):
        # move model to GPU
        self.model.to(self.device)

        for epoch in range(self.args.num_epochs):
            self.model.train()  # Set model to training mode

            epoch_loss = 0.0
            num_batches = len(dataloader)

            with tqdm(dataloader, unit="batch") as tepoch:

                tepoch.set_description(f"Epoch {epoch + 1}/{self.args.num_epochs}")

                for inputs, labels in tepoch:
                    # Move inputs and labels to the device
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()

                    outputs = self.model(inputs.to(torch.int32))  # outputs shape: [batch_size, 5, 4]
                    # Shape: [batch_size, 4]

                    # since we are predicting the action for last step, we only need the label for last one
                    labels = labels[:, -1].long()  # Shape: [batch_size * 5]

                    # Compute the loss, ignoring padding (-1 labels)
                    loss = self.criterion(outputs, labels)

                    # Backward pass and optimization
                    loss.backward()
                    self.optimizer.step()

                    # Accumulate the batch loss to compute the epoch loss later
                    epoch_loss += loss.item()

                    # Update tqdm description with the current batch loss
                    tepoch.set_postfix(batch_loss=loss.item())

                # Calculate average loss for the epoch
                avg_epoch_loss = epoch_loss / num_batches
                print(f"Epoch [{epoch + 1}/{self.args.num_epochs}], Loss: {avg_epoch_loss:.4f}")

                # Log the epoch loss to TensorBoard
                writer.add_scalar('Loss/train', avg_epoch_loss, epoch)

            # save the check point every epoch
            torch.save(self.model.state_dict(), f'{self.args.checkpoint_dir}/{epoch}.pth')


if __name__ == '__main__':
    # chunk the dataset into smaller files
    input_file = 'HugeDatasetReach4096.csv'
    process_csv_in_chunks(input_file)

    # process the chunked file into sequence data for training
    for i in range(1, 50):
        input_file = f'episodes_chunk_{i}.csv'
        sequences, labels = process_csv_file(input_file)
        save_to_pt_file(sequences, labels, f'chunk_{i}.pt')

    # start training
    args = get_args()

    train_set = SequenceDataset(directory="./Data", num_chunks=5)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    writer = SummaryWriter()

    trainer = Trainer(args)
    trainer.train_model(train_loader, writer)

    writer.close()
