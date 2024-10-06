import torch
from Model.BaseTransformer import TransformerLayer
from Model.CustomFNN import CustomFNN
from Model.CustomLSTM import CustomLSTM
from Model.CustomResnet import ResNet, ResidualBlock
from Model.Model import ExpertLearningModel


def init_components(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer_dense_layer_dim = [20, args.transformer_desnse_layer_dim, 20]
    transformer = TransformerLayer(20, device, args.transformer_qkv_dim, transformer_dense_layer_dim,
                                   args.transformer_num_head)

    resnet = ResNet(ResidualBlock, [2, 2, 2], 20, 64, 128)

    lstm = CustomLSTM(128, 192, 1, 128, device)

    fnn = CustomFNN([128, 64, 16, 4], device)

    final_model = ExpertLearningModel(transformer, resnet, lstm, fnn, device, args.batch_size)

    return final_model

