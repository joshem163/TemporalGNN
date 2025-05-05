import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv,global_mean_pool

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, n_layers, num_timesteps,drop_out):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_timesteps, hidden_dim))
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=n_layers, num_decoder_layers=n_layers)
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)  # Flatten the output of the transformer
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, src):
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        src_emb = src_emb.permute(1, 0, 2)  # (seq_len, batch, feature)
        transformer_output = self.transformer.encoder(src_emb)
        transformer_output = transformer_output.permute(1, 0, 2).contiguous().view(src.size(0), -1)  # Flatten
        transformer_output = self.dropout(transformer_output)
        predictions = self.fc(transformer_output)
        return predictions


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)  # FIXED: include batch
        return x


class CNNTransformer(nn.Module):
    def __init__(self, num_classes, cnn_channels=64, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(4, cnn_channels, kernel_size=3, padding=1),  # (B, 64, 20, 10)
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),  # (B, 64, 20, 10)
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU()
        )

        # Flatten spatial grid into sequence of patches
        self.flatten_patches = lambda x: x.flatten(2).transpose(1, 2)  # (B, N=200, D=64)

        # Linear projection to d_model
        self.embedding = nn.Linear(cnn_channels, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, 4, 20, 10)
        x = self.cnn(x)  # (B, 64, 20, 10)
        x = self.flatten_patches(x)  # (B, 200, 64)
        x = self.embedding(x)  # (B, 200, d_model)

        # Add CLS token
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat((cls_token, x), dim=1)  # (B, 201, d_model)

        x = self.transformer(x)  # (B, 201, d_model)
        cls_output = x[:, 0]  # (B, d_model)
        return self.classifier(cls_output)


class T3Former(nn.Module):
    def __init__(self,
                 transformer_input_dim, transformer_hidden_dim, transformer_output_dim,
                 n_heads, n_layers, num_timesteps,
                 cnn_num_classes, dropout_p, cnn_channels=64, cnn_d_model=128, cnn_nhead=4, cnn_num_layers=4,
                 final_output_dim=10):  # Added dropout_p
        super().__init__()

        # Transformer branch
        self.transformer_branch = TransformerClassifier(
            input_dim=transformer_input_dim,
            hidden_dim=transformer_hidden_dim,
            output_dim=transformer_output_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            num_timesteps=num_timesteps,
            drop_out=dropout_p
        )

        # CNN-Transformer branch
        self.cnn_transformer_branch = CNNTransformer(
            num_classes=final_output_dim,
            cnn_channels=cnn_channels,
            d_model=cnn_d_model,
            nhead=cnn_nhead,
            num_layers=cnn_num_layers
        )

        # Final classifier
        combined_feature_dim = transformer_output_dim + final_output_dim
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_final = nn.Linear(combined_feature_dim, cnn_num_classes)

    def forward(self, cnn_input, transformer_input):
        out1 = self.transformer_branch(transformer_input)  # (B, transformer_output_dim)
        out2 = self.cnn_transformer_branch(cnn_input)  # (B, cnn_num_classes)

        combined = torch.cat([out1, out2], dim=1)  # (B, transformer_output_dim + cnn_num_classes)
        combined = self.dropout(combined)
        output = self.fc_final(combined)  # (B, final_output_dim)
        return output


class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, n_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        #return attn_output.mean(dim=1)  # mean over the 3 views
        return attn_output.reshape(attn_output.size(0), -1)


class T3SAGE(nn.Module):
    def __init__(self,
                 sage_input_dim,transformer_input_dim, hidden_dim, output_dim,
                 n_heads, n_layers, num_timesteps1,num_timesteps2, dropout_p,
                 final_output_dim=10):  # Added dropout_p
        super().__init__()

        # Transformer branch
        self.transformer_branch = TransformerClassifier(
            input_dim=transformer_input_dim,
            hidden_dim=hidden_dim,
            output_dim=final_output_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            num_timesteps=num_timesteps1,
            drop_out=dropout_p
        )
        self.transformer_branch2 = TransformerClassifier(
            input_dim=transformer_input_dim,
            hidden_dim=hidden_dim,
            output_dim=final_output_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            num_timesteps=num_timesteps2,
            drop_out=dropout_p
        )
        # CNN-Transformer branch
        self.sage_branch = SAGE(sage_input_dim,hidden_dim,final_output_dim,n_layers,dropout_p
        )

        # Final classifier
        combined_feature_dim = 3*final_output_dim
        self.attn_fusion = AttentionFusion(embed_dim=final_output_dim, n_heads=1)

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_final = nn.Linear(combined_feature_dim, output_dim)

    def forward(self, x_gsage, edge_index, batch, transformer_input, transformer_input1):
        out1 = self.transformer_branch(transformer_input)
        out2 = self.transformer_branch2(transformer_input1)
        out3 = self.sage_branch(x_gsage, edge_index, batch)

        stacked = torch.stack([out1, out2, out3], dim=1)  # (batch_size, 3, final_output_dim)
        attn_out = self.attn_fusion(stacked)  # (batch_size, final_output_dim)
        # print(attn_out)
        output = self.fc_final(attn_out)

        #combined = torch.cat([out1, out2, out3], dim=1)
        #combined = self.dropout(combined)
        #output = self.fc_final(combined)
        return output



