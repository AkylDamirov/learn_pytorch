
import torch
import torch.nn as nn
class PatchEmbedding(nn.Module):
    # 2. Initialize the class with appropriate variables
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768):
        super().__init__()

        self.patch_size = patch_size

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,  # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # 6. Make sure the output shape has the right order
        return x_flattened.permute(0, 2,
                                   1)  # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]


class ViT(nn.Module):
    def __init__(self,
                 img_size=224, #from table 3
                 num_channels=3,
                 patch_size=16,
                 embedding_dim=768, #from table 1
                 dropout=0.1,
                 mlp_size=3072, #from table 1
                 num_transformer_layers=12, #from table 1
                 num_heads=12, #from table 1
                 num_classes=1000): #generic number of classes
        super().__init__()

        #assert image size is divisible by patch size
        assert img_size % patch_size == 0, 'Image must be divisible by patch size'

        #create patch embedding
        self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        #create class token
        self.class_token = nn.Parameter(torch.rand(1, 1, embedding_dim),
                                        requires_grad=True)

        #create positional embedding
        num_patches = (img_size * img_size) // patch_size**2  # N = HW/P^2
        self.positional_embedding = nn.Parameter(torch.rand(1, num_patches+1,embedding_dim))

        #create patch + position embedding dropout
        self.embedding_dropout = nn.Dropout(p=dropout)

        #create stack transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                                  nhead=num_heads,
                                                                                                  dim_feedforward=mlp_size,
                                                                                                  activation='gelu',
                                                                                                  batch_first=True,
                                                                                                  norm_first=True), # Create a single Transformer Encoder Layer
                                                         num_layers=num_transformer_layers) #stack it N times

        #create MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        #get some dimensions from x
        batch_size = x.shape[0]

        #create the patch embedding
        x = self.patch_embedding(x)

        #first, expand the class token across the batch size
        class_token = self.class_token.expand(batch_size, -1, -1)  # "-1" means infer the dimension

        #prepand the class token to the patch embedding
        x = torch.cat((class_token,x), dim=1)

        #add the positional embedding to patch embedding with class token
        x = self.positional_embedding + x

        #dropout on patch + positional embedding
        x = self.embedding_dropout(x)

        #pass embedding through transformer encoder stack
        x = self.transformer_encoder(x)

        #pass 0th index of x through MLP head
        x = self.mlp_head(x[:, 0])

        return x
