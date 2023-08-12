import torch

class LinearProbe(torch.nn.Module):
    """Linear classifier to convert input_emb to the shape of output_emb."""
    
    def __init__(self, input_emb, output_dim, device):
        """Constructor.
        
        Args:
            input_emb: list of tensor for input embeddings. Assume the input_emb is of shape [C, H, W]
            output_dim: int for the output dimension.
        """
        super().__init__()
        self.linear_layers = []
        self.final_layer = None
        self.device = device

        # Create linear layer for each input embedding.
        for i, emb in enumerate(input_emb):
            dims = emb.shape
            input_dim = dims[-1] * dims[-2]
            self.linear_layers.append(
                torch.nn.Linear(input_dim, output_dim).to(device)
                )

        # Create final layer to combine all linear layers.
        if len(input_emb) > 1:
            self.final_layer = torch.nn.Linear(len(input_emb) * output_dim, output_dim).to(device)

    def forward(self, input_emb):
        """Run linear layer to convert input_emb to dimension of output_emb.

        For input_emb that has more than one channel, apply linear layer for each
        channel and combine with final_layer.
        
        Args:
            input_emb: list of tensor for input embeddings.
        """
        # Sanity check
        if len(input_emb) != len(self.linear_layers):
            raise ValueError("Number of input embedding doesn't match number of linear layers.")
        
        # Apply linear layer on each input embeddings.
        output_emb = []
        for i, layer in enumerate(self.linear_layers):
            emb = input_emb[i]
            emb = emb.squeeze()
            emb = torch.mean(emb, dim=(0))
            emb = emb.reshape(1, -1)
            out = layer(emb)
            output_emb.append(out)
        
        # Apply final layer to concatenate all output embeddings.
        if self.final_layer:
            output_emb = torch.cat(output_emb, dim=1)
            output_emb = [self.final_layer(output_emb)]
        return output_emb[0]

def linear_probe_tests():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create dummy embeddings.
    grounding_dino_emb = [
        torch.randn(1, 256, 80, 80).to(device),
        torch.randn(1, 512, 40, 40).to(device),
        torch.randn(1, 1024, 20, 20).to(device),
    ]
    sam_emb = [torch.randn(1, 256, 64, 64).to(device)]
    biomed_clip_emb = torch.randn(1, 512).to(device)

    # Create linear layers.
    grounding_dino_linear = LinearProbe(
        grounding_dino_emb, 
        biomed_clip_emb.shape[1],
        device,
        )
    sam_linear = LinearProbe(
        sam_emb, 
        biomed_clip_emb.shape[1],
        device,
        )

    # Run linear layers.
    grounding_dino_emb_aligned = grounding_dino_linear(grounding_dino_emb)
    sam_emb_aligned = sam_linear(sam_emb)
    print(grounding_dino_emb_aligned.shape)
    print(sam_emb_aligned.shape)


if __name__ == "__main__":
    linear_probe_tests()