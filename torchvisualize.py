import torch
from torchviz import make_dot
from neural import NeuralNetwork

# Define your actual input size
input_size = 10  # Replace with actual input size from training

# Instantiate model
model = NeuralNetwork(input_size=input_size)

# Load trained weights
model.load_state_dict(torch.load('/deac/csc/vanbastelaerGrp/guptd23/csc790_AI/Assignement1/Alzhemier_Code/nn_output/alzheimer_neural_model.pth'))
model.eval()

# Dummy input tensor
dummy_input = torch.randn(1, input_size)

# Forward pass
output = model(dummy_input)

# Visualize layers only (no gradients or parameters)
dot = make_dot(output,show_attrs=True)

# Save to file
dot.render("model_layers_visualization", format="png")
