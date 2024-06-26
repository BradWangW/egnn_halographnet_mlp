

# Choose the GNN architecture between "DeepSet", "GCN", "EdgeNet", "PointNet", "MetaNet"
#use_model = "DeepSet"
# use_model = "GCN"
# use_model = "EGNN"
use_model = "EGCL"
#use_model = "PointNet"
#use_model = "EdgePoint"
#use_model = "MetaNet"

# Learning rate
learning_rate = 8.374990124370304e-05 * 5
# Weight decay
weight_decay = 2.8984552987828555e-07
# Number of layers of each graph layer
n_layers = 2
# Number of nearest neighbors in kNN / radius of NNs
k_nn = 9.395159417206347

# Number of epochs
n_epochs = 50
# If training, set to True, otherwise loads a pretrained model and tests it
training = True
# Simulation suite, choose between "IllustrisTNG" and "SIMBA"
#simsuite = "SIMBA"
simsuite = "IllustrisTNG"
# Simulation set, choose between "CV" and "LH"
simset = "LH"
# Number of simulations considered, maximum 27 for CV and 1000 for LH
n_sims = 1000

hidden_channels = 16

params = [use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simsuite, simset, n_sims, hidden_channels]
