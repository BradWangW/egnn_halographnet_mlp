#----------------------------------------------------
# Main routine for training and testing GNN models
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
#----------------------------------------------------

import time, datetime, psutil, os, glob, torch
from Source.networks import *
from Source.training import *
from Source.plotting import *
from Source.load_data import *

# Main routine to train the neural net
# If testsuite==True, it takes a model already pretrained in the other suite and tests it in the selected one
def main(params, verbose = True, testsuite = False, de = False):

    # Load hyperparameters
    use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simsuite, simset, n_sims, hidden_channels = params
    if de:
        params[8] = 'CV'
        simset = 'CV'
    
    # Load data and create dataset
    dataset, node_features = create_dataset(simsuite, simset, n_sims)
    train_loader, valid_loader, test_loader = split_datasets(dataset)

    # Initialize model
    model = ModelGNN(use_model, node_features, n_layers, k_nn, hidden_channels=hidden_channels)
    model.to(device)
    
    # Number of trainable parameters
    n_params = str(sum(p.numel() for p in model.parameters() if p.requires_grad))

    if verbose: print("Model: "+namemodel(params)+"\n"+n_params+' trainable parameters'+'\n')

    # Train the net
    if training:
        if verbose: print("Training!\n")
        train_losses, valid_losses = training_routine(model, train_loader, valid_loader, params, verbose)

    # Test the net
    if verbose: print("\nTesting!\n")

    # Load the trained model
    state_dict = torch.load("Models/"+namemodel(params), map_location=device)
    model.load_state_dict(state_dict)
    
    if de:
        params[8] = 'LH'
        simset = 'LH'
        n_sims = 1000
        print('Testing on LH')

    if testsuite: 
        params[7]=changesuite(simsuite)   # change for loading the model
        simsuite = changesuite(simsuite)
        print('Testing on', simsuite)
        
    dataset, node_features = create_dataset(simsuite, simset, n_sims)
    _, _, test_loader = split_datasets(dataset)

    # Test the model
    test_loss, rel_err, suffix = test(test_loader, model, params, de=de)
    if verbose: print("Test Loss: {:.2e}, Relative error: {:.2e}".format(test_loss, rel_err))

    # Plot loss trends
    if training:
        plot_losses(train_losses, valid_losses, test_loss, rel_err, params)

    # Plot true vs predicted halo masses
    plot_out_true_scatter(params, testsuite, de=de, suffix=suffix)
    
    suffix = '_params_' + n_params + '_de' * de + '.png'

    return test_loss, suffix


#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    for path in ["Plots", "Models", "Outputs"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load default parameters
    from Hyperparameters.params_SIMBA_CV import params
    
    print(params)
    
    _, suffix = main(params, testsuite=True, de=True)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))

    # Rename the latest two file
    list_of_files = glob.glob('Plots/*')
    list_of_files.sort(key=os.path.getctime, reverse=True)
    
    for latest_file in list_of_files[:1]:
        os.rename(
            latest_file, 
            latest_file[:-4] + suffix
        )