import torch
import numpy as np
import time
# from utils.misc import saveHDF5
from tqdm import tqdm

def learn(dkf, dataset, mask, epoch_start=0, epoch_end=1000, 
          batch_size=200, shuffle=False,
          savefreq=None, savefile=None, 
          dataset_eval=None, mask_eval=None, 
          replicate_K=None,
          normalization='frame'):
    """
    Train DKF model using PyTorch
    """
    assert not dkf.params['validate_only'], 'cannot learn in validate only mode'
    assert len(dataset.shape) == 3, 'Expecting 3D tensor for data'
    assert dataset.shape[2] == dkf.params['dim_observations'], 'Dim observations not valid'
    
    # Convert numpy arrays to PyTorch tensors
    dataset = torch.from_numpy(dataset).float()
    mask = torch.from_numpy(mask).float()
    if dataset_eval is not None:
        dataset_eval = torch.from_numpy(dataset_eval).float()
        mask_eval = torch.from_numpy(mask_eval).float()
    
    N = dataset.shape[0]
    idxlist = np.arange(N)
    
    # Initialize lists for tracking metrics
    bound_train_list, bound_valid_list, bound_tsbn_list, nll_valid_list = [], [], [], []
    mu_list_train, cov_list_train, mu_list_valid, cov_list_valid = [], [], [], []

    # Training loop
    for epoch in range(epoch_start, epoch_end):
        epoch_start_time = time.time()
        
        # Shuffle data
        if shuffle:
            np.random.shuffle(idxlist)
        
        # Batch processing
        total_loss = 0
        for bnum in range(0, N, batch_size):
            batch_idx = idxlist[bnum:bnum+batch_size]
            
            # Get batch data
            X = dataset[batch_idx]
            M = mask[batch_idx]
            
            # Pad batch if smaller than batch_size
            if X.shape[0] < batch_size:
                pad_size = batch_size - X.shape[0]
                X = torch.cat([X, torch.zeros(pad_size, *X.shape[1:])], dim=0)
                M = torch.cat([M, torch.zeros(pad_size, *M.shape[1:])], dim=0)
            
            # Reduce sequence length based on mask
            maxT = int(M.sum(1).max())
            X = X[:, :maxT]
            M = M[:, :maxT]
            
            # Generate noise
            eps = torch.randn(X.shape[0], maxT, dkf.params['dim_stochastic'])
            
            # Forward pass and loss calculation
            loss, negCLL, KL = dkf.loss(X, M, eps)
            
            # Backward pass and optimization
            dkf.optimizer.zero_grad()
            loss.backward()
            dkf.optimizer.step()
            
            # Update counters and metrics
            M_sum = M.sum()
            if replicate_K is not None:
                loss, negCLL, KL = loss/replicate_K, negCLL/replicate_K, KL/replicate_K
                M_sum = M_sum/replicate_K
            
            total_loss += loss.item()
            
            # Logging
            if bnum % 10 == 0:
                if normalization == 'frame':
                    bval = loss.item() / float(M_sum)
                elif normalization == 'sequence':
                    bval = loss.item() / float(X.shape[0])
                else:
                    raise ValueError('Invalid normalization')
                
                print(f'Batch: {bnum}, Loss: {bval:.4f}, NLL: {negCLL:.4f}, KL: {KL:.4f}')
        
        # Calculate epoch metrics
        if normalization == 'frame':
            epoch_loss = total_loss / float(mask.sum())
        elif normalization == 'sequence':
            epoch_loss = total_loss / float(N)
        else:
            raise ValueError('Invalid normalization')
        
        bound_train_list.append((epoch, epoch_loss))
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s')
        
        # Save intermediate results
        if savefreq is not None and epoch % savefreq == 0 and savefile is not None:
            print(f'Saving model at epoch {epoch}')
            torch.save(dkf.state_dict(), f'{savefile}-EP{epoch}.pt')
            
            # Evaluation on validation set
            intermediate = {}
            if dataset_eval is not None and mask_eval is not None:
                with torch.no_grad():
                    # Evaluate on validation set
                    val_loss = evaluate_bound(dkf, dataset_eval, mask_eval, batch_size, normalization)
                    bound_valid_list.append((epoch, val_loss))
                    
                    # Calculate NLL
                    val_nll = importance_sampling_nll(dkf, dataset_eval, mask_eval, batch_size, normalization)
                    nll_valid_list.append(val_nll)
                
                intermediate['valid_bound'] = np.array(bound_valid_list)
                intermediate['train_bound'] = np.array(bound_train_list)
                intermediate['valid_nll'] = np.array(nll_valid_list)
                
                # For synthetic data
                if 'synthetic' in dkf.params['dataset']:
                    mu_train, cov_train, mu_valid, cov_valid = synthetic_proc(dkf, dataset, dataset_eval)
                    mu_list_train.append(mu_train)
                    cov_list_train.append(cov_train)
                    mu_list_valid.append(mu_valid)
                    cov_list_valid.append(cov_valid)
                    
                    intermediate['mu_posterior_train'] = np.concatenate(mu_list_train, axis=2)
                    intermediate['cov_posterior_train'] = np.concatenate(cov_list_train, axis=2)
                    intermediate['mu_posterior_valid'] = np.concatenate(mu_list_valid, axis=2)
                    intermediate['cov_posterior_valid'] = np.concatenate(cov_list_valid, axis=2)
                
                # saveHDF5(f'{savefile}-EP{epoch}-stats.h5', intermediate)
    
    # Final results
    retMap = {
        'train_bound': np.array(bound_train_list),
        'valid_bound': np.array(bound_valid_list),
        'valid_nll': np.array(nll_valid_list)
    }
    
    if 'synthetic' in dkf.params['dataset']:
        retMap.update({
            'mu_posterior_train': np.concatenate(mu_list_train, axis=2),
            'cov_posterior_train': np.concatenate(cov_list_train, axis=2),
            'mu_posterior_valid': np.concatenate(mu_list_valid, axis=2),
            'cov_posterior_valid': np.concatenate(cov_list_valid, axis=2)
        })
    
    return retMap

def evaluate_bound(dkf, dataset, mask, batch_size, normalization):
    """Evaluate model on given dataset"""
    total_loss = 0
    N = dataset.shape[0]
    
    with torch.no_grad():
        for bnum in range(0, N, batch_size):
            X = dataset[bnum:bnum+batch_size]
            M = mask[bnum:bnum+batch_size]
            
            # Pad batch if needed
            if X.shape[0] < batch_size:
                pad_size = batch_size - X.shape[0]
                X = torch.cat([X, torch.zeros(pad_size, *X.shape[1:])], dim=0)
                M = torch.cat([M, torch.zeros(pad_size, *M.shape[1:])], dim=0)
            
            # Reduce sequence length
            maxT = int(M.sum(1).max())
            X = X[:, :maxT]
            M = M[:, :maxT]
            
            eps = torch.randn(X.shape[0], maxT, dkf.params['dim_stochastic'])
            loss, _, _ = dkf.loss(X, M, eps)
            
            if normalization == 'frame':
                total_loss += loss.item() / M.sum()
            else:
                total_loss += loss.item() / X.shape[0]
    
    return total_loss / (N // batch_size)

def importance_sampling_nll(dkf, dataset, mask, batch_size, normalization, num_samples=10):
    """Estimate NLL using importance sampling"""
    total_nll = 0
    N = dataset.shape[0]
    
    with torch.no_grad():
        for bnum in range(0, N, batch_size):
            X = dataset[bnum:bnum+batch_size]
            M = mask[bnum:bnum+batch_size]
            
            # Pad batch if needed
            if X.shape[0] < batch_size:
                pad_size = batch_size - X.shape[0]
                X = torch.cat([X, torch.zeros(pad_size, *X.shape[1:])], dim=0)
                M = torch.cat([M, torch.zeros(pad_size, *M.shape[1:])], dim=0)
            
            maxT = int(M.sum(1).max())
            X = X[:, :maxT]
            M = M[:, :maxT]
            
            batch_nll = 0
            for _ in range(num_samples):
                eps = torch.randn(X.shape[0], maxT, dkf.params['dim_stochastic'])
                _, z, mu, logcov = dkf.infer(X, M, eps)
                
                # Calculate log-probabilities
                log_p_x_given_z = dkf.log_prob(X, z)
                log_q_z_given_x = Normal(mu, torch.exp(0.5*logcov)).log_prob(z).sum(-1)
                log_p_z = dkf.log_prior(z)
                
                # Importance weighted estimate
                log_weight = log_p_x_given_z + log_p_z - log_q_z_given_x
                batch_nll += -torch.logsumexp(log_weight, dim=0) + np.log(num_samples)
            
            if normalization == 'frame':
                total_nll += batch_nll.sum().item() / M.sum()
            else:
                total_nll += batch_nll.mean().item()
    
    return total_nll / (N // batch_size)

def synthetic_proc(dkf, dataset, dataset_eval, num_samples=100):
    """Collect statistics on synthetic dataset"""
    all_mus, all_logcovs = [], []
    all_mus_eval, all_logcovs_eval = [], []
    
    with torch.no_grad():
        # Training set statistics
        for _ in range(num_samples):
            _, mus, logcovs = dkf.infer(dataset)
            all_mus.append(mus.cpu().numpy())
            all_logcovs.append(logcovs.cpu().numpy())
        
        # Validation set statistics
        for _ in range(num_samples):
            _, mus, logcovs = dkf.infer(dataset_eval)
            all_mus_eval.append(mus.cpu().numpy())
            all_logcovs_eval.append(logcovs.cpu().numpy())
    
    # Calculate means
    mu_train = np.mean(np.stack(all_mus, axis=-1), axis=-1, keepdims=True)
    cov_train = np.mean(np.exp(np.stack(all_logcovs, axis=-1)), axis=-1, keepdims=True)
    
    mu_valid = np.mean(np.stack(all_mus_eval, axis=-1), axis=-1, keepdims=True)
    cov_valid = np.mean(np.exp(np.stack(all_logcovs_eval, axis=-1)), axis=-1, keepdims=True)
    
    return mu_train, cov_train, mu_valid, cov_valid
