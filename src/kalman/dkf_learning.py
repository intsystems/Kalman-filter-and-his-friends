import torch
import numpy as np
import time


def learn(
    dkf,
    dataset,
    mask,
    epoch_start=0,
    epoch_end=1000,
    batch_size=200,
    shuffle=False,
    savefreq=None,
    savefile=None,
    dataset_eval=None,
    mask_eval=None,
    normalization="frame",
):
    """
    Train DKF model using PyTorch.

    Parameters
    ----------
    dkf : DeepKalmanFilter
        The model to train. Must have an `optimizer` attribute.
    dataset : np.ndarray or torch.Tensor
        Training data, shape (N, T, obs_dim).
    mask : np.ndarray or torch.Tensor
        Binary mask, shape (N, T).
    epoch_start, epoch_end : int
        Training epoch range.
    batch_size : int
        Mini-batch size.
    shuffle : bool
        Whether to shuffle data each epoch.
    savefreq : int, optional
        Save checkpoint every `savefreq` epochs.
    savefile : str, optional
        Path prefix for checkpoint files.
    dataset_eval, mask_eval : array-like, optional
        Validation data and mask.
    normalization : str
        'frame' or 'sequence'.

    Returns
    -------
    dict with keys 'train_bound', 'valid_bound'.
    """
    assert len(dataset.shape) == 3, "Expecting 3D tensor for data (N, T, obs_dim)"

    if isinstance(dataset, np.ndarray):
        dataset = torch.from_numpy(dataset).float()
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).float()
    if dataset_eval is not None and isinstance(dataset_eval, np.ndarray):
        dataset_eval = torch.from_numpy(dataset_eval).float()
        mask_eval = torch.from_numpy(mask_eval).float()

    N = dataset.shape[0]
    idxlist = np.arange(N)

    bound_train_list = []
    bound_valid_list = []

    for epoch in range(epoch_start, epoch_end):
        epoch_start_time = time.time()

        if shuffle:
            np.random.shuffle(idxlist)

        total_loss = 0.0
        total_mask = 0.0

        for bnum in range(0, N, batch_size):
            batch_idx = idxlist[bnum : bnum + batch_size]

            # (B, T, obs_dim) -> (T, B, obs_dim)
            X = dataset[batch_idx].permute(1, 0, 2)
            M = mask[batch_idx].permute(1, 0)

            # Trim to max valid length
            maxT = int(M.sum(0).max().item())
            X = X[:maxT]
            M = M[:maxT]

            dkf.optimizer.zero_grad()
            loss, nll, kl = dkf.loss(X, M)
            loss.backward()
            dkf.optimizer.step()

            total_loss += loss.item()
            total_mask += M.sum().item()

        if normalization == "frame":
            epoch_loss = total_loss / max(total_mask, 1.0)
        else:
            epoch_loss = total_loss / float(N)

        bound_train_list.append((epoch, epoch_loss))
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")

        if savefreq is not None and epoch % savefreq == 0 and savefile is not None:
            torch.save(dkf.state_dict(), f"{savefile}-EP{epoch}.pt")

            if dataset_eval is not None and mask_eval is not None:
                with torch.no_grad():
                    val_loss = evaluate_bound(
                        dkf, dataset_eval, mask_eval, batch_size, normalization
                    )
                    bound_valid_list.append((epoch, val_loss))

    return {
        "train_bound": np.array(bound_train_list),
        "valid_bound": np.array(bound_valid_list),
    }


def evaluate_bound(dkf, dataset, mask, batch_size, normalization):
    """Evaluate ELBO on the given dataset."""
    total_loss = 0.0
    total_mask = 0.0
    N = dataset.shape[0]
    n_batches = 0

    with torch.no_grad():
        for bnum in range(0, N, batch_size):
            X = dataset[bnum : bnum + batch_size].permute(1, 0, 2)
            M = mask[bnum : bnum + batch_size].permute(1, 0)

            maxT = int(M.sum(0).max().item())
            X = X[:maxT]
            M = M[:maxT]

            loss, _, _ = dkf.loss(X, M)
            total_loss += loss.item()
            total_mask += M.sum().item()
            n_batches += 1

    if normalization == "frame":
        return total_loss / max(total_mask, 1.0)
    else:
        return total_loss / float(N)
