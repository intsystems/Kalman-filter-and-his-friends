import pytest
import torch
from kalman.gaussian import GaussianState

@pytest.fixture
def simple_gaussian():
    """Create a simple 2D Gaussian for testing"""
    mean = torch.tensor([[1.0], [2.0]])
    covariance = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    return GaussianState(mean, covariance)

@pytest.fixture
def batch_gaussian():
    """Create a batch of 2D Gaussians for testing"""
    mean = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]) 
    covariance = torch.tensor([[[2.0, 0.5], [0.5, 1.0]], [[1.0, 0.3], [0.3, 2.0]]])
    return GaussianState(mean, covariance)

def test_proper_initialization():
    mean = torch.tensor([[1.0], [2.0]])
    covariance = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    state = GaussianState(mean, covariance)
    
    assert torch.allclose(state.mean, mean)
    assert torch.allclose(state.covariance, covariance)
    assert state.precision is None

def test_cloning(simple_gaussian):
    cloned = simple_gaussian.clone()
    
    assert torch.allclose(cloned.mean, simple_gaussian.mean)
    assert torch.allclose(cloned.covariance, simple_gaussian.covariance)
    assert id(cloned.mean) != id(simple_gaussian.mean)
    assert id(cloned.covariance) != id(simple_gaussian.covariance)

def test_indexing(batch_gaussian):
    first_gaussian = batch_gaussian[0]
    
    assert first_gaussian.mean.shape == (2, 1)
    assert first_gaussian.covariance.shape == (2, 2)
    assert torch.allclose(first_gaussian.mean, torch.tensor([[1.0], [2.0]]))

def test_mahalanobis_distance(simple_gaussian):
    measure = torch.tensor([[0.0], [0.0]])
    distance = simple_gaussian.mahalanobis(measure)
    
    assert distance > 0
    assert simple_gaussian.precision is not None

def test_log_likelihood(simple_gaussian):
    measure = torch.tensor([[1.0], [2.0]])
    log_likelihood = simple_gaussian.log_likelihood(measure)
    
    other_measure = torch.tensor([[0.0], [0.0]])
    other_log_likelihood = simple_gaussian.log_likelihood(other_measure)
    assert log_likelihood > other_log_likelihood

def test_likelihood(simple_gaussian):
    measure = torch.tensor([[1.0], [2.0]])
    likelihood = simple_gaussian.likelihood(measure)
    
    assert likelihood > 0
    other_measure = torch.tensor([[0.0], [0.0]])
    other_likelihood = simple_gaussian.likelihood(other_measure)
    assert likelihood > other_likelihood

def test_batch_operations(batch_gaussian):
    measures = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    likelihoods = batch_gaussian.likelihood(measures)
    
    assert likelihoods.shape == (2,)
    assert torch.all(likelihoods > 0) 