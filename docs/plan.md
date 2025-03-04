# План архитектуры Фильтра Калмана

Будет некоторый базовый класс Фильтров, от которого будут наследоваться другие фильтры.

```
class BaseFilter(nn.Module):
    """
    Abstract base class for Kalman Filters
    """

    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

    def predict(
        self, 
        state_mean: torch.Tensor, 
        state_cov: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step predict.
        Returns:
            predicted_state_mean, predicted_state_cov
        """
        pass

    def update(
        self, 
        state_mean: torch.Tensor, 
        state_cov: torch.Tensor, 
        measurement: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step update.
        Returns:
            updated_state_mean, updated_state_cov
        """
        pass
    
    def predict_update(
        self, 
        state_mean: torch.Tensor,
        state_cov: torch.Tensor,
        measurement: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step predict and update in one function.
        Returns: 
            updated_state_mean, updated_state_cov
        """
        updated_state_mean, updated_state_cov = self.update(state_mean, state_cov, measurement)
        predicted_state_mean, predicted_state_cov = self.predict(updated_state_mean, updated_state_cov)
        # ...
        return predicted_state_mean, predicted_state_cov

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes an entire sequence of observations in shape (T, B, obs_dim).
        Returns all_states, pairs (all_means, all_covs) with shapes:
            all_means: (T, B, state_dim)
            all_covs:  (T, B, state_dim, state_dim)
        """
        pass
```

В нём будет три способа его использовать.
1. Самому руками вызывать функции update, predict по очереди.
2. Вызывать update_predict за один раз, если не будем производить никаких манипуляция над стейтом самостоятельно, например.
3. Использовать forward, когда данные уже есть, а не поступают в онлайн формате поэтапно. Например, какие-то эбмединги нейронной сети, и вот мы думаем, что Фильтр Калмана это то, что нужно для них, но поэтапно писать не хотим.

Можно сделать DataClass, в котором будем хранить состояние, а не выдавать парой, чтобы выглядело покрасивее.
Сами матрицы будут храниться внутри класса и инициализироваться, при создании. Также можно будет сделать методы для доступа к ним и обновления руками каждой из них.