import torch
from pytorch_forecasting.metrics import QuantileLoss


class HorizonWeightedQuantileLoss(QuantileLoss):
    """
    QuantileLoss(pinball)에 horizon 가중치를 적용.
    - t+1에 더 큰 가중치 부여 가능
    - normalize_to_mean=True면 (평균 가중치=1)이 되도록 자동 정규화(학습 스케일 안정)
    """

    def __init__(
        self,
        quantiles=(0.1, 0.5, 0.9),
        horizon_weights=None,          # e.g.: [3.0, 1.0, 1.0]
        t1_weight=1.0,                 # horizon_weights가 없을 때 t+1만 키우는 용도
        prediction_length=None,        # horizon_weights가 없을 때 길이 만들기
        normalize_to_mean=True,        # True: mean(weights)=1로 스케일 고정
        reduction="mean",
        **kwargs,
    ):
        super().__init__(quantiles=list(quantiles), reduction=reduction, **kwargs)

        if horizon_weights is None:
            if prediction_length is None:
                raise ValueError("prediction_length must be provided when horizon_weights is None.")
            w = [1.0] * int(prediction_length)
            if len(w) > 0:
                w[0] = float(t1_weight)
            horizon_weights = w

        w = torch.as_tensor(horizon_weights, dtype=torch.float32)
        self.register_buffer("horizon_weights", w, persistent=False)
        self.normalize_to_mean = bool(normalize_to_mean)

    def _get_horizon_w(self, T: int, device, dtype):
        w = self.horizon_weights.to(device=device, dtype=dtype)

        # 길이가 부족하면 1.0으로 패딩
        # 길이가 길면 잘라내기
        if w.numel() < T:
            pad = torch.ones(T - w.numel(), device=device, dtype=dtype)
            w = torch.cat([w, pad], dim=0)
        elif w.numel() > T:
            w = w[:T]

        if self.normalize_to_mean:
            # mean(w)=1이 되도록 정규화 (loss 스케일 폭주 방지)
            w = w * (T / w.sum().clamp_min(1e-12))

        return w  # (T,)

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        # base pinball loss: (B, T, Q)  (혹은 (B, T))
        base = super().loss(y_pred, y_actual)

        if base.ndim < 2:
            return base

        T = base.size(1)
        w = self._get_horizon_w(T, device=base.device, dtype=base.dtype)  # (T,)

        # broadcast: (1, T, 1) 또는 (1, T)
        if base.ndim == 2:
            return base * w.view(1, T)
        else:
            return base * w.view(1, T, 1)
