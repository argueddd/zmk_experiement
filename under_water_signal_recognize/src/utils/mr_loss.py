import torch


def normalized_covariance_matrix_torch(D: torch.Tensor) -> torch.Tensor:
    C = D @ D.T
    norms = torch.norm(D, p=2, dim=1)
    norm_matrix = torch.outer(norms, norms)
    norm_matrix = torch.where(norm_matrix == 0, torch.tensor(1e-8, device=D.device), norm_matrix)
    return C / norm_matrix


def compute_MR_torch(A: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    J = A.shape[0]
    A_alpha = torch.matrix_power(A, int(alpha))
    tr_A = torch.clamp(torch.trace(A), min=1e-8)
    tr_A_alpha = torch.clamp(torch.trace(A_alpha), min=1e-8)
    numerator = torch.log(tr_A_alpha) - alpha * torch.log(tr_A)
    denominator = torch.log(torch.tensor(J ** (alpha - 1), dtype=torch.float32, device=A.device))
    return 1 + numerator / denominator


def batch_mr_loss(batch_X: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """
    输入 shape: (B, 1, 199, 310)
    转换后: 每个样本用 [310, 199] 的矩阵计算 M_R
    """
    mr_list = []
    for i in range(batch_X.shape[0]):
        x = batch_X[i].squeeze(0)        # shape: [199, 310]
        x = x.transpose(0, 1)            # shape: [310, 199]
        norm_cov = normalized_covariance_matrix_torch(x)
        mr = compute_MR_torch(norm_cov, alpha=2)
        mr_list.append(mr)
    mr_tensor = -torch.stack(mr_list)
    return mr_tensor.mean()
    # return torch.exp(mr_tensor/(-sigma)).mean()
    # return mr_tensor.mean()

