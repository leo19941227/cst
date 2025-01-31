import torch

def kl_divergence_diagonal_gaussians(mu1, logvar1, mu2, logvar2):
    """
    Computes KL(p || q) for two diagonal Gaussians:
      p = N(mu1, exp(logvar1))
      q = N(mu2, exp(logvar2))

    Args:
        mu1, logvar1: Tensors of shape [batch_size, ..., D]
        mu2, logvar2: Tensors of shape [batch_size, ..., D]
          (the "..." can be extra dims if needed, but
           typically it's just [batch_size, D])

    Returns:
        kl: Tensor of shape [batch_size, ...] containing
            the KL for each sample in the batch.
    """

    # Convert log-variance to variance
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)

    # KL divergence
    # = 0.5 * sum( (var1 / var2)
    #              + ((mu2 - mu1)^2 / var2)
    #              - 1
    #              + log(var2) - log(var1) )
    kl = 0.5 * torch.sum(
        (var1 / var2)
        + ((mu2 - mu1)**2 / var2)
        - 1.0
        + (logvar2 - logvar1),
        dim=-1   # sum over the last dimension D
    )
    return kl

# ---- EXAMPLE USAGE ----
if __name__ == "__main__":
    # Suppose we have a batch of 4 samples, each with dimension D=3
    batch_size, dim = 10000, 1

    # "Posterior" parameters (e.g., from an encoder)
    mu1      = torch.randn(batch_size, dim) * 1
    logvar1  = torch.randn(batch_size, dim) * 0

    # "Prior" parameters or another Gaussian distribution
    mu2      = torch.zeros(batch_size, dim)
    logvar2  = torch.zeros(batch_size, dim)

    kl_vals = kl_divergence_diagonal_gaussians(mu1, logvar1, mu2, logvar2)
    print(mu1.abs().topk(10, dim=0).values)
    # print(logvar1.exp())
    # print("KL shape:", kl_vals.shape)      # [batch_size]
    print("KL values:", kl_vals.topk(10, dim=0).values)
    print(kl_vals.mean())
