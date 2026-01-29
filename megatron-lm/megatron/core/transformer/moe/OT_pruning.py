import numpy as np
import torch

def ot_based_ensemble_pruning(prob, sigma, k: int):
    """
    prob: (E,)  -- single token's expert probability vector (mu_i)
    sigma: (E,E) -- global expert covariance
    k: number of experts to select
    Returns: 0/1 vector (E,)
    """
    p = np.asarray(prob,  dtype=np.float64)     # convert to numpy
    Sx = np.asarray(sigma, dtype=np.float64)
    E = Sx.shape[0]
    eps = 1e-12

    # Select first expert: argmax p_i^2 / Sigma[i,i]
    diag = np.clip(np.diag(Sx), eps, None)
    scores0 = (p * p) / diag
    S = [int(np.argmax(scores0))]

    # Initialize R (diagonal block of inverse Cholesky)
    R = np.array([[1.0 / np.sqrt(diag[S[0]])]], dtype=np.float64)

    for t in range(1, k):
        cand = list(set(range(E)) - set(S))
        best_score = -np.inf
        best_idx = -1
        best_R = None

        # Pre-fetch selected columns (avoid repeated indexing overhead)
        for j in cand:
            beta_j = Sx[S, j]                    # shape: (t,)
            b_jj  = Sx[j, j]

            alpha_j = R @ beta_j                 # (t,)
            s = float(b_jj - alpha_j @ alpha_j)  # Schur complement
            if s <= eps:                         # numerical stability
                s = eps
            r_j = 1.0 / np.sqrt(s)               # scalar

            gamma_j = (-r_j * alpha_j.reshape(1, -1)) @ R   # (1, t)

            # Assemble extended R ((t+1)x(t+1))
            R_j = np.block([
                [R,                      np.zeros((R.shape[0], 1), dtype=np.float64)],
                [gamma_j.astype(np.float64),      np.array([[r_j]], dtype=np.float64)]
            ])

            # Compute score: || R_j @ p_sub ||^2
            idx = S + [j]
            p_sub = p[idx].reshape(-1, 1)        # (t+1, 1)
            sc = float((R_j @ p_sub).T @ (R_j @ p_sub))
            if sc > best_score:
                best_score, best_idx, best_R = sc, j, R_j

        S.append(best_idx)
        R = best_R

    out = np.zeros(E, dtype=int)
    out[S] = 1
    return out


@torch.no_grad()
def otep_batched(scores: torch.Tensor, sigma: torch.Tensor, k: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Batched OTEP selection
    scores: [T, E]  -- expert probability vector for each token
    sigma : [E, E]  -- global expert covariance (symmetric, positive semi-definite, jitter recommended)
    k     : number of experts to select
    Returns: [T, E] bool routing mask (exactly k True per row)
    """
    assert scores.dim() == 2 and sigma.dim() == 2
    T, E = scores.shape
    assert sigma.shape == (E, E)
    device = scores.device
    dtype  = scores.dtype

    # Pre-compute diagonal (numerical stability)
    diag = torch.diag(sigma).clamp_min(eps)                     # [E]
    inv_sqrt_diag = diag.rsqrt()                                # 1/sqrt(diag)

    # Result containers
    selected_mask = torch.zeros(T, E, dtype=torch.bool, device=device)
    selected_idx  = torch.full((T, k), -1, dtype=torch.long, device=device)  # matrix of -1s

    # Inverse Cholesky factor R and z=R p_S packed buffer (only use top-left txt)
    R = torch.zeros(T, k, k, dtype=dtype, device=device)        # one R per token
    z = torch.zeros(T, k, dtype=dtype, device=device)           # one z per token

    # ====== Round 0: select first expert j0 (vectorized) ======
    score0 = (scores ** 2) / diag.unsqueeze(0)                  # [T,E]
    j0 = torch.argmax(score0, dim=1)                            # [T] compare columns, return max per row
    selected_idx[:, 0] = j0
    selected_mask.scatter_(1, j0.unsqueeze(1), True)

    # Initialize R, z, ||z||^2
    R[:, 0, 0] = inv_sqrt_diag.gather(0, j0)                    # R00 = 1/sqrt(Sigma[j0,j0])
    z[:, 0]    = R[:, 0, 0] * scores.gather(1, j0.unsqueeze(1)).squeeze(1)  # z0 = R00 * p_j0
    z_norm_sq  = z[:, 0] ** 2

    # ====== Remaining k-1 rounds: score all (T,E) at once ======
    for t in range(1, k):
        # Get selected indices S (each token has t indices)
        S_t = selected_idx[:, :t]                               # [T,t]

        # Sigma[S, :]: use advanced indexing to get [T,t,E] (t rows per token)
        beta = sigma[S_t, :]                                    # [T,t,E]

        # alpha = R * beta (batched: [T,t,t] @ [T,t,E] -> [T,t,E])
        R_tt = R[:, :t, :t]                                     # [T,t,t]
        alpha = torch.bmm(R_tt, beta)                           # [T,t,E] compute all columns' aj

        # s_j = Sigma[j,j] - ||alpha_j||^2
        a2 = alpha.pow(2).sum(dim=1)                            # [T,E]
        s  = diag.unsqueeze(0) - a2                             # [T,E]
        s  = torch.clamp(s, min=eps)
        r  = s.rsqrt()                                          # [T,E]

        # a^T z (z: [T,k]; alpha: [T,t,E])
        # Expand z to match alpha's t dimension, then inner product over t
        aTz = (alpha * z[:, :t].unsqueeze(2)).sum(dim=1)        # [T,E]

        # Score for each candidate j: ||z||^2 + (r*(p_j - a^T z))^2
        delta = scores - aTz                                    # [T,E]
        cand_score = z_norm_sq.unsqueeze(1) + (r * delta).pow(2)  # [T,E]

        # Mask already selected experts
        cand_score = cand_score.masked_fill(selected_mask, float('-inf'))

        # Select best j* for this round
        j_star = torch.argmax(cand_score, dim=1)                # [T]
        selected_idx[:, t] = j_star
        selected_mask.scatter_(1, j_star.unsqueeze(1), True)

        # Get alpha*, s*, r* for selected j*
        arng = torch.arange(T, device=device)
        alpha_sel = alpha[arng, :, j_star]                      # [T,t]
        s_sel     = s[arng, j_star]                             # [T]
        r_sel     = s_sel.rsqrt()                               # [T]

        # Update z's new component: z_new = r*(p_j - alpha^T z)
        delta_sel = delta[arng, j_star]                         # [T]
        z_new     = r_sel * delta_sel                           # [T]
        z[:, t]   = z_new
        z_norm_sq = z_norm_sq + z_new.pow(2)

        # Update R's new row/column (block construction):
        # gamma = - r * alpha^T R (1xt), batched: [T,1,t] = [T,1,t] @ [T,t,t]
        gamma = - r_sel.view(T, 1, 1) * torch.bmm(alpha_sel.unsqueeze(1), R_tt)  # [T,1,t]

        # Write R's new block
        R[:, t, :t] = gamma.squeeze(1)                          # lower triangular new row
        R[:, :t, t] = 0                                         # upper triangular new column (all 0)
        R[:, t, t]  = r_sel                                     # bottom-right scalar
        # Other elements unchanged (top-left txt)

    return selected_mask



@torch.no_grad()
def otep_batched_chunked(
    scores: torch.Tensor,     # [T, E], expert probability per token
    sigma:  torch.Tensor,     # [E, E], global expert covariance
    k: int,
    chunk_size: int = 256,
    eps: float = 1e-12,
) -> torch.BoolTensor:
    """
    Low-memory batched OTEP: chunk along expert dimension (E direction).
    Returns [T, E] bool routing mask (exactly k True per row).
    """
    assert scores.dim() == 2 and sigma.dim() == 2
    T, E = scores.shape
    assert sigma.shape == (E, E)
    device, dtype = scores.device, scores.dtype

    scores = scores.to(device=device, dtype=dtype)
    sigma  = sigma.to(device=device, dtype=dtype)

    # Result containers
    selected_mask = torch.zeros(T, E, dtype=torch.bool, device=device)
    selected_idx  = torch.full((T, k), -1, dtype=torch.long, device=device)

    # Pre-compute diagonal
    diag = torch.diag(sigma).clamp_min(eps)        # [E]
    inv_sqrt_diag = diag.rsqrt()                   # [E]

    # Inverse Cholesky and z=R p_S packed buffer (only use top-left txt / first t elements)
    R = torch.zeros(T, k, k, dtype=dtype, device=device)   # [T,k,k]
    z = torch.zeros(T, k,     dtype=dtype, device=device)  # [T,k]
    z_norm_sq = torch.zeros(T, dtype=dtype, device=device) # [T]

    # ==================== Round 0: chunked argmax over E ====================
    best0 = torch.full((T,), float("-inf"), dtype=dtype, device=device)
    j0    = torch.full((T,), -1,           dtype=torch.long, device=device)

    for start in range(0, E, chunk_size):
        J = torch.arange(start, min(start + chunk_size, E), device=device)
        # (scores[:,J]**2) / diag[J]
        s_chunk = scores[:, J]                                # [T,C]
        d_chunk = diag[J]                                     # [C]
        sc0 = (s_chunk * s_chunk) / d_chunk.unsqueeze(0)      # [T,C]
        # Update argmax per row
        local_best, local_idx = torch.max(sc0, dim=1)         # [T], [T]
        update = local_best > best0
        j0[update]    = J[local_idx[update]]
        best0[update] = local_best[update]

    # Record first expert
    selected_idx[:, 0] = j0
    selected_mask.scatter_(1, j0.unsqueeze(1), True)

    # Initialize R, z, ||z||^2
    R[:, 0, 0] = inv_sqrt_diag.gather(0, j0)  # R00 = 1/sqrt(Sigma[j0,j0])
    z0 = R[:, 0, 0] * scores.gather(1, j0.unsqueeze(1)).squeeze(1)
    z[:, 0] = z0
    z_norm_sq = z0 * z0

    # ==================== Remaining k-1 rounds ====================
    for t in range(1, k):
        S_t = selected_idx[:, :t]                       # [T,t]
        R_tt = R[:, :t, :t]                             # [T,t,t]
        z_t  = z[:, :t]                                 # [T,t]

        best = torch.full((T,), float("-inf"), dtype=dtype, device=device)
        j_star = torch.full((T,), -1, dtype=torch.long, device=device)

        for start in range(0, E, chunk_size):
            J = torch.arange(start, min(start + chunk_size, E), device=device)
            C = J.numel()

            # beta = Sigma[S, J] -> via column slice + row gather: [T,t,C]
            sigma_J = sigma.index_select(1, J)                  # [E,C]
            sigma_J_exp = sigma_J.unsqueeze(0).expand(T, -1, -1)  # [T,E,C]
            row_index = S_t.unsqueeze(-1).expand(T, t, C)       # [T,t,C]
            beta = torch.gather(sigma_J_exp, 1, row_index)      # [T,t,C]

            # alpha = R * beta (batched): [T,t,t] @ [T,t,C] -> [T,t,C]
            alpha = torch.bmm(R_tt, beta)                       # [T,t,C]

            # s_j = Sigma[j,j] - ||alpha_j||^2
            a2 = alpha.pow(2).sum(dim=1)                        # [T,C]
            s = diag[J].unsqueeze(0) - a2                       # [T,C]
            s = torch.clamp(s, min=eps)
            r = s.rsqrt()                                       # [T,C]

            # a^T z
            aTz = (alpha * z_t.unsqueeze(2)).sum(dim=1)         # [T,C]

            # cand_score = ||z||^2 + (r*(p_j - a^T z))^2
            p_chunk = scores[:, J]                               # [T,C]
            delta = p_chunk - aTz                                # [T,C]
            sc = z_norm_sq.unsqueeze(1) + (r * delta).pow(2)     # [T,C]

            # Mask already selected experts
            sc = sc.masked_fill(selected_mask[:, J], float('-inf'))

            # Update global best within this chunk
            local_best, local_idx = torch.max(sc, dim=1)         # [T]
            update = local_best > best
            j_star[update] = J[local_idx[update]]
            best[update]   = local_best[update]

        # Record this round's selection
        selected_idx[:, t] = j_star
        selected_mask.scatter_(1, j_star.unsqueeze(1), True)

        # Get alpha*, s*, r* for selected j* to update R and z
        arng = torch.arange(T, device=device)

        # To get alpha_sel: need to fetch Sigma[S, j*] and compute alpha = R_tt @ beta_sel
        # First get beta_sel: [T,t]
        sigma_j = sigma.index_select(1, j_star)                 # [E,T]
        sigma_j = sigma_j.transpose(0,1)                        # [T,E]
        beta_sel = torch.gather(sigma_j, 1, S_t)                # [T,t]
        alpha_sel = torch.bmm(R_tt, beta_sel.unsqueeze(2)).squeeze(2)  # [T,t]

        # s_sel, r_sel
        s_sel = (diag.gather(0, j_star) - (alpha_sel * alpha_sel).sum(dim=1)).clamp_min(eps)  # [T]
        r_sel = s_sel.rsqrt()                                                                        # [T]

        # z_new
        p_sel = scores.gather(1, j_star.unsqueeze(1)).squeeze(1)                       # [T]
        aTz_sel = (alpha_sel * z_t).sum(dim=1)                                         # [T]
        z_new = r_sel * (p_sel - aTz_sel)                                              # [T]
        z[:, t] = z_new
        z_norm_sq = z_norm_sq + z_new.pow(2)

        # Update R's new row/column
        gamma = - r_sel.view(T, 1, 1) * torch.bmm(alpha_sel.unsqueeze(1), R_tt)        # [T,1,t]
        R[:, t, :t] = gamma.squeeze(1)                                                 # lower triangular
        R[:, :t, t] = 0                                                                # upper triangular
        R[:, t, t]  = r_sel

    return selected_mask


@torch.no_grad()
def otep_batched_shuffle(
    scores: torch.Tensor,
    sigma: torch.Tensor,
    k: int,
    n_shuffles: int = 8,
    random_state: int | None = None,
    tie_break: str = "none"  # "score" | "random" | "none"
):
    """
    Randomly permute expert dimension multiple times, call otep_batched for selection,
    aggregate occurrence counts, then take Top-k.

    Parameters
    ----------
    scores : [T, E]       expert probability vector per token
    sigma  : [E, E]       global expert covariance (symmetric, positive semi-definite)
    k      : int          number of experts to select per row
    n_shuffles : int      number of permutations
    random_state : int    random seed (for reproducibility)
    tie_break : str       how to break ties when counts are equal:
                          - "score": use original scores with small weight
                          - "random": use small random noise
                          - "none": no processing (topk uses stable index ordering)

    Returns
    -------
    final_mask : [T, E] bool   routing mask with exactly k True per row
    counts     : [T, E] int32  selection count for each expert across shuffles
    """
    assert scores.dim() == 2 and sigma.dim() == 2
    T, E = scores.shape
    assert sigma.shape == (E, E)
    assert 1 <= k <= E, "k must be in range 1..E"

    device = scores.device
    dtype  = scores.dtype

    # Counter (accumulated across shuffles)
    counts = torch.zeros(T, E, dtype=torch.int32, device=device)

    # Random source (reproducible)
    g = torch.Generator(device=device)
    if random_state is not None:
        g.manual_seed(random_state)

    for _ in range(n_shuffles):
        # Permute expert dimension
        perm = torch.randperm(E, generator=g, device=device)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(E, device=device)

        scores_p = scores[:, perm]                               # [T,E]
        sigma_p  = sigma.index_select(0, perm).index_select(1, perm)  # [E,E]

        # Select in permuted coordinate system
        mask_p = otep_batched(scores_p, sigma_p, k)              # [T,E] bool

        # Map back to original coordinates
        mask_back = mask_p[:, inv_perm]                          # [T,E] bool

        # Accumulate occurrence counts
        counts += mask_back.to(counts.dtype)

    # --- Tie breaking (optional, small magnitude doesn't change count dominance) ---
    priority = counts.to(dtype)  # convert counts to float for adding small tie-break
    if tie_break == "score":
        # Use original scores (normalized) with small weight
        mean = scores.mean(dim=1, keepdim=True)
        std  = scores.std(dim=1, keepdim=True).clamp_min(1e-12)
        s_norm = (scores - mean) / std
        priority = priority + 1e-3 * s_norm
    elif tie_break == "random":
        priority = priority + 1e-3 * torch.rand_like(scores, device=scores.device)

    # Take Top-k per row
    topk_vals, topk_idx = torch.topk(priority, k, dim=1, largest=True, sorted=False)  # [T,k]
    final_mask = torch.zeros(T, E, dtype=torch.bool, device=device)
    final_mask.scatter_(1, topk_idx, True)

    return final_mask
