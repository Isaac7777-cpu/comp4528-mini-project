import torch


def top_k_exp(): 
    torch.manual_seed(0)
    logits_BlV = torch.randn(3, 5, 4) * 5
    top_k = 2       # Keeping only top-2 logits per time step

    print("Original logits: ")
    print(logits_BlV)

    topk_results = logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)   # Note that this will return basically a tuple 
                                                                                # but it is a custom datatype 'topk'
    # print("Top k results:")
    # print(topk_results)

    topk_vals = topk_results[0]
    # print("Top_k values: ")
    # print(topk_vals)

    threshold = topk_vals.amin(dim=-1, keepdim=True)                            # The threshold represents the minimum values among the top-k 
                                                                                # values.

    # print("Threshold:")
    # print(threshold)

    idx_to_remove = logits_BlV < threshold
    print("Index to remove: ")
    print(idx_to_remove)

    masked_logits = logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    print("Masked_logit")
    print(masked_logits)

def top_p_exp():
    logits_BlV = torch.randn(3, 5, 4) * 5
    top_p = 0.9
    print("Logits:")
    print(logits_BlV)
    
    # Get the sorting and the index first.
    sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
    print("Sorted Logits:")
    print(sorted_logits)
    # print("Sorted Index")
    # print(sorted_idx)
    
    # Obtain the probability and the cumulative one
    individual_prob = sorted_logits.softmax(dim=-1)
    cum_prob = individual_prob.cumsum(dim=-1)
    # print("Individual Probs:")
    # print(individual_prob)
    print("Cumulative Probs:")
    print(cum_prob)
    
    sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
    sorted_idx_to_remove[..., -1:] = False
    # print("Sorted Index to Remove:")
    # print(sorted_idx_to_remove)
    
    logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    print("Updated logits_BlV:")
    print(logits_BlV)


# top_k_exp()
top_p_exp()
