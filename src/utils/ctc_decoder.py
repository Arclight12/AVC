import torch

def ctc_best_path_decode(log_probs, blank_id=None):
    """
    Greedy decode for CTC log probabilities.
    Args:
        log_probs: (B, T, C) or (T, B, C) tensor.
        blank_id: ID of the blank token.
    Returns:
        List[List[int]]: Decoded sequences.
    """
    # Ensure (B, T, C)
    if log_probs.shape[1] != log_probs.shape[0] and log_probs.shape[0] > log_probs.shape[1]:
         # Heuristic check if T is first dim. Usually T > B.
         # But let's assume input is (B, T, C) as per standard batch_first=True in many places,
         # OR check the trainer output.
         # The trainer output for GRU was (T//4, B, C).
         pass
         
    # If input is (T, B, C), permute to (B, T, C)
    if log_probs.shape[0] > log_probs.shape[1]: 
        # Likely (T, B, C)
        log_probs = log_probs.transpose(0, 1)
        
    # Now (B, T, C)
    probs = torch.exp(log_probs)
    argmax = torch.argmax(probs, dim=2) # (B, T)
    
    decoded_batch = []
    for i in range(argmax.size(0)):
        seq = []
        prev = -1
        for t in range(argmax.size(1)):
            idx = argmax[i, t].item()
            if idx != blank_id:
                if idx != prev:
                    seq.append(idx)
            prev = idx
        decoded_batch.append(seq)
        
    return decoded_batch
