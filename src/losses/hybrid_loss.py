def hybrid_loss(ctc_loss_value, attn_loss_value, lambda_ctc=0.6):
    """
    Returns weighted combination: lambda_ctc * ctc + (1-lambda_ctc) * attn.
    """
    return lambda_ctc * ctc_loss_value + (1.0 - lambda_ctc) * attn_loss_value

def lambda_schedule(epoch, total_epochs, start=0.7, mid=0.5, end=0.3):
    """
    Simple schedule:
      - first 20% epochs: start
      - middle 60%: linear from start->mid
      - final 20%: linear from mid->end
    """
    if total_epochs <= 0:
        return start
    p = epoch / float(total_epochs)
    if p <= 0.2:
        return start
    if p <= 0.8:
        q = (p - 0.2) / 0.6
        return start + q * (mid - start)
    q = (p - 0.8) / 0.2
    return mid + q * (end - mid)
