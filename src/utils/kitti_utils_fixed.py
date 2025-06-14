"""
Fixed version of computeOverallErr that handles empty sequences.
"""

def computeOverallErr(seq_err):
    t_err = 0
    r_err = 0
    seq_len = len(seq_err)
    
    if seq_len == 0:
        # Return 0 error for empty sequences instead of dividing by zero
        return 0.0, 0.0
    
    for item in seq_err:
        r_err += item[1]
        t_err += item[2]
    ave_t_err = t_err / seq_len
    ave_r_err = r_err / seq_len
    return ave_t_err, ave_r_err