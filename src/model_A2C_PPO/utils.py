import numpy as np

def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = chunk[1:]
    return input_seq, target_seq 

def split_by_run(sequence, dones):
    done_idx = np.nonzero(dones)
    done_idx = np.append(done_idx, len(dones)-1)
    start = 0
    seq = [[] for _ in done_idx]
    for c, idx in enumerate(done_idx):
        end = idx+1
        seq[c] = sequence[start:idx+1]
        start = end
    return seq

def zero_pad_sequence_list(sequence_list):
    max_length = 0
    for _ in sequence_list:
        max_length = np.maximum(max_length,len(_))
    for s in sequence_list:
        np.append(s, [0 for _ in range(max_length-len(s))])
    return sequence_list

def n_step_sequences(sequence, n_steps=10):
    obs_seq = []
    for i in range(len(sequence)-n_steps):
        obs_seq.append(sequence[i:i+n_steps])
    return np.array(obs_seq)
    
