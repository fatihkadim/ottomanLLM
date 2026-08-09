from math import sin
from math import cos
import torch


def frequence_R(q):
    """
    Q∈R ** Lxd
    K∈R ** Lxd
    """
    d = q.shape[-1] # son dim

    i_list = []
    for i in range(0,d // 2):
        i_list.append(i)

    i_list_tensor = torch.tensor(i_list)
    inv_freq = 1 / (10000) ** (2*i_list_tensor/d)

    return inv_freq

def rotate_half(x):
    even = x[:, 0::2]
    odd = x[:, 1::2]

    new_even = -odd
    new_odd = even

    x_new = torch.stack([new_even, new_odd], dim=-1)
    x_new = x_new.reshape(x.shape)

    return x_new

def apply_rotary_emb(q,k):
    #angles = positions x inf_freq
    token_count = q.shape[0]
    p = []
    for i in range(0,token_count):
        p.append(i)
    positions = torch.tensor(p).unsqueeze(1)
    inv_freq = frequence_R(q,k)
    angles = positions * inv_freq

    cos_values = torch.cos(angles)
    sin_values = torch.sin(angles)

    cos_values = cos_values.repeat_interleave(2, dim=-1)
    sin_values = sin_values.repeat_interleave(2, dim=-1)

    q_new = q*cos_values + rotate_half(q)*sin_values
    k_new = k*cos_values + rotate_half(k)*sin_values    

    return q_new,k_new