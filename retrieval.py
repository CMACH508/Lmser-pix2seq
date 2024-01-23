import torch
import numpy as np
from hyper_params import hp

z_code = []
mask_z_code = []

z_root = './results/0.0/npz/' # z of original full sketches. (0.0 fixed!)
mask_z_root = './results/0.0/retnpz/' # z of healed sketches. (0.0, 0.1, 0.3, 0.5)

if __name__ == '__main__':
    for cat in hp.category:
        print(f'{cat} loading')
        z1 = np.load(z_root + cat, allow_pickle=True)
        z_code.append(torch.from_numpy(z1['z']))
        z2 = np.load(mask_z_root + cat, allow_pickle=True)
        mask_z_code.append(torch.from_numpy(z2['z']))

    z_code = torch.cat(z_code,0).view(len(z_code*2500), -1).cuda()
    mask_z_code = torch.cat(mask_z_code,0).view(len(z_code*2500), -1).cuda()

    ans = []
    correct_1 = 0
    correct_10 = 0
    correct_50 = 0

    for i in range(len(z_code)):
        dist = torch.norm(z_code[i].view(-1, z_code.shape[-1])[:, None] - mask_z_code, 2, 2)
        sorted_index = torch.argsort(dist).detach().cpu()
        if i == sorted_index[0, 0]:
            correct_1 += 1
        if i in sorted_index[0, :9]:
            correct_10 += 1
        if i in sorted_index[0, :49]:
            correct_50 += 1

    print('top1:', correct_1 / len(z_code))
    print('top10:', correct_10 / len(z_code))
    print('top50:', correct_50 / len(z_code))
    print('done')
