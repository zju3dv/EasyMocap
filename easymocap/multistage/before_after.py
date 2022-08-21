import torch

class Remove:
    def __init__(self, key, index=[], ranges=[]) -> None:
        self.key = key
        self.ranges = ranges
        self.index = index

    def before(self, body_params):
        val = body_params[self.key]
        if self.ranges[0] == 0:
            val_zeros = torch.zeros_like(val[:, :self.ranges[1]])
            val = torch.cat([val_zeros, val[:, self.ranges[1]:]], dim=1)
        body_params[self.key] = val
        return body_params

class RemoveHand:
    def __init__(self, start=60) -> None:
        pass

    def before(self, body_params):
        poses = body_params['poses']
        val_zeros = torch.zeros_like(poses[:, 60:])
        val = torch.cat([poses[:, :60], val_zeros], dim=1)
        body_params['poses'] = val
        return body_params

class Keep:
    def __init__(self, key, ranges=[], index=[]) -> None:
        self.key = key
        self.ranges = ranges
        self.index = index
    
    def before(self, body_params):
        val = body_params[self.key]
        val_zeros = val.detach().clone()
        if len(self.ranges) > 0:
            val_zeros[..., self.ranges[0]:self.ranges[1]] = val[..., self.ranges[0]:self.ranges[1]]
        elif len(self.index) > 0:
            val_zeros[..., self.index] = val[..., self.index]
        body_params[self.key] = val_zeros
        return body_params
    
    def final(self, body_params):
        return body_params

class VPoser2Full:
    def __init__(self, key) -> None:
        pass

    def __call__(self, body_model, body_params, infos):
        if not 'Embedding' in body_model.__class__.__name__:
            return body_params
        poses = body_params['poses']
        poses_full = body_model.decode(poses, add_rot=False)
        body_params['poses'] = poses_full
        return body_params