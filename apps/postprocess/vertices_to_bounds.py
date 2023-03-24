import os
import numpy as np
from tqdm import tqdm
from easymocap.mytools.reader import read_json
from os.path import join

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    verticesnames = os.listdir(args.path)
    bounds = []
    for vertname in tqdm(verticesnames):
        vertices = read_json(join(args.path, vertname))
        bounds_frame = []
        for data in vertices:
            verts = np.array(data['vertices'])
            bound = np.array([verts.min(axis=0), verts.max(axis=0)])
            bounds_frame.append(bound)
        bounds.append(bounds_frame)
    np.save('test.npy', np.stack(bounds))