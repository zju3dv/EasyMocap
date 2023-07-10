import os
from easymocap.mytools.file_utils import write_keypoints3d, write_smpl, write_vertices
from easymocap.annotator.file_utils import save_annot
from os.path import join
from tqdm import tqdm

class Write:
    def __init__(self, output='/tmp', name='keypoints3d') -> None:
        self.output = output
        self.name = name
    
    def __call__(self, keypoints3d):
        for nf in tqdm(range(keypoints3d.shape[0]), desc='writing to {}/{}'.format(self.output, self.name)):
            res = [{
                'id': 0,
                'keypoints3d': keypoints3d[nf]
            }]
            dumpname = join(self.output, self.name, '{:06d}.json'.format(nf))
            write_keypoints3d(dumpname, res)
        return {}

class WriteAll:
    def __init__(self, name, output='/tmp') -> None:
        self.output = output
        self.name = name
    
    def __call__(self, results, meta):
        for nf in tqdm(range(len(results)), desc='writing to {}/{}'.format(self.output, self.name)):
            res = [{'id': r['id'], 'keypoints3d': r['keypoints3d']} for r in results[nf]]
            res.sort(key=lambda x: x['id'])
            imgnames = meta['imgnames'][nf]
            if len(imgnames) > 0:
                name = os.path.basename(imgnames[0])
                name = name.replace('.jpg', '')
            else:
                name = '{:06f}'.format(nf)
            dumpname = join(self.output, self.name, '{}.json'.format(name))
            write_keypoints3d(dumpname, res)

class Write2D:
    def __init__(self, name, output='/tmp') -> None:
        self.output = output
        self.name = name
    
    def __call__(self, results, meta):
        for nf in tqdm(range(len(results)), desc='writing to {}/{}'.format(self.output, self.name)):
            subs = meta['subs'][nf]
            result = results[nf]
            annots_all = {sub: [] for sub in subs}
            for res in result:
                for nv, v in enumerate(res['views']):
                    annots_all[subs[v]].append({
                        'personID': res['id'],
                        'bbox': res['bbox'][nv],
                        'keypoints': res['keypoints2d'][nv],
                    })
            for nv, sub in enumerate(subs):
                annots = {
                    'filename': f'{sub}/{nf:06d}.jpg',
                    'height': meta['image_shape'][nf][nv][0],
                    'width': meta['image_shape'][nf][nv][1],
                    'annots': annots_all[sub],
                    'isKeyframe': False
                }
                dumpname = join(self.output, self.name, sub, '{:06d}.json'.format(nf))
                save_annot(dumpname, annots)

class WriteSMPL:
    def __init__(self, name='smpl', write_vertices=False) -> None:
        self.name = name
        # TODO: make available
        self.write_vertices = write_vertices
    
    def __call__(self, params=None, results=None, meta=None, model=None):
        results_all = []
        if results is None and params is not None:
            # copy params to results
            results = {0: {'params': params, 'keypoints3d': None, 'frames': list(range(len(params['Rh'])))}}
        for index in tqdm(meta['index'], desc=self.name):
            results_frame = []
            for pid, result in results.items():
                if index >= result['frames'][0] and index <= result['frames'][-1]:
                    frame_rel = result['frames'].index(index)
                    results_frame.append({
                        'id': pid,
                        # 'keypoints3d': result['keypoints3d'][frame_rel]
                    })
                    for key in ['Rh', 'Th', 'poses', 'shapes']:
                        if result['params'][key].shape[0] == 1:
                            results_frame[-1][key] = result['params'][key]
                        else:
                            results_frame[-1][key] = result['params'][key][frame_rel:frame_rel+1]
                    param = results_frame[-1]
                    pred = model(param)['keypoints'][0]
                    results_frame[-1]['keypoints3d'] = pred
                    if self.write_vertices:
                        vert = model(param, ret_vertices=True)['keypoints'][0]
                        results_frame[-1]['vertices'] = vert
            write_smpl(join(self.output, self.name, '{:06d}.json'.format(meta['frame'][index])), results_frame)
            write_keypoints3d(join(self.output, 'keypoints3d', '{:06d}.json'.format(meta['frame'][index])), results_frame)
            if self.write_vertices:
                write_vertices(join(self.output, 'vertices', '{:06d}.json'.format(meta['frame'][index])), results_frame)
                for res in results_frame:
                    res.pop('vertices')
            results_all.append(results_frame)
        return {'results_perframe': results_all}