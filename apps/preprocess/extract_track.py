from easymocap.annotator.file_utils import read_json, save_json
from easymocap.config import load_object_from_cmd
import numpy as np
from easymocap.mytools.debug_utils import log, myerror, mywarn, run_cmd
from tqdm import tqdm
import os
from os.path import join

class Tracker:
    def __init__(self, missing_frame=10, thres_iou=0.5) -> None:
        self.cache = {}
        self.max_id = -1
        self.time = 0
        self.dist_mode = 'bbox'
        self.min_accept_dist = thres_iou
        self.missing_frame = missing_frame
        self.failed = {}

    def step(self):
        self.time += 1
        removelist = []
        for pid, track in self.cache.items():
            if self.time - track['end_time'] > self.missing_frame:
                mywarn('[{:06d}] Delete person {:3d} with {:6d} frames'.format(self.time, pid, track['end_time'] - track['start_time']))
                removelist.append(pid)
        for pid in removelist:
            self.failed[pid] = self.cache.pop(pid)

    def init(self, data):
        pid = data['personID']
        self.max_id = max(self.max_id, pid)
        self.cache[pid] = {
            'start_time': self.time,
            'end_time': self.time,
            'missing_frame': [],
            'infos': [data]
        }
        return True, pid
    
    def update(self, data, pid):
        track = self.cache[pid]
        if self.time == track['end_time'] + 1 or self.time != 1:
            track['end_time'] = self.time
        else:
            mywarn('[{:06d}] Refind person {:3d} from {:06d}'.format(self.time, pid, track['end_time']))
            track['end_time'] = self.time
            for f in range(track['end_time'] + 1, self.time):
                track['missing_frame'].append(f)
        track['infos'].append(data)
    
    def calculate_distance(self, data, infos):
        # TODO: require the last frame
        info = infos[-1]
        if self.dist_mode == 'bbox':
            bbox_now = data['bbox']
            bbox_pre = info['bbox']
            area_now = (bbox_now[2] - bbox_now[0])*(bbox_now[3]-bbox_now[1])
            area_pre = (bbox_pre[2] - bbox_pre[0])*(bbox_pre[3]-bbox_pre[1])
            # compute IOU
            # max of left
            xx1 = max(bbox_now[0], bbox_pre[0])
            yy1 = max(bbox_now[1], bbox_pre[1])
            # min of right
            xx2 = min(bbox_now[0+2], bbox_pre[0+2])
            yy2 = min(bbox_now[1+2], bbox_pre[1+2])
            # w h
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            over = (w*h)/(area_pre+area_now-w*h)
            distance = 1 - over
        return distance

    def track(self, data):
        keys = list(self.cache.keys())
        distance = np.zeros(len(keys)) + 999.
        for ikey, key in enumerate(keys):
            if self.cache[key]['end_time'] == self.time:
                # already assigned in current frame
                continue
            else:
                dist = self.calculate_distance(data, self.cache[key]['infos'])
            distance[ikey] = dist
        if (distance > 10).all():
            # all tracks have been assigned in current frame
            return False, -1
        best_id = distance.argmin()
        if distance[best_id] > self.min_accept_dist:
            mywarn('[{:06d}] Tracking failed with distance {}'.format(self.time, distance[best_id]))
            return False, -1
        else:
            self.cache[keys[best_id]]
            self.update(data, keys[best_id])
            return True, keys[best_id]

    def add(self, data):
        flag, pid = self.track(data)
        if not flag:
            log('[{:06d}] Create person {:3d}'.format(self.time, self.max_id+1))
            data['personID'] = self.max_id + 1
            flag, pid = self.init(data)
        return flag, pid
    
    def report(self):
        removelist = []
        for pid, track in self.cache.items():
            if track['end_time'] - track['start_time'] < self.missing_frame:
                removelist.append(pid)
        for pid in removelist:
            self.failed[pid] = self.cache.pop(pid)
        # success
        log('- Tracked detection:')
        for pid, track in self.cache.items():
            log('{:3d} [{:6d}->{:6d}], missing {}'.format(pid, track['start_time'], track['end_time'], track['missing_frame']))

def track2d(datas):
    # sort the first frame by size
    annots0 = datas['annots'][0]
    len_first_frame = len(annots0)
    tracker = Tracker(thres_iou=args.thres_iou)
    annots0.sort(key=lambda x:-(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))
    for i, annot in enumerate(annots0):
        annot['personID'] = i
    for nf, annots in enumerate(datas['annots']):
        if nf == 0:
            # new the tracker
            for annot in annots:
                flag, pid = tracker.init(annot)
            continue
        # track all the frames
        tracker.step()
        annots0.sort(key=lambda x:-(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))
        for annot in annots:
            # greedy match
            flag, pid = tracker.add(annot)
            if flag:
                annot['personID'] = pid
    from easymocap.annotator.file_utils import save_annot
    nFrames = len(data['annname'])
    for nf in tqdm(range(nFrames), desc='writing track'):
        annname = data['annname'][nf]
        annots = data['annots'][nf]
        annots.sort(key=lambda x:x['personID'])
        annots_origin = read_json(annname)
        annots_origin['annots'] = annots
        save_annot(annname, annots_origin)
    tracker.report()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=
    '''
    For common usage:
        python3 apps/preprocess/extract_track.py ${data}
    For fast motion:
        python3 apps/preprocess/extract_track.py ${data} --thres_iou 0.8
    ''')
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--subs', type=str, default=[], nargs='+')
    parser.add_argument('--max', type=int, default=-1)
    parser.add_argument('--thres_iou', type=float, default=0.5)
    parser.add_argument('--annot_track', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    opt_data = ['args.path', args.path]
    annotbase = join(args.path, 'database.json')
    if not os.path.exists(annotbase):
        save_json(annotbase, {})
    if False:
        trackname = join(args.path, 'track.json')
        track_info = read_json(trackname)
        annot_info = {}
        for sub, flag in track_info.items():
            if not os.path.exists(join(args.path, 'images', sub)):
                continue
            annot_info[sub] = {
                'skip': 0,
                'tracked': flag,
                'detected': flag,
                'reconstructed': 0,
            }
        save_json(annotbase, annot_info)
        import ipdb;ipdb.set_trace()
        exit()
    annotbase = join(args.path, 'database.json')
    track_info = read_json(annotbase)
    
    if not os.path.exists(annotbase):
        save_json(annotbase, {})

    if len(args.subs) >0:
        opt_data.append('args.subs')
        opt_data.append(args.subs)
    else:
        # check subs
        subs_all = sorted(os.listdir(join(args.path, 'annots')))
        subs = []
        for sub in subs_all:
            # check if the sub has been tracked
            if sub not in track_info:
                track_info[sub] = {
                    'skip': 0,
                    'tracked': 0,
                    'detected': 0,
                    'reconstructed': 0,
                }
            if track_info[sub]['skip'] or track_info[sub]['tracked']:
                mywarn('- skip {}'.format(sub))
                continue
            len_img = len(os.listdir(join(args.path, 'images', sub)))
            len_ann = len(os.listdir(join(args.path, 'annots', sub)))
            if len_img != len_ann:
                mywarn('- skip {} as no enough detections'.format(sub))
                continue
            subs.append(sub)
        opt_data.append('args.subs')
        opt_data.append(subs)
    dataset = load_object_from_cmd('config/data/multivideo-mp.yml', opt_data)
    for idx in range(len(dataset)):
        # try:
        data = dataset[idx]
        # except:
        #     myerror('- Failed to load {}'.format(dataset.subs[idx]))
        #     continue
        track2d(data)
        imgname = data['imgname'][0]
        sub = os.path.basename(os.path.dirname(imgname))
        valid = input('Does this track right? [y/n]')
        if valid == 'y':
            track_info[sub]['tracked'] = 1
            track_info[sub]['detected'] = 1
        elif args.annot_track:
            # detect of reclip this
            cmd = f'python3 apps/annotation/annot_track.py {args.path} --sub {sub}'
            run_cmd(cmd)
        save_json(annotbase, track_info)
        track_info = read_json(annotbase)
    # check track
    failed_subs = []
    for sub, flag in track_info.items():
        if not flag['tracked']:
            failed_subs.append(sub)
    if len(failed_subs) > 0:
        mywarn('- Success subs: {}'.format(len(track_info.keys()) - len(failed_subs)))
        mywarn('- Failed subs: {}'.format(failed_subs))
        log('Run the following command to annotate the failed tracks:')
        log('python3 apps/annotation/annot_track.py ${{data}} --sub {}'.format(' '.join(failed_subs)))