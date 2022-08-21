'''
  @ Date: 2021-04-15 17:39:34
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-05-23 15:06:00
  @ FilePath: /EasyMocapPublic/easymocap/annotator/basic_keyboard.py
'''
from glob import glob
from tqdm import tqdm
from .basic_callback import get_key

def print_help(annotator, **kwargs):
    """print the help"""
    print('Here is the help:')
    print(  '------------------')
    for key, val in annotator.register_keys.items():
        if isinstance(val, list):
            print('    {}: '.format(key, ': '), str(val[0].__doc__))
            for v in val[1:]:
                print('       ', str(v.__doc__))
        else:
            print('    {}: '.format(key, ': '), str(val.__doc__))

def print_help_mv(annotator, **kwargs):
    print_help(annotator)
    print(  '------------------')
    print('Here is the help for each view:')
    print(  '------------------')
    for key, val in annotator.register_keys_view.items():
        print('    {}: '.format(key, ': '), str(val.__doc__))

def close(annotator, **kwargs):
    """quit the annotation"""
    if annotator.working:
        annotator.set_frame(annotator.frame)
    else:
        annotator.save_and_quit()
        # annotator.pbar.close()
def close_wo_save(annotator, **kwargs):
    """quit the annotation without saving"""
    annotator.save_and_quit(key='n')

def skip(annotator, **kwargs):
    """skip the annotation"""
    annotator.save_and_quit(key='y')

def get_any_move(df):
    get_frame = lambda x, f: f + df
    clip_frame = lambda x, f: max(0, min(x.nFrames-1, f))
    def move(annotator, **kwargs):
        newframe = get_frame(annotator, annotator.frame)
        newframe = clip_frame(annotator, newframe)
        annotator.frame = newframe
    move.__doc__ = '{} frames'.format(df)
    return move

def get_move(wasd):
    get_frame = {
        'a': lambda x, f: f - 1,
        'd': lambda x, f: f + 1,
        'w': lambda x, f: f - 10,
        's': lambda x, f: f + 10,
        'f': lambda x, f: f + 100,
        'g': lambda x, f: f - 100,
    }[wasd]
    text = {
        'a': 'Move to last frame',
        'd': 'Move to next frame',
        'w': 'Move to last step frame',
        's': 'Move to next step frame',
        'f': 'Move to last step frame',
        'g': 'Move to next step frame'
    }
    clip_frame = lambda x, f: max(x.start, min(x.nFrames-1, min(x.end-1, f)))
    def move(annotator, **kwargs):
        newframe = get_frame(annotator, annotator.frame)
        newframe = clip_frame(annotator, newframe)
        annotator.frame = newframe
    move.__doc__ = text[wasd]
    return move

def set_personID(i):
    def func(self, param, **kwargs):
        active = param['select']['bbox']
        if active == -1 and active >= len(param['annots']['annots']):
            return 0
        else:
            param['annots']['annots'][active]['personID'] = i
        return 0
    func.__doc__ = "set the bbox ID to {}".format(i)
    return func

def choose_personID(i):
    def func(self, param, **kwargs):
        for idata, data in enumerate(param['annots']['annots']):
            if data['personID'] == i:
                param['select']['bbox'] = idata
        return 0
    func.__doc__ = "choose the bbox of ID {}".format(i)
    return func

def capture_screen(self, param):
    "capture the screen"
    if param['capture_screen']:
        param['capture_screen'] = False
    else:
        param['capture_screen'] = True

remain = 0
keys_pre = []

def cont_automatic(self, param):
    "continue automatic"
    global remain, keys_pre
    if remain > 0:
        keys = keys_pre
        repeats = remain
    else:
        print('Examples: ')
        print('  - noshow r t: automatic removing and tracking')
        print('  - noshow nostop r t r c: automatic removing and tracking, if missing, just copy')
        keys = input('Enter the ordered key(separate with blank): ').split(' ')
        keys_pre = keys
        try:
            repeats = int(input('Input the repeat times(0->{}): '.format(len(self.dataset)-self.frame)))
        except:
            repeats = 0
        if repeats == -1:
            repeats = len(self.dataset)
        repeats = min(repeats, len(self.dataset)-self.frame+1)
    if len(keys) < 1:
        return 0
    noshow = 'noshow' in keys
    if noshow:
        self.no_img = True
    nostop = 'nostop' in keys
    param['stop'] = False
    for nf in tqdm(range(repeats), desc='auto {}'.format('->'.join(keys))):
        for key in keys:
            self.run(key=key, noshow=noshow)
        if chr(get_key()) == 'q' or (param['stop'] and not nostop):
            remain = repeats - nf
            break
        self.run(key='d', noshow=noshow)
    else:
        remain = 0
        keys_pre = []
    self.no_img = False

def automatic(self, param):
    "Automatic running"
    global remain, keys_pre
    print('Examples: ')
    print('  - noshow r t: automatic removing and tracking')
    print('  - noshow nostop r t r c: automatic removing and tracking, if missing, just copy')
    keys = input('Enter the ordered key(separate with blank): ').split(' ')
    keys_pre = keys
    try:
        repeats = int(input('Input the repeat times(0->{}): '.format(self.nFrames-self.frame)))
    except:
        repeats = 0
    repeats = min(repeats, self.nFrames-self.frame+1)
    if len(keys) < 1:
        return 0
    noshow = 'noshow' in keys
    if noshow:
        self.no_img = True
    nostop = 'nostop' in keys
    param['stop'] = False
    for nf in tqdm(range(repeats), desc='auto {}'.format('->'.join(keys))):
        for key in keys:
            self.run(key=key, noshow=noshow)
        if chr(get_key()) == 'q' or (param['stop'] and not nostop):
            remain = repeats - nf
            break
        self.run(key='d', noshow=noshow)
    else:
        remain = 0
        keys_pre = []
    self.no_img = False

def set_keyframe(self, param):
    "set/unset the key-frame"
    param['annots']['isKeyframe'] = not param['annots']['isKeyframe']

register_keys = {
    'h': print_help,
    'H': print_help_mv,
    'q': close,
    'Q': close_wo_save,
    ' ': skip,
    'p': capture_screen,
    'A': automatic,
    'z': cont_automatic,
    'k': set_keyframe
}

for key in 'wasdfg':
    register_keys[key] = get_move(key)

for i in range(5):
    register_keys[str(i)] = set_personID(i)
    register_keys['s'+str(i)] = choose_personID(i)