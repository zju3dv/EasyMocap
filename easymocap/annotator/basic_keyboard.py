from tqdm import tqdm
from .basic_callback import get_key

def print_help(annotator, **kwargs):
    """print the help"""
    print('Here is the help:')
    print(  '------------------')
    for key, val in annotator.register_keys.items():
        # print('    {}: {}'.format(key, ': ', str(val.__doc__)))
        print('    {}: '.format(key, ': '), str(val.__doc__))

def close(annotator, param, **kwargs):
    """quit the annotation"""
    if annotator.working:
        annotator.clear_working()
    else:
        annotator.save_and_quit()
        # annotator.pbar.close()

def get_move(wasd):
    get_frame = {
        'a': lambda x, f: f - 1,
        'd': lambda x, f: f + 1,
        'w': lambda x, f: f - x.step,
        's': lambda x, f: f + x.step
    }[wasd]
    text = {
        'a': 'Move to last frame',
        'd': 'Move to next frame',
        'w': 'Move to last step frame',
        's': 'Move to next step frame'
    }
    clip_frame = lambda x, f: max(0, min(x.nFrames-1, f))
    def move(annotator, **kwargs):
        newframe = get_frame(annotator, annotator.frame)
        newframe = clip_frame(annotator, newframe)
        annotator.frame = newframe
    move.__doc__ = text[wasd]    
    return move

def set_personID(i):
    def func(self, param, **kwargs):
        active = param['select']['bbox']
        if active == -1:
            return 0
        else:
            param['annots']['annots'][active]['personID'] = i
        return 0
    func.__doc__ = "set the bbox ID to {}".format(i)
    return func

def delete_bbox(self, param, **kwargs):
    "delete the person"
    active = param['select']['bbox']
    if active == -1:
        return 0
    else:
        param['annots']['annots'].pop(active)
        param['select']['bbox'] = -1
    return 0

def capture_screen(self, param):
    "capture the screen"
    if param['capture_screen']:
        param['capture_screen'] = False
    else:
        param['capture_screen'] = True

def automatic(self, param):
    "Automatic running"
    keys = input('Enter the ordered key(separate with blank): ').split(' ')
    repeats = int(input('Input the repeat times: (0->{})'.format(len(self.dataset)-self.frame)))
    for nf in tqdm(range(repeats), desc='auto {}'.format('->'.join(keys))):
        for key in keys:
            self.run(key=key)
        if chr(get_key()) == 'q':
            break
        self.run(key='d')

register_keys = {
    'h': print_help,
    'q': close,
    'x': delete_bbox,
    'p': capture_screen,
    'A': automatic
}

for key in 'wasd':
    register_keys[key] = get_move(key)

for i in range(10):
    register_keys[str(i)] = set_personID(i)