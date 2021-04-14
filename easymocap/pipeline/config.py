
class Config:
    OPT_R = False
    OPT_T = False
    OPT_POSE = False
    OPT_SHAPE = False
    OPT_HAND = False
    OPT_EXPR = False
    ROBUST_3D_ = False
    ROBUST_3D = False
    verbose = False
    model = 'smpl'
    device = None
    def __init__(self, args=None) -> None:
        if args is not None:
            self.verbose = args.verbose
            self.model = args.model
            self.ROBUST_3D_ = args.robust3d