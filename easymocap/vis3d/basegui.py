import open3d as o3d
import os
from os.path import join
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import cv2

class BaseWindow: # this window is the basic of Open3D new render style
    colormap = {
        0: (94/255, 124/255, 226/255, 1.), # 青色
        1: (255/255, 200/255, 87/255, 1.), # yellow
        2: (74/255.,  189/255.,  172/255., 1.), # green
        3: (8/255, 76/255, 97/255, 1.), # blue
        4: (219/255, 58/255, 52/255, 1.), # red
        5: (77/255, 40/255, 49/255, 1.), # brown
        1000: [1., 0., 0., 1.],
        1001: [0., 1., 0., 1.],
        1002: [0., 0., 1., 1.],
    }
    def __init__(self, out, panel_rate=0.2) -> None:
        self.out = out
        if out is not None:
            os.makedirs(out, exist_ok=True)
        self.window = gui.Application.instance.create_window(
            "EasyMocap Visualization", 1920, 1080)
        w = self.window  # for more concise code
        em = w.theme.font_size
        self.em = em
        # set the frame region
        self.panel_rate = panel_rate
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        # create scene
        self.scene = self.add_scene(w)
        self.window.add_child(self.scene)
        self.panel = self.add_panel(w, em)
        self.factory = {}
        self._render_cnt = 0
        self.is_done = False
    
    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        panel_width = contentRect.width * self.panel_rate
        self.scene.frame = gui.Rect(contentRect.x, contentRect.y,
                                       contentRect.width - panel_width,
                                       contentRect.height)
        self.panel.frame = gui.Rect(self.scene.frame.get_right(),
                                    contentRect.y, panel_width,
                                    contentRect.height)

    def _on_close(self):
        self.is_done = True
        return True  # False would cancel the close

    def add_scene(self, window):
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        scene.scene.set_background([1, 1, 1, 1])
        scene.scene.scene.set_sun_light(
            [-1, -1, -1],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        scene.scene.scene.enable_sun_light(True)
        scene.scene.show_skybox(True)
        # scene.scene.show_ground_plane(True)
        scene.scene.show_axes(True)
        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5],
                                                    [5, 5, 5])

        if False:
            scene.setup_camera(60, bbox, [0, 0, 0])
            # fov, bounds, center_of_rotation
        else:
            K = np.array([
                1000., 0., 500.,
                0., 1000., 500.,
                0., 0., 1.
            ]).reshape(3, 3)
            T = np.array([0., 0., 10.]).reshape(3, 1)
            RT = np.eye(4)
            R = cv2.Rodrigues(np.array([1., 0., 0.])*(np.pi/6 +np.pi/2))[0]
            RT[:3, 3:] = T
            RT[:3, :3] = R
            scene.setup_camera(K, RT, 1000, 1000, bbox)
        return scene

    def add_color(self, name, init, callback):
        _col = gui.ColorEdit()
        _col.set_on_value_changed(callback)
        _col.color_value.set_color(init[0], init[1], init[2])
        label = gui.Label(name)
        return label, _col
    
    def add_vec3d(self, name, init, callback):
        label = gui.Label(name)
        # Create a widget for showing/editing a 3D vector
        vedit = gui.VectorEdit()
        vedit.vector_value = init
        vedit.set_on_value_changed(callback)
        return label, vedit
    
    def add_panel(self, window, em):
        panel = gui.Vert(20, 
            gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        window.add_child(panel)
        return panel

    def get_default_mat(self, pid=-1, color=[0., 1., 1., 1.], shader='defaultLit'):
        mat = rendering.MaterialRecord()
        if pid != -1 and pid in list(self.colormap.keys()):
            color = self.colormap[pid]
        mat.base_color = color
        mat.shader = shader
        return mat

    def remove_geometry(self, name):
        if name in self.factory.keys():
            self.scene.scene.remove_geometry(name)

    def add_geometry(self, name, geom, mat=None):
        if name in self.factory.keys():
            self.scene.scene.remove_geometry(name)
        self.factory[name] = {
            'geom': geom,
            'mat': mat
        }
        self.scene.scene.add_geometry(name, geom, mat)
    
    @staticmethod
    def _chessboard_params():
        params = {
            'center': [0, 0, -0.1],
            'xdir': [1, 0, 0],
            'ydir': [0, 1, 0],
            'step': 1,
            'xrange': 5,
            'yrange': 5,
            'white': [1., 1., 1.],
            'black': [0.5, 0.5, 0.5],
            'two_sides': True,
        }
        return params

    def add_camera(self, path=None, cameras=None):
        from ..visualize.o3dwrapper import create_camera
        mesh = create_camera(path=path, cameras=cameras)
        self.add_geometry('camera', mesh, self.get_default_mat(color=[1., 1., 1., 1.]))

    def add_chessboard(self, params):
        from ..visualize.o3dwrapper import create_ground
        mesh = create_ground(**params)
        self.add_geometry('chessboard', mesh, self.get_default_mat(color=[1., 1., 1., 1.]))

    def update_geometry(self, name):
        self.scene.scene.remove_geometry(name)
        self.scene.scene.add_geometry(name, self.factory[name]['geom'], self.factory[name]['mat'])
    
    def capture_callback(self, image):
        if self.out is None:
            print('[Vis] Please set output folder by --out <path>')
            return 0
        outname = join(self.out, '{:06d}.jpg'.format(self._render_cnt))
        img = image
        quality = 9  # png
        if outname.endswith(".jpg"):
            quality = 100
        o3d.io.write_image(outname, img, quality)
        print('[Vis] render image to {}'.format(outname))
        self._render_cnt += 1
    
    def add_chessboard_widget(self, param):
        chessboard = gui.CollapsableVert("Chessboard setting", 10,
            gui.Margins(self.em, 0, 0, 0))

        grid = gui.VGrid(2, 10)
        label0, widget0 = self.add_color(
            'black', param['black'], self._on_chess_col_0)
        label1, widget1 = self.add_color(
            'white', param['white'], self._on_chess_col_1)
        grid.add_child(label0)
        grid.add_child(widget0)
        grid.add_child(label1)
        grid.add_child(widget1)
        # 增加棋盘格的范围选项
        label_range, widget_range = self.add_vec3d(
            'ranges', 
            [param['xrange'], param['yrange'], param['step']],
            self._on_chess_range)
        grid.add_child(label_range)
        grid.add_child(widget_range)
        # 增加棋盘格的中心选项
        label_center, widget_center = self.add_vec3d(
            'center', 
            param['center'],
            self._on_chess_center)
        grid.add_child(label_center)
        grid.add_child(widget_center)
        chessboard.add_child(grid)
        self.panel.add_child(chessboard)
    
    def _on_chess_center(self, ranges):
        for i in range(3):
            self.param_chessboard['center'][i] = float(ranges[i])
        self.add_chessboard(self.param_chessboard)

    def _on_chess_range(self, ranges):
        self.param_chessboard['xrange'] = int(ranges[0])
        self.param_chessboard['yrange'] = int(ranges[1])
        self.param_chessboard['step'] = float(ranges[2])
        self.add_chessboard(self.param_chessboard)

    def _on_chess_col_0(self, new_color):
        color = [
            new_color.red, new_color.green,
            new_color.blue
        ]
        self.param_chessboard['black'] = color
        self.add_chessboard(self.param_chessboard)
    
    def _on_chess_col_1(self, new_color):
        color = [
            new_color.red, new_color.green,
            new_color.blue
        ]
        self.param_chessboard['white'] = color
        self.add_chessboard(self.param_chessboard)