
import numpy as np
from scipy.stats import zscore
import vispy.scene


def draw(vertices, roi_IDs=None, subset_IDs=None, subsubset_IDs=None, 
    center_ID=None, activations=None, size=6):
    canvas = vispy.scene.SceneCanvas(keys=None, show=True)
    view = canvas.central_widget.add_view()
    if roi_IDs is None:
        rgbs = (0, 1, 0, 1)
    else:
        rgbs = np.zeros([len(vertices),4]) # base rgb mat
        rgbs[:,3] = 1 # alpha column
        rgbs[:,:3] = .5 # all points gray
        rgbs[roi_IDs,1] = 1 # green-out roi
        if subset_IDs is not None:
            rgbs[subset_IDs,:3] = [1, 0, 0] # red-out subset
            if subsubset_IDs is not None:
                rgbs[subsubset_IDs,:3] = [1, 1, 1] # white-out subset

    # options to plot random slice of searchlight activation patterns
    if activations is not None:
        # Take a random slice from the activations.
        IDs = subsubset_IDs
        data = zscore(activations[10,:]) # normalize data
        # rescale it for rgb values between 0:1
        data = (data - np.max(data))/-np.ptp(data)
        # combine rescaled data with other rgb channels
        data_color = np.hstack([np.repeat(0,len(data)).reshape(-1,1),
                                np.repeat(0,len(data)).reshape(-1,1),
                                data.reshape(-1,1)])
        rgbs[IDs,:3] = data_color # blue-scale activation pattern

    if center_ID is not None:
        rgbs[center_ID,:3] = [1, 1, 0] # white-out center

    # Draw.
    scatter = vispy.scene.visuals.Markers()
    scatter.set_data(vertices, edge_color=rgbs, edge_width=.0001, face_color=rgbs, size=size)
    view.add(scatter)
    view.camera = 'turntable'  # or try 'arcball'




# pcloud.draw(s.lh_verts[:,:3], s.lh_roi_vert_IDs, size=3)
# pcloud.draw(s.lh_verts[:,:3], 
#     s.lh_roi_vert_IDs, s.sphere_vert_IDs, size=3)

# pcloud.draw(s.lh_verts[:,:3], 
#     s.sphere_vert_IDs, s.flat_vert_IDs, size=3)

# # activations
# pcloud.draw(s.lh_verts[:,:3], 
#     s.lh_roi_vert_IDs, s.sphere_vert_IDs, s.flat_vert_IDs, 
#     center_vert, activations, size=3)

# # no activations
# pcloud.draw(s.lh_verts[:,:3], 
#     s.lh_roi_vert_IDs, s.sphere_vert_IDs, s.flat_vert_IDs, 
#     size=3)

# canvas.bgcolor=(0,0,0) # change background color
# axis = visuals.XYZAxis(parent=view.scene) # add a colored 3D axis for orientation
# if __name__ == '__main__':
#     import sys
#     if sys.flags.interactive != 1:
#         vispy.app.run()