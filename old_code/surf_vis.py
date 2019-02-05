from draw import draw
import surf_dep
import nibabel as nib
import numpy as np
# i suggest taking a look at lh_ vertices, its a numpy array where every row is x,y,z coordinates of each vertex point of the freesurfer surface.

# draw the mesh points in treeD space with vispy
#draw(lh_verts, size=6) # change the size parameter around for funzies

a = surf_dep.surf4search(4,'parahippocampal')

vertices =np.vstack([a.lh_verts[:,:3],a.rh_verts[:,:3]])

roi_IDs = np.concatenate([a.lh_roi_vert_IDs,a.rh_roi_vert_IDs+len(a.lh_verts)])


draw(vertices,roi_IDs, size=1)

draw(np.vstack([a.lh_verts[:,:3],rh_verts[:,:3]]))