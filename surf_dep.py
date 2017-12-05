import os
import numpy as np
import nibabel as nib
from scipy.stats import zscore
from scipy.signal import detrend
from sklearn.linear_model import LogisticRegression

'''
Setup steps:
- load the anatomical surface data (graymid) for vertex and face IDs (also for visualization)
- load the functional surface data for analysis
- load an ROI label to extract a subset of the functional surface (ultimately cortex)
Analysis steps:
- extract surface ROI from complete surface
- run searchlight on surface ROI (ROI_surf)
- save results
- map vertices back to voxel space
'''

class surf4search(object):
    def __init__(self, subj_num, roi):
        
        # directories
        self.base_dir = os.path.expanduser('~')+'/GoogleDrive/FC_FMRI_DATA/Sub{:03d}fs'.format(subj_num)

        # brain parameters
        self.surface = 'pial'
        self.roi = roi
        # attributes
        #attributes = np.genfromtxt(self.base_dir+'/attributes.txt', skip_header=1)
        #self.targets = attributes[:,0].astype(np.int)
        #self.chunks = attributes[:,1].astype(np.int)
        
        # searchlight parameters
        ''' The euclidean radius is the "sub-sphere" who's purpose is to constrain computations 
        of the searchlight at each node. This sphere should be at least as large as the searchlight
        radius. The geodesic radius is the distance across the surface (of primary interest). '''
        self.euclidean_radius = 20 # mm?
        self.geodesic_radius = 20
        self.clf = LogisticRegression()
        self.n_perms = 5

        #### Load everything during initialization.
        self.load_anatomical_surface()
        #self.load_functional_surface(preprocess=False)
        self.load_surface_roi(roi=roi)
        print 'Loaded {surface} with {roi} as ROI.'.format(surface=self.surface, roi=roi)
        ####


        # for storing results
        # fill test results with [vertex_ID, vert_count, redundant, test_score, p_val]
        # fill perm results with [vertex_ID, shuf_score, ..., shuf_score]
        self.results_test = np.full([self.lh_roi_vert_IDs.size, 5], np.nan)
        self.results_perm = np.full([self.lh_roi_vert_IDs.size, 1+self.n_perms], np.nan)


    def load_anatomical_surface(self):
        # Load main surface, which will later be subset.
        self.lh_surf_fname = self.base_dir+'/surf/lh.'+self.surface
        self.rh_surf_fname = self.base_dir+'/surf/rh.'+self.surface
        self.lh_verts, self.lh_faces = nib.freesurfer.read_geometry(self.lh_surf_fname)
        self.rh_verts, self.rh_faces = nib.freesurfer.read_geometry(self.rh_surf_fname)
        self.lh_vert_IDs = np.arange(self.lh_verts.shape[0]).reshape(-1,1)
        self.rh_vert_IDs = np.arange(self.rh_verts.shape[0]).reshape(-1,1)
        # add 4th column to carry original node IDs (necessary for searchlight)
        # Add a column to the node array, which carries around the original node IDs.
        # This array will keep getting smaller, but in the end we need the node IDs 
        # in full surface space to index from the node activation data.
        self.lh_verts = np.hstack([self.lh_verts, self.lh_vert_IDs])
        self.rh_verts = np.hstack([self.rh_verts, self.rh_vert_IDs])

    def load_functional_surface(self, preprocess=True):
        # MGH images hold TRxVERTEX activation values (all runs should be concatenated here).
        self.lh_func_surf_fname = self.base_dir+'/bold/lh.'+self.surface+'.mgh'
        self.rh_func_surf_fname = self.base_dir+'/bold/rh.'+self.surface+'.mgh'
        self.lh_func_surf_img = nib.freesurfer.mghformat.load(self.lh_func_surf_fname)
        self.rh_func_surf_img = nib.freesurfer.mghformat.load(self.rh_func_surf_fname)
        # Extract the activations from the mgh images.
        # Transpose and reduce to 2D so array is [n_trs, n_nodes]
        self.lh_surf_data = np.squeeze(self.lh_func_surf_img.get_data().T)
        self.rh_surf_data = np.squeeze(self.rh_func_surf_img.get_data().T)
        
        # Preprocess functional surface data.
        if preprocess==True:
            # Detrend.
            run_breakpts = [ np.argmax(self.chunks==r) for r in np.unique(self.chunks) ]
            lh_detrended = detrend(self.lh_surf_data, axis=0, type='linear', bp=run_breakpts)
            rh_detrended = detrend(self.rh_surf_data, axis=0, type='linear', bp=run_breakpts)
            # Zscore.
            self.lh_surf_data = zscore(lh_detrended, axis=0, ddof=0) # each voxel (separately) across time
            self.rh_surf_data = zscore(rh_detrended, axis=0, ddof=0)
            # Shift.
            self.chunks[3:] = self.chunks[:-3]
            self.chunks = self.chunks[3:]
            self.targets = self.targets[3:]
            self.lh_surf_data = self.lh_surf_data[3:,:]
            self.rh_surf_data = self.rh_surf_data[3:,:]
            # Remove rest periods (and beginning shift)
            nonrest_indx = self.targets!=0
            self.chunks = self.chunks[nonrest_indx]
            self.targets = self.targets[nonrest_indx]
            self.lh_surf_data = self.lh_surf_data[nonrest_indx,:]
            self.rh_surf_data = self.rh_surf_data[nonrest_indx,:]
            # trial_averaged_roi = np.mean(final_array_output.reshape(-1, num_TRs_in_example, num_roi_voxels), axis=1)
            # trial_averaged_runs = np.mean(final_runs_output.reshape(-1, num_TRs_in_example), axis=1)
            # trial_averaged_labels = np.mean(final_label_output.reshape(-1, num_TRs_in_example), axis=1)


    def load_surface_roi(self, roi=None):
        ''' Use a freesurfer label file to extract a subset of the surface mesh. This reduces 
        number of nodes and faces. It is expected to at least extract the cortex from the
        complete surface. Freesurfer surface files include the medial wall and (maybe?) problematic 
        areas around subcortical areas. The cortex label includes only the cortex nodes. '''
        if roi is None:
            roi = self.roi
        else:
            self.roi = roi
        # First load the label file to get the vertex indices of the ROI.
        self.lh_roi_fname = self.base_dir+'/annot2label/lh.'+roi+'.label'
        self.rh_roi_fname = self.base_dir+'/annot2label/rh.'+roi+'.label'
        self.lh_roi_vert_IDs = nib.freesurfer.io.read_label(self.lh_roi_fname)
        self.rh_roi_vert_IDs = nib.freesurfer.io.read_label(self.rh_roi_fname)

        # Extract ROI surface from complete surface.
        self.lh_roi_verts = self.lh_verts[self.lh_roi_vert_IDs, :]
        self.rh_roi_verts = self.rh_verts[self.rh_roi_vert_IDs, :]

        # Extract desired subset of nodes from the functional vertices.
        # NOTE: the original node IDs are still held in 4th column.
        #self.lh_roi_data = self.lh_surf_data[:, self.lh_roi_vert_IDs]
        #self.rh_roi_data = self.rh_surf_data[:, self.rh_roi_vert_IDs]

        # Extract only faces that have all 3 nodes within the desired subset.
        # (gdist crashes otherwise)
        self.lh_roi_face_indx = np.all(np.isin(self.lh_faces, self.lh_roi_vert_IDs), axis=1)
        self.rh_roi_face_indx = np.all(np.isin(self.rh_faces, self.rh_roi_vert_IDs), axis=1)
        self.lh_roi_faces = self.lh_faces[self.lh_roi_face_indx,:]
        self.rh_roi_faces = self.rh_faces[self.rh_roi_face_indx,:]

        # Re-assign vertex IDs of the faces subset, so that the vertex IDs of each face match 
        # the location of nodes in shortened node array. **adding np.sort last-minute--need to check if this is okay.
        # (getting error at np.digitize saying ("bins must be monotonically increasing or decreasing"))
        self.lh_roi_faces_reassigned = np.digitize(self.lh_roi_faces, np.sort(self.lh_roi_vert_IDs), right=True)
        self.rh_roi_faces_reassigned = np.digitize(self.rh_roi_faces, np.sort(self.rh_roi_vert_IDs), right=True)









#### NOW to get vertices back into regular space,
# you could use the the 4th column of subset_verts, and the older faces assignments.

# # Behavior
# attr_fname = os.path.join(SUBJECT_DIR, 'behavior/attributes.txt')
# dtypes = {'names': ('Hts','TRs','labels','runs','cumTRs'),
#           'formats': (np.int, np.int, np.int, np.int, np.int)}
# attributes = np.loadtxt(attr_fname, delimiter=',', skiprows=1, dtype=dtypes)
# labels = attributes['labels']
# runs = attributes['runs']


#============================
# Call searchlight function.
#============================

# results = searchlight.surf(nodes=sub_nodes, faces=sub_faces,
#                             node_activity=node_activity,
#                             sphere_radius=source_radius,
#                             searchlight_radius=searchlight_radius,
#                             runs=runs, labels=labels,
#                             clf=clf, n_perms=n_perms,
#                             )










'''
REQUIRED ANATOMICAL INFO (for single hemisphere)

surface file (e.g., lh.white) :
    DIMS - [n_surface_nodes, 3] (for x,y,z coords), and [n_faces, 3] (for 3 node IDs that make face)
    This is the actual "mesh surface" that we evaluate. It holds information about 
    node locations and nodes of each face in a single tuple. This file is derived
    in freesurfer and used for almost everything here.
    [ use nib.freesurfer.read_geometry() to get into python ]  

cortex label (i.e., *h.cortex.label) :
    DIMS - [n_cortical_nodes]
    This is a list of indices of all nodes that are within the subject's cortex.
    When freesurfer constructs the mesh, it includes the medial wall, which should be
    removed from the surface before doing much else. This file allows us to do that.
    [ use nib.freesurfer.io.read_label() to get into python ]  

node activation values :
    DIMS - [n_surface_nodes, n_TRs]
    This is a typical machine learning array where labels are rows and
    features are columns. This includes all surface vertices, and the 
    searchlight will reduce the feature space to a small subset each iteration.
    This comes from a "surface-encoded volume file", one with .mgh extension.
    This file is returned from the mri_vol2surf command.
    This should be one array of all runs concatenated.
    [ use nib.freesurfer.mghformat.load() to get into python ] 


OPTIONAL INFO

sulcus curvature file :
    DIMS - [n_surface_nodes]
    This provides a value of curvature for each node. It is only to use for
    plotting, as it provides shading for the figure.
'''