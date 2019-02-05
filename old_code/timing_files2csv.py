from glm_timing import glm_timing
import os

subject = 1

phases = ['baseline','fear_conditioning','extinction','localizer_1','localizer_2']

output_dir = os.path.join('/Users/ach3377/Desktop', 'timing_files')
if not os.path.exists(output_dir):
	os.mkdir(output_dir)

for phase in phases:

	glm_timing(subject, phase).phase_events().to_csv(
		os.path.join(output_dir, '%s_timing_file.csv'%(phase)))