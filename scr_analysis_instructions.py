'''
how to generate files for Autonomate batch analysis 

file_names = glob(files)

batch_file = open('batch_file.txt.','w')

{batch_file.write('%s\n'%(file)) for file in file_names}

batch_file.close ?

'''
