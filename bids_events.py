from fc_config import *
from glm_timing import *
from preprocess_library import meta
bids_dir = '/Users/ach3377/Desktop/fc-bids'

for sub in all_sub_args:
    EVENTS = {}
    subj = meta(sub)
    for phase in ['memory_run_1','memory_run_2','memory_run_3']:
        e = glm_timing(sub,phase)
        events = e.mem_events(stims=True)
        events['response_time'] = (e.phase_meta['oldnew.RT'].values)/1000
        events.response_time[events.response == 'NR'] = 'n/a'
        events = events.rename(columns={'memcond':'memory_condition',
                                        'encode':'encode_phase',
                                        'acc':'low_confidence_accuracy',
                                        'hc_acc':'high_confidence_accuracy',
                                        'stim':'stimulus'})
        events.response = events.response.apply(lambda x: 'n/a' if x == 'NR' else x)
        events.encode_phase = events.encode_phase.apply(lambda x: 'acquisition' if x=='fear_conditioning' else x)
        EVENTS[phase] = events
    MEM = pd.concat(EVENTS.values()).reset_index()

    for phase in ['baseline','fear_conditioning','extinction']:
        e = glm_timing(sub,phase)
        if phase == 'fear_conditioning':
            events = e.phase_events(stims=True,shock=True)
            events['shock'] = e.proc.values
        else:                           
            events = e.phase_events(stims=True)  
        
        #collect response   
        events['response'] = e.phase_meta['stim.RESP'].values.astype(int)
        # if phase == 'baseline': events.response = events.response.apply(lambda x: 'animal' if x==1 else 'tool' if x==2 else 'NR')
        # else:                   events.response = events.response.apply(lambda x: 'expect' if x==1 else 'do_not_expect' if x==2 else 'NR')
        events.response = events.response.apply(lambda x: 'n/a' if x not in [1,2] else x) 
        events['response_time'] = (e.phase_meta['stim.RT'].values)/1000

        events = events.rename(columns={'stim':'stimulus'})
        events['recognition_memory'] = ''
        for i, stim in enumerate(events.stimulus):
            events.loc[i,'recognition_memory'] = MEM.loc[np.where(MEM.stimulus == stim)[0][0],'response']
        
        EVENTS[phase] = events
    
    for phase in ['extinction_recall']:
        e = glm_timing(sub,phase)
        events = e.phase_events(stims=True)
        events['response'] = e.phase_meta['stim.RESP'].values.astype(int)
        # events.response = events.response.apply(lambda x: 'expect' if x==1 else 'do_not_expect' if x==2 else 'NR')
        events = events.rename(columns={'stim':'stimulus'})
        EVENTS[phase] = events

    for phase in ['localizer_1','localizer_2']:
        if sub == 107 and '2' in phase: pass
        else:
            e = glm_timing(sub,phase)
            events = e.loc_blocks()
            events = events[['onset','duration','trial_type']]
        EVENTS[phase] = events

    for i, phase in enumerate(['baseline','fear_conditioning','extinction','extinction_recall','memory_run_1','memory_run_2','memory_run_3','localizer_1','localizer_2']):
        substr = 'sub-FC%s'%(subj.fsub[-3:])
        if i <= 2:
            outdir = os.path.join(bids_dir,substr,'ses-1','func')
            if phase == 'fear_conditioning': outstr = os.path.join(outdir,substr+'_ses-1_task-acquisition_events.tsv')
            else: outstr = os.path.join(outdir,substr+'_ses-1_task-'+phase+'_events.tsv')
        elif i > 2:
            outdir = os.path.join(bids_dir,substr,'ses-2','func')
            if phase == 'extinction_recall': outstr = os.path.join(outdir,substr+'_ses-2_task-renewal_events.tsv')
            elif 'memory' in phase:          outstr = os.path.join(outdir,substr+'_ses-2_task-memory_run-0%s_events.tsv'%(phase[-1:]))
            elif 'localizer' in phase:       outstr = os.path.join(outdir,substr+'_ses-2_task-localizer_run-0%s_events.tsv'%(phase[-1:]))
        EVENTS[phase].to_csv(outstr,sep='\t',na_rep='n/a',index=False)
