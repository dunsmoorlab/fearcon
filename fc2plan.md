# FearCon pt. II Research Plan

## Summary of behavioral results
- We replicated findings presented in Dunsmoor et al., 2015 _Cortex_ 
- This study additionally compares Healthy vs. PTSD
    - The CS+ retroactive memory enhancement is for baseline items is __stronger__ in PTSD.
    - The CS+ proactive memory enhancement for extinction memories is __weaker__ in PTSD.
* HOWEVER, there is neither a main effect nor interaction of `Group`

![img](../graphing/behavior/group_mem_final.png)

#### Behavioral reseults concerns
- Memory effects were strongest when considering only _high confidence_ corrected recognition
- 2 control and 1 PTSD subjects are exluded due to low corrected recognition

## Hypotheses
The behavioral results suggest that there are differences in emotional memory processing between healthy controls and PTSD participants following aversive learning during memory retrieval. The goal is to show this using neuroimaging data.

Specifically, behavior suggests a different spread of emotional memory enhancement effect following conditioning. In controls, the emotional enhancement has a stronger effect on subsequently encoded items, while in PTSD the enhacement is stronger on previously encoded neutral items. I think this may represent both increased generalization and a failure of new learning (extinction) in the PTSD group.
One way to show this neurally is using representational similarity analysis.

I also think that we might be able to show differences in functional connectivity in the Control group between conditioning and extinction, and demonstrate that the same patterns of connectivity are reinstated the following day when category exemplars are reincountered. I think that patterns of connectivity will be different across Group, especially when it comes to encoding and retrieving extinction memories. PTSD group has obvious neural deficits in retrieving extinction memories demonstrated in FearCon pt. I, we should also be able to show that in some capacity here. Preliminary results looking at item-level encoding-retrieval overlap shows that controls have increased representational similarity for extinction CS+ items in the left dlPFC. Based on literature this could imply healthy adults are representing new saftey learning in areas associated with top-down cognitive control moreso than PTSD participants. 

Based on the behavior alone, we can predict:

1. Controls will have higher neural similarity between conditioning and extinction CS+ items during the memory test compared to PTSD
2. PTSD will have higher neural similarity between baseline and conditioning CS+ items compares to controls
3. Examining CS+ vs. CS- neural similarity will reveal an interaction of `Encoding Phase` and `Group` in regions important for encoding & retrieval of associative conditioning/extinction memories, as well as in sensory cortex (category learning generalization)

## Analyses
1. measure representational change of the CS+/CS-. The behavior this neural analysis is based on is the success of the category conditioning paradigm.
    - replicating seperation in animal/tool cortex would be a good place to start
        + use localizer category vs. scrambled images to identify category cortex
        + serves as a sanity check, AND we can compare the results across group
    -  Compare how the representational difference between CS+/CS- at the time of encoding differs from the representations elicited during the episodic memory test
        +  Can run a searchlight analysis where in each sphere, we compute the CS+ to CS- RSM for both encoding during conditioning and retrieval during memory test
        +  Then compute a spearmans rank correlation between the two RSMs, in order to determine find consolidation dependent effects on neural representations
    -  An interesting follow up would compare if any consolidation effects of aversive learning on CS+/CS- neural representations is selective only to remembered category exemplars or not
        +  Compare remembered items vs. forgotten items (maybe high confidence hits vs low confidence hits and misses?)
        +  Compare old items vs. new items
        +  again we have the added benefit of being able to compare across group
2. Replicate and expand results presented in Ritchey _et al._, 2013, but instead of enherently emotional items use items that have underwent aversive learning.
    - investigate both set level and item level RSA, breaking it up by CS, encoding, and memory
3. Functional connectivity analysis
    - During conditioning/extinction learning to show different patterns within group across phase and across group during extinction
    - During the retrieval test to determine if the same patterns of connectivity are reinstated when viewing previously encountered items, related items from the same time period, or novel items from the same category
    - There are different ways to go about this analysis
        + low frequency connectivity analysis (Tambini _et al._, 2017)
        + seed based / task based functional connectivity (Ritchey _et al._, 2013)
        + Full correlation Matrix Analysis (__FCMA__; [Example](https://brainiak.org/tutorials/09-fcma/))
            * I really like the idea of applying this approach here, because the question we are trying to ask is if there are different patterns of activity between learning phases and group
            * this analysis allows for connectivity based classification that might be sensitive to differences between conditioning/extinction where the normal MVPA classifier I previously tried was not


## Review of relevant literature
#### Dunsmoor et al., 2014 _Cerebral Cortex_
[Averisve Learning Modulates Cortical Representations of Object Cortex](https://academic.oup.com/cercor/article/24/11/2859/297931)

__Results__

- Analyses include RSA and PPI following category conditioning.
- Found increased overall neural activity in CS+ related visual cortex
- increased CS+ vs CS- representational dissimilarity in respective category cortices

__Notes__

- Excluded voxels with mean CS+ > CS- activity in RSA to reduce SNR

#### Ritchey et al., 2013 _Cerebral Cortex_
[Neural Similarity Between Encoding and Retrieval is Related to Memory via Hippocampal Interactions](https://academic.oup.com/cercor/article/23/12/2818/464061)

__Methods__


- viewed emotionally negative, positive, or neutral images, 24hr delayed recognition memory test
- computed RSA on item level, or grouped by valence, task (E vs. R), and memory (R vs. F) across items
    + set level groupings _exclude_ item pairs (no diagonal)
- __trial betas were Z-scored according to mean activity within a trial, something that I'm not doing (yet)__
- logistic regression analysis - trial mean ROI activation during either encoding retrieval or with pattern similarity to to predict binary memory outcome
    + significant non-zero coefficient implies mean activation estiamtes and encoding retrieval similarity make sepreateble contributions to memory
    + restricted subsequent memory analyses to regions whihc ER similarity reminaed significant predictor of memory
- also ran a mediation analysis testing if HC activity mediates relationship between ER similarity and 5 choice memory responses

__Results__

- In ROIs with main effect of match level also looked at effect of emotion
    - middle occipital gyrus showed strongest match X emotion interaction
- Found lots of ROIs that vary representations based on _match_ , i.e. (item or set level)
- Also found lots of ROIs that vary representations based on _memory_, regardless of match
- Interaction Match X Memory mostly in MTL and VTC
- Hippocampal mediation analysis found sig. mediation of ER similarity and memory in inferior frontal, inferior parietal, and occipital ROIs
- MOG ER similarity correlated with amygdala activity for negative _remembered_ items more than negaitve _forgotten_ items
- In the end though, HC was more important than Amygdala in predicting memory, suggesting amygdala responses are driven by "recovery and porcessing of item-specific detials"

#### Tambini et al., 2017 _Nature Neuroscience_
[Emotional brain states carry over and enhance future memory formation](https://www.nature.com/articles/nn.4468)

Then relevence here is in the methods, as it might not be fair to compare the emotional --> neutral viewing group the same as our conditioning --> extinction.

__Methods__

- low frequency connectivity of amygdala and anterior hippocampus
    + filter out task frequency to essentially look at "resting state"
- RSA Approach included permutation testing

#### Ritchey et al., 2018 _Neuropsychologia_
[Dissociable medial temporal pathways for encoding emotional item and context information](https://www.sciencedirect.com/science/article/pii/S0028393218307826)

#### Wing, et al., 2015 _Journal of Cognitive Neuroscience_
[Reinstatement of Individual Past Events Revealed by the Similarity of Distributed Activation Patterns during Encoding and Retrieval](https://www.mitpressjournals.org/doi/full/10.1162/jocn_a_00740?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%3dpubmed)















