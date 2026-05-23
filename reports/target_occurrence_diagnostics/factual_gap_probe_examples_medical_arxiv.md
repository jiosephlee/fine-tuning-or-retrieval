# Factual Probe Source-vs-Explanation Gap Examples

This report is generated from `factual_7b_source_vs_explanations_target_occurrence_probe_level.csv`.

# Medical factual probe gap examples

Selected the largest 20 7B factual cloze gaps in each direction.
`source_minus_explanations` is final-step `source_only` log-prob minus final-step `with_explanations` log-prob.
Occurrence columns are normalized exact full-target phrase appearances per 1,000 words.

## Source > explanations

### 1. `Management_of_atypical_complicated_abdominal_masses_in_the_s` probe `107`

- `source_log_prob`: -27.818
- `explanations_log_prob`: -60.241
- `source_minus_explanations`: 32.423
- `target_words`: 17
- `source_occ_per_1k`: 0.403
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", gadolinium-enhanced MRI is generally not recommended during pregnancy,
```

**Target**

```text
unless strictly necessary, due to rheumatological, inflammatory or infiltrative skin conditions, risk of stillbirth or neonatal death.
```

**Fact**

```text
According to the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", gadolinium-enhanced MRI is generally not recommended during pregnancy, unless strictly necessary, due to rheumatological, inflammatory or infiltrative skin conditions, risk of stillbirth or neonatal death.
```

### 2. `Monoallelic_PARN_mutation_presenting_as_pancytopenia_hepatic` probe `144`

- `source_log_prob`: -16.001
- `explanations_log_prob`: -40.406
- `source_minus_explanations`: 24.405
- `target_words`: 13
- `source_occ_per_1k`: 0.424
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the case report "Monoallelic PARN mutation presenting as pancytopenia, hepatic fibrosis and idiopathic pulmonary fibrosis", the authors state that management of telomere biology disorders is currently
```

**Target**

```text
supportive and tailored to symptom severity, organ transplantation needs and risk factor mitigation
```

**Fact**

```text
In the case report "Monoallelic PARN mutation presenting as pancytopenia, hepatic fibrosis and idiopathic pulmonary fibrosis", the authors state that management of telomere biology disorders is currently supportive and tailored to symptom severity, organ transplantation needs and risk factor mitigation
```

### 3. `Multiphasic_anaphylaxis_in_the_emergency_and_intensive_care` probe `105`

- `source_log_prob`: -42.677
- `explanations_log_prob`: -64.438
- `source_minus_explanations`: 21.760
- `target_words`: 13
- `source_occ_per_1k`: 0.403
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the case report "Multiphasic anaphylaxis in the emergency and intensive care setting", skin testing was not done after the patient’s multiphasic anaphylactic reaction to clarithromycin because
```

**Target**

```text
risking renewed serious anaphylaxis through, for example, skin testing was not deemed proportional
```

**Fact**

```text
In the case report "Multiphasic anaphylaxis in the emergency and intensive care setting", skin testing was not done after the patient’s multiphasic anaphylactic reaction to clarithromycin because risking renewed serious anaphylaxis through, for example, skin testing was not deemed proportional
```

### 4. `TAVinTAVinTAV_after_treated_endocarditis_procedural_strategy` probe `44`

- `source_log_prob`: -5.753
- `explanations_log_prob`: -26.273
- `source_minus_explanations`: 20.520
- `target_words`: 8
- `source_occ_per_1k`: 0.400
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "TAV-in-TAV-in-TAV after treated endocarditis: procedural strategy, imaging planning and 1-year outcome", transcatheter aortic valve implantation (TAVI) is described as an established treatment option for patients with severe aortic stenosis
```

**Target**

```text
across a broad range of surgical risk profiles.
```

**Fact**

```text
According to the case report "TAV-in-TAV-in-TAV after treated endocarditis: procedural strategy, imaging planning and 1-year outcome", transcatheter aortic valve implantation (TAVI) is described as an established treatment option for patients with severe aortic stenosis across a broad range of surgical risk profiles.
```

### 5. `Monoallelic_PARN_mutation_presenting_as_pancytopenia_hepatic` probe `118`

- `source_log_prob`: -9.924
- `explanations_log_prob`: -30.087
- `source_minus_explanations`: 20.164
- `target_words`: 13
- `source_occ_per_1k`: 0.424
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Monoallelic PARN mutation presenting as pancytopenia, hepatic fibrosis and idiopathic pulmonary fibrosis", deep sequencing of the 3-prime terminal of TERC RNA was used to describe PARN’s role in TERC RNA processing, and PARN is required for
```

**Target**

```text
the removal of post-transcriptionally acquired oligo(A) tails that target nuclear RNAs for degradation
```

**Fact**

```text
According to the case report "Monoallelic PARN mutation presenting as pancytopenia, hepatic fibrosis and idiopathic pulmonary fibrosis", deep sequencing of the 3-prime terminal of TERC RNA was used to describe PARN’s role in TERC RNA processing, and PARN is required for the removal of post-transcriptionally acquired oligo(A) tails that target nuclear RNAs for degradation
```

### 6. `Multiphasic_anaphylaxis_in_the_emergency_and_intensive_care` probe `67`

- `source_log_prob`: -35.709
- `explanations_log_prob`: -54.599
- `source_minus_explanations`: 18.890
- `target_words`: 16
- `source_occ_per_1k`: 0.403
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the case report "Multiphasic anaphylaxis in the emergency and intensive care setting", a woman in her 60s presented to the emergency department with suspected severe allergic reaction after taking oral clarithromycin, and as part of the initial treatment for her skin reaction, she was given
```

**Target**

```text
slow intravenous injection of 2 mg clemastine, for treatment of the skin reaction following local guidelines.
```

**Fact**

```text
In the case report "Multiphasic anaphylaxis in the emergency and intensive care setting", a woman in her 60s presented to the emergency department with suspected severe allergic reaction after taking oral clarithromycin, and as part of the initial treatment for her skin reaction, she was given slow intravenous injection of 2 mg clemastine, for treatment of the skin reaction following local guidelines.
```

### 7. `Central_hypoventilation_and_choking_episodes_revealing_Chiar` probe `118`

- `source_log_prob`: -7.658
- `explanations_log_prob`: -25.781
- `source_minus_explanations`: 18.123
- `target_words`: 15
- `source_occ_per_1k`: 0.387
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", the authors propose a brainstem pathology to explain the patient's lack of dyspnea during episodes of severe oxygen desaturation, which likely interrupts
```

**Target**

```text
the critical afferent neural pathways that must ascend to the cortex to produce conscious sensation.
```

**Fact**

```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", the authors propose a brainstem pathology to explain the patient's lack of dyspnea during episodes of severe oxygen desaturation, which likely interrupts the critical afferent neural pathways that must ascend to the cortex to produce conscious sensation.
```

### 8. `Multiphasic_anaphylaxis_in_the_emergency_and_intensive_care` probe `42`

- `source_log_prob`: -16.628
- `explanations_log_prob`: -34.272
- `source_minus_explanations`: 17.644
- `target_words`: 9
- `source_occ_per_1k`: 0.403
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Multiphasic anaphylaxis in the emergency and intensive care setting," anaphylaxis is described in the background section as
```

**Target**

```text
a severe and—especially in anaphylactic shock—potentially life-threatening allergic reaction
```

**Fact**

```text
According to the case report "Multiphasic anaphylaxis in the emergency and intensive care setting," anaphylaxis is described in the background section as a severe and—especially in anaphylactic shock—potentially life-threatening allergic reaction
```

### 9. `Management_of_atypical_complicated_abdominal_masses_in_the_s` probe `68`

- `source_log_prob`: -25.920
- `explanations_log_prob`: -42.857
- `source_minus_explanations`: 16.937
- `target_words`: 13
- `source_occ_per_1k`: 0.403
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", at 21 weeks plus 6 days of gestation, after her abdominal pain and dyspnoea worsened and there was concern for complications such as rupture, torsion, and infection, the patient underwent
```

**Target**

```text
laparotomy with a longitudinal xiphos-pubic cut at 21 w+6 d after informed consent.
```

**Fact**

```text
According to the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", at 21 weeks plus 6 days of gestation, after her abdominal pain and dyspnoea worsened and there was concern for complications such as rupture, torsion, and infection, the patient underwent laparotomy with a longitudinal xiphos-pubic cut at 21 w+6 d after informed consent.
```

### 10. `Multiphasic_anaphylaxis_in_the_emergency_and_intensive_care` probe `90`

- `source_log_prob`: -10.409
- `explanations_log_prob`: -26.421
- `source_minus_explanations`: 16.013
- `target_words`: 6
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Multiphasic anaphylaxis in the emergency and intensive care setting", after hospitalization for multiphasic anaphylaxis to clarithromycin, the patient was started on
```

**Target**

```text
levocetirizine as a new standard medication.
```

**Fact**

```text
According to the case report "Multiphasic anaphylaxis in the emergency and intensive care setting", after hospitalization for multiphasic anaphylaxis to clarithromycin, the patient was started on levocetirizine as a new standard medication.
```

### 11. `Management_of_atypical_complicated_abdominal_masses_in_the_s` probe `88`

- `source_log_prob`: -13.010
- `explanations_log_prob`: -28.732
- `source_minus_explanations`: 15.722
- `target_words`: 14
- `source_occ_per_1k`: 0.403
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", the authors state that the caudal myoma would have represented
```

**Target**

```text
a huge obstacle for the fetal head progression during the expulsive phase of labour.
```

**Fact**

```text
In the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", the authors state that the caudal myoma would have represented a huge obstacle for the fetal head progression during the expulsive phase of labour.
```

### 12. `Pancreatopleural_fistula_in_childhood` probe `102`

- `source_log_prob`: -24.403
- `explanations_log_prob`: -39.922
- `source_minus_explanations`: 15.518
- `target_words`: 10
- `source_occ_per_1k`: 0.361
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the case report "Pancreatopleural fistula in childhood", after the child developed pancreatopleural fistula with recurrent blood-stained pleural effusions and conservative management was chosen first, he was kept nil per oral and was started on
```

**Target**

```text
Octreotide infusion in the following dose 30 µg/kg/day (10–40 µg/kg/day).
```

**Fact**

```text
In the case report "Pancreatopleural fistula in childhood", after the child developed pancreatopleural fistula with recurrent blood-stained pleural effusions and conservative management was chosen first, he was kept nil per oral and was started on Octreotide infusion in the following dose 30 µg/kg/day (10–40 µg/kg/day).
```

### 13. `Monoallelic_PARN_mutation_presenting_as_pancytopenia_hepatic` probe `48`

- `source_log_prob`: -9.258
- `explanations_log_prob`: -24.699
- `source_minus_explanations`: 15.441
- `target_words`: 7
- `source_occ_per_1k`: 0.424
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the case report "Monoallelic PARN mutation presenting as pancytopenia, hepatic fibrosis and idiopathic pulmonary fibrosis", telomere biology disorders (TBDs), described in the background as rare premature-ageing syndromes related to abnormal telomeres, are
```

**Target**

```text
caused by Mendelian defects in telomere-related genes
```

**Fact**

```text
In the case report "Monoallelic PARN mutation presenting as pancytopenia, hepatic fibrosis and idiopathic pulmonary fibrosis", telomere biology disorders (TBDs), described in the background as rare premature-ageing syndromes related to abnormal telomeres, are caused by Mendelian defects in telomere-related genes
```

### 14. `Dermatomyositis_masquerading_as_angioedema_a_crucial_differe` probe `28`

- `source_log_prob`: -1.261
- `explanations_log_prob`: -16.347
- `source_minus_explanations`: 15.086
- `target_words`: 5
- `source_occ_per_1k`: 0.432
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Dermatomyositis masquerading as angioedema: a crucial differential not to overlook", dermatomyositis is described as a rare autoimmune disease affecting the skin, muscle, and respiratory system, and prompt treatment with intense immunosuppression aims to reduce
```

**Target**

```text
disease progression and multiorgan damage.
```

**Fact**

```text
According to the case report "Dermatomyositis masquerading as angioedema: a crucial differential not to overlook", dermatomyositis is described as a rare autoimmune disease affecting the skin, muscle, and respiratory system, and prompt treatment with intense immunosuppression aims to reduce disease progression and multiorgan damage.
```

### 15. `Management_of_atypical_complicated_abdominal_masses_in_the_s` probe `120`

- `source_log_prob`: -23.010
- `explanations_log_prob`: -37.528
- `source_minus_explanations`: 14.517
- `target_words`: 17
- `source_occ_per_1k`: 0.403
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", before the surgical approach for uterine myomas, the cited papers showed the importance of
```

**Target**

```text
an accurate US examination of the size, location, number and distance of uterine myomas from the placenta
```

**Fact**

```text
According to the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", before the surgical approach for uterine myomas, the cited papers showed the importance of an accurate US examination of the size, location, number and distance of uterine myomas from the placenta
```

### 16. `Central_hypoventilation_and_choking_episodes_revealing_Chiar` probe `53`

- `source_log_prob`: -1.925
- `explanations_log_prob`: -16.079
- `source_minus_explanations`: 14.155
- `target_words`: 9
- `source_occ_per_1k`: 0.387
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", the pathophysiology of Chiari malformation type I is described as involving a developmentally small posterior fossa, creating
```

**Target**

```text
a volumetric mismatch between neural structures and available space
```

**Fact**

```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", the pathophysiology of Chiari malformation type I is described as involving a developmentally small posterior fossa, creating a volumetric mismatch between neural structures and available space
```

### 17. `Autoimmune_heparininduced_thrombocytopenia_following_cardiac` probe `35`

- `source_log_prob`: -10.756
- `explanations_log_prob`: -23.975
- `source_minus_explanations`: 13.219
- `target_words`: 9
- `source_occ_per_1k`: 0.450
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", the surgery proceeded uneventfully,
```

**Target**

```text
without major blood losses or need for blood transfusion
```

**Fact**

```text
In the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", the surgery proceeded uneventfully, without major blood losses or need for blood transfusion
```

### 18. `Monoallelic_PARN_mutation_presenting_as_pancytopenia_hepatic` probe `105`

- `source_log_prob`: -4.105
- `explanations_log_prob`: -17.286
- `source_minus_explanations`: 13.181
- `target_words`: 8
- `source_occ_per_1k`: 0.424
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Monoallelic PARN mutation presenting as pancytopenia, hepatic fibrosis and idiopathic pulmonary fibrosis", the discussion notes that telomeres are repetitive DNA sequences capping chromosome ends and that they
```

**Target**

```text
preserve genomic integrity by shielding chromosomes from degradation.
```

**Fact**

```text
According to the case report "Monoallelic PARN mutation presenting as pancytopenia, hepatic fibrosis and idiopathic pulmonary fibrosis", the discussion notes that telomeres are repetitive DNA sequences capping chromosome ends and that they preserve genomic integrity by shielding chromosomes from degradation.
```

### 19. `Management_of_atypical_complicated_abdominal_masses_in_the_s` probe `58`

- `source_log_prob`: -7.973
- `explanations_log_prob`: -20.648
- `source_minus_explanations`: 12.675
- `target_words`: 6
- `source_occ_per_1k`: 0.403
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", after ultrasound showed two large abdominal masses, the pregnant patient underwent a non-contrast MRI that described thick walls and inhomogeneous fluid content with
```

**Target**

```text
focal hyperintense T1-T2 weighted solid components
```

**Fact**

```text
According to the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", after ultrasound showed two large abdominal masses, the pregnant patient underwent a non-contrast MRI that described thick walls and inhomogeneous fluid content with focal hyperintense T1-T2 weighted solid components
```

### 20. `Management_of_atypical_complicated_abdominal_masses_in_the_s` probe `59`

- `source_log_prob`: -28.681
- `explanations_log_prob`: -41.310
- `source_minus_explanations`: 12.629
- `target_words`: 12
- `source_occ_per_1k`: 0.403
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", after ultrasound had shown a lower pelvic mass and a uterine myoma with cystic degeneration and a non-contrast MRI was performed, the cleavage of the caudal mass was not clear and the caudal mass seemed to be in continuity with
```

**Target**

```text
the posterior leiomyoma that had already been examined at the US evaluation
```

**Fact**

```text
In the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", after ultrasound had shown a lower pelvic mass and a uterine myoma with cystic degeneration and a non-contrast MRI was performed, the cleavage of the caudal mass was not clear and the caudal mass seemed to be in continuity with the posterior leiomyoma that had already been examined at the US evaluation
```

## Explanations > source

### 1. `Dermatomyositis_masquerading_as_angioedema_a_crucial_differe` probe `127`

- `source_log_prob`: -61.391
- `explanations_log_prob`: -28.157
- `source_minus_explanations`: -33.234
- `target_words`: 5
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Dermatomyositis masquerading as angioedema: a crucial differential not to overlook", in the International Guideline for Idiopathic Inflammatory Myopathy-Associated Cancer Screening (IMACS) assessment, the patient had two high-risk factors (dermatomyositis and dysphagia), and the two intermediate-risk factors were
```

**Target**

```text
SAE1 antibody positive, male sex.
```

**Fact**

```text
According to the case report "Dermatomyositis masquerading as angioedema: a crucial differential not to overlook", in the International Guideline for Idiopathic Inflammatory Myopathy-Associated Cancer Screening (IMACS) assessment, the patient had two high-risk factors (dermatomyositis and dysphagia), and the two intermediate-risk factors were SAE1 antibody positive, male sex.
```

### 2. `Dermatomyositis_masquerading_as_angioedema_a_crucial_differe` probe `25`

- `source_log_prob`: -37.146
- `explanations_log_prob`: -4.776
- `source_minus_explanations`: -32.370
- `target_words`: 2
- `source_occ_per_1k`: 0.432
- `para9_occ_per_1k`: 0.174
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Dermatomyositis masquerading as angioedema: a crucial differential not to overlook", the authors note that there is limited evidence to guide optimal pharmacotherapy for
```

**Target**

```text
anti-SAE1–positive dermatomyositis.
```

**Fact**

```text
According to the case report "Dermatomyositis masquerading as angioedema: a crucial differential not to overlook", the authors note that there is limited evidence to guide optimal pharmacotherapy for anti-SAE1–positive dermatomyositis.
```

### 3. `Dermatomyositis_masquerading_as_angioedema_a_crucial_differe` probe `114`

- `source_log_prob`: -34.045
- `explanations_log_prob`: -5.417
- `source_minus_explanations`: -28.628
- `target_words`: 3
- `source_occ_per_1k`: 0.432
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.056

**Cloze**

```text
According to the case report "Dermatomyositis masquerading as angioedema: a crucial differential not to overlook", the autoimmune condition ultimately reported in the patient who presented with prominent cervical soft tissue swelling that was initially mistaken for idiopathic angioedema was
```

**Target**

```text
anti-SAE1 antibody–positive dermatomyositis
```

**Fact**

```text
According to the case report "Dermatomyositis masquerading as angioedema: a crucial differential not to overlook", the autoimmune condition ultimately reported in the patient who presented with prominent cervical soft tissue swelling that was initially mistaken for idiopathic angioedema was anti-SAE1 antibody–positive dermatomyositis
```

### 4. `Dermatomyositis_masquerading_as_angioedema_a_crucial_differe` probe `81`

- `source_log_prob`: -49.726
- `explanations_log_prob`: -22.551
- `source_minus_explanations`: -27.175
- `target_words`: 7
- `source_occ_per_1k`: 0.432
- `para9_occ_per_1k`: 0.043
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Dermatomyositis masquerading as angioedema: a crucial differential not to overlook", a previously well man in his 40s was ultimately diagnosed after evaluation for inflammatory neck swelling, dysphagia, and myositis with
```

**Target**

```text
moderate to severe SAE1 antibody positive dermatomyositis
```

**Fact**

```text
According to the case report "Dermatomyositis masquerading as angioedema: a crucial differential not to overlook", a previously well man in his 40s was ultimately diagnosed after evaluation for inflammatory neck swelling, dysphagia, and myositis with moderate to severe SAE1 antibody positive dermatomyositis
```

### 5. `Central_hypoventilation_and_choking_episodes_revealing_Chiar` probe `128`

- `source_log_prob`: -43.529
- `explanations_log_prob`: -20.903
- `source_minus_explanations`: -22.626
- `target_words`: 12
- `source_occ_per_1k`: 0.387
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", after the earlier sleep evaluation had limitations, the study arranged at an external sleep centre for the patient was
```

**Target**

```text
Type I polysomnography with full EEG monitoring, end-tidal capnography and video recording
```

**Fact**

```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", after the earlier sleep evaluation had limitations, the study arranged at an external sleep centre for the patient was Type I polysomnography with full EEG monitoring, end-tidal capnography and video recording
```

### 6. `Autoimmune_heparininduced_thrombocytopenia_following_cardiac` probe `61`

- `source_log_prob`: -60.880
- `explanations_log_prob`: -40.977
- `source_minus_explanations`: -19.903
- `target_words`: 8
- `source_occ_per_1k`: 0.450
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", the Fc receptor-blocking monoclonal antibody IV.3 failed to fully inhibit serotonin release at 0.1 U/mL UFH, explaining
```

**Target**

```text
the initial designation of an ‘indeterminate’ SRA result.
```

**Fact**

```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", the Fc receptor-blocking monoclonal antibody IV.3 failed to fully inhibit serotonin release at 0.1 U/mL UFH, explaining the initial designation of an ‘indeterminate’ SRA result.
```

### 7. `From_dog_bite_to_dialysis_complementmediated_haemolytic_urae` probe `132`

- `source_log_prob`: -39.564
- `explanations_log_prob`: -23.397
- `source_minus_explanations`: -16.167
- `target_words`: 5
- `source_occ_per_1k`: 0.435
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the case report "From dog bite to dialysis: complement-mediated haemolytic uraemic syndrome", the patient with dialysis-dependent acute kidney injury had a transient return of diuresis before eculizumab was started, and her diuresis returned
```

**Target**

```text
after the final PEX sessions
```

**Fact**

```text
In the case report "From dog bite to dialysis: complement-mediated haemolytic uraemic syndrome", the patient with dialysis-dependent acute kidney injury had a transient return of diuresis before eculizumab was started, and her diuresis returned after the final PEX sessions
```

### 8. `Management_of_atypical_complicated_abdominal_masses_in_the_s` probe `97`

- `source_log_prob`: -72.754
- `explanations_log_prob`: -58.623
- `source_minus_explanations`: -14.131
- `target_words`: 19
- `source_occ_per_1k`: 0.403
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", the patient’s initial ultrasound and MRI evaluation suggested that the abdominal masses were two huge decidualised endometriomas, and the caudal solid component of the pelvic mass seemed like
```

**Target**

```text
a posterior intramural-subserosal myoma of the uterus with cystic degeneration of 58×71 mm at both US and MRI imaging
```

**Fact**

```text
According to the case report "Management of atypical complicated abdominal masses in the second trimester of pregnancy", the patient’s initial ultrasound and MRI evaluation suggested that the abdominal masses were two huge decidualised endometriomas, and the caudal solid component of the pelvic mass seemed like a posterior intramural-subserosal myoma of the uterus with cystic degeneration of 58×71 mm at both US and MRI imaging
```

### 9. `TAVinTAVinTAV_after_treated_endocarditis_procedural_strategy` probe `125`

- `source_log_prob`: -18.778
- `explanations_log_prob`: -6.014
- `source_minus_explanations`: -12.764
- `target_words`: 5
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "TAV-in-TAV-in-TAV after treated endocarditis: procedural strategy, imaging planning and 1-year outcome", after preprocedural CT and 3D modelling showed feasibility and adequate coronary clearance, the third transcatheter valve the clinical team proceeded with was
```

**Target**

```text
a self-expanding Acurate neo2 prosthesis.
```

**Fact**

```text
According to the case report "TAV-in-TAV-in-TAV after treated endocarditis: procedural strategy, imaging planning and 1-year outcome", after preprocedural CT and 3D modelling showed feasibility and adequate coronary clearance, the third transcatheter valve the clinical team proceeded with was a self-expanding Acurate neo2 prosthesis.
```

### 10. `Acute_neurological_Behets_disease_complicated_by_a_syndrome` probe `24`

- `source_log_prob`: -28.269
- `explanations_log_prob`: -15.827
- `source_minus_explanations`: -12.441
- `target_words`: 5
- `source_occ_per_1k`: 0.390
- `para9_occ_per_1k`: 0.117
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Acute neurological Behçet's disease complicated by a syndrome of inappropriate secretion of antidiuretic hormone", this patient’s rare clinical scenario involved simultaneous diagnoses of
```

**Target**

```text
systemic BD and acute neuro-BD.
```

**Fact**

```text
According to the case report "Acute neurological Behçet's disease complicated by a syndrome of inappropriate secretion of antidiuretic hormone", this patient’s rare clinical scenario involved simultaneous diagnoses of systemic BD and acute neuro-BD.
```

### 11. `Monoallelic_PARN_mutation_presenting_as_pancytopenia_hepatic` probe `122`

- `source_log_prob`: -54.506
- `explanations_log_prob`: -42.372
- `source_minus_explanations`: -12.133
- `target_words`: 2
- `source_occ_per_1k`: 0.424
- `para9_occ_per_1k`: 0.357
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Monoallelic PARN mutation presenting as pancytopenia, hepatic fibrosis and idiopathic pulmonary fibrosis", monoallelic germline pathogenic variants have been implicated in
```

**Target**

```text
telomere-related PFBMFT4.
```

**Fact**

```text
According to the case report "Monoallelic PARN mutation presenting as pancytopenia, hepatic fibrosis and idiopathic pulmonary fibrosis", monoallelic germline pathogenic variants have been implicated in telomere-related PFBMFT4.
```

### 12. `From_dog_bite_to_dialysis_complementmediated_haemolytic_urae` probe `123`

- `source_log_prob`: -39.119
- `explanations_log_prob`: -27.067
- `source_minus_explanations`: -12.052
- `target_words`: 7
- `source_occ_per_1k`: 0.435
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "From dog bite to dialysis: complement-mediated haemolytic uraemic syndrome", the extended complement testing was performed
```

**Target**

```text
8 days after the final PEX session
```

**Fact**

```text
According to the case report "From dog bite to dialysis: complement-mediated haemolytic uraemic syndrome", the extended complement testing was performed 8 days after the final PEX session
```

### 13. `TAVinTAVinTAV_after_treated_endocarditis_procedural_strategy` probe `99`

- `source_log_prob`: -16.117
- `explanations_log_prob`: -4.288
- `source_minus_explanations`: -11.829
- `target_words`: 2
- `source_occ_per_1k`: 0.400
- `para9_occ_per_1k`: 0.359
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "TAV-in-TAV-in-TAV after treated endocarditis: procedural strategy, imaging planning and 1-year outcome", follow-up transthoracic echocardiography at 1 month after the third transcatheter aortic valve implantation showed marked haemodynamic improvement, and the aortic valve area (AVA) had increased to
```

**Target**

```text
1.1 cm².
```

**Fact**

```text
According to the case report "TAV-in-TAV-in-TAV after treated endocarditis: procedural strategy, imaging planning and 1-year outcome", follow-up transthoracic echocardiography at 1 month after the third transcatheter aortic valve implantation showed marked haemodynamic improvement, and the aortic valve area (AVA) had increased to 1.1 cm².
```

### 14. `Immunemediated_necrotising_myopathy_following_semaglutide_tr` probe `78`

- `source_log_prob`: -31.528
- `explanations_log_prob`: -19.764
- `source_minus_explanations`: -11.764
- `target_words`: 3
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Immune-mediated necrotising myopathy following semaglutide treatment: a contributing factor?", during evaluation for progressive proximal weakness, dysphagia, and elevated creatine kinase after a single semaglutide dose, the patient's myositis panel was negative for
```

**Target**

```text
SRP and HMGCR.
```

**Fact**

```text
According to the case report "Immune-mediated necrotising myopathy following semaglutide treatment: a contributing factor?", during evaluation for progressive proximal weakness, dysphagia, and elevated creatine kinase after a single semaglutide dose, the patient's myositis panel was negative for SRP and HMGCR.
```

### 15. `Autoimmune_heparininduced_thrombocytopenia_following_cardiac` probe `81`

- `source_log_prob`: -41.096
- `explanations_log_prob`: -29.352
- `source_minus_explanations`: -11.744
- `target_words`: 10
- `source_occ_per_1k`: 0.450
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", a last blood sample tested on POD124 after platelet count recovery showed a serotonin-release assay profile consistent with predominantly heparin-dependent HIT antibodies, with
```

**Target**

```text
100% release in the presence of UFH at 0.3 U/mL
```

**Fact**

```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", a last blood sample tested on POD124 after platelet count recovery showed a serotonin-release assay profile consistent with predominantly heparin-dependent HIT antibodies, with 100% release in the presence of UFH at 0.3 U/mL
```

### 16. `From_dog_bite_to_dialysis_complementmediated_haemolytic_urae` probe `131`

- `source_log_prob`: -44.286
- `explanations_log_prob`: -32.688
- `source_minus_explanations`: -11.598
- `target_words`: 8
- `source_occ_per_1k`: 0.435
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "From dog bite to dialysis: complement-mediated haemolytic uraemic syndrome", the absence of pathogenic or likely pathogenic complement gene variants in this patient supported
```

**Target**

```text
infection-triggered CM-HUS without a demonstrable inherited complement predisposition.
```

**Fact**

```text
According to the case report "From dog bite to dialysis: complement-mediated haemolytic uraemic syndrome", the absence of pathogenic or likely pathogenic complement gene variants in this patient supported infection-triggered CM-HUS without a demonstrable inherited complement predisposition.
```

### 17. `Immunemediated_necrotising_myopathy_following_semaglutide_tr` probe `29`

- `source_log_prob`: -25.905
- `explanations_log_prob`: -14.414
- `source_minus_explanations`: -11.492
- `target_words`: 7
- `source_occ_per_1k`: 0.400
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the case report "Immune-mediated necrotising myopathy following semaglutide treatment: a contributing factor?", the patient with the first reported case of immune-mediated necrotising myopathy probably associated with GLP-1 receptor agonist use appeared to respond to
```

**Target**

```text
a short course of immunosuppression and IVIG.
```

**Fact**

```text
In the case report "Immune-mediated necrotising myopathy following semaglutide treatment: a contributing factor?", the patient with the first reported case of immune-mediated necrotising myopathy probably associated with GLP-1 receptor agonist use appeared to respond to a short course of immunosuppression and IVIG.
```

### 18. `TAVinTAVinTAV_after_treated_endocarditis_procedural_strategy` probe `111`

- `source_log_prob`: -59.695
- `explanations_log_prob`: -48.532
- `source_minus_explanations`: -11.162
- `target_words`: 15
- `source_occ_per_1k`: 0.400
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "TAV-in-TAV-in-TAV after treated endocarditis: procedural strategy, imaging planning and 1-year outcome", after the patient’s prosthetic valve endocarditis had been successfully treated and no active infection was present, the team decided in 2024 to proceed with a third transcatheter aortic valve intervention because surgery was considered prohibitive. The planned strategy for this third transcatheter intervention was based on
```

**Target**

```text
CT-confirmed feasibility of a strategy aimed at preserving coronary flow and achieving acceptable haemodynamic performance
```

**Fact**

```text
According to the case report "TAV-in-TAV-in-TAV after treated endocarditis: procedural strategy, imaging planning and 1-year outcome", after the patient’s prosthetic valve endocarditis had been successfully treated and no active infection was present, the team decided in 2024 to proceed with a third transcatheter aortic valve intervention because surgery was considered prohibitive. The planned strategy for this third transcatheter intervention was based on CT-confirmed feasibility of a strategy aimed at preserving coronary flow and achieving ...
```

### 19. `TAVinTAVinTAV_after_treated_endocarditis_procedural_strategy` probe `35`

- `source_log_prob`: -37.175
- `explanations_log_prob`: -26.268
- `source_minus_explanations`: -10.907
- `target_words`: 8
- `source_occ_per_1k`: 0.400
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "TAV-in-TAV-in-TAV after treated endocarditis: procedural strategy, imaging planning and 1-year outcome", the authors state that meticulous procedural planning is especially important in patients with known coronary disease or in another group of patients, namely
```

**Target**

```text
those in whom future access may be essential.
```

**Fact**

```text
According to the case report "TAV-in-TAV-in-TAV after treated endocarditis: procedural strategy, imaging planning and 1-year outcome", the authors state that meticulous procedural planning is especially important in patients with known coronary disease or in another group of patients, namely those in whom future access may be essential.
```

### 20. `Autoimmune_heparininduced_thrombocytopenia_following_cardiac` probe `89`

- `source_log_prob`: -32.792
- `explanations_log_prob`: -22.030
- `source_minus_explanations`: -10.763
- `target_words`: 9
- `source_occ_per_1k`: 0.450
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", a brisk correction of thrombocytopenia usually occurs when
```

**Target**

```text
high-dose intravenous Ig is given to patients with aHIT
```

**Fact**

```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", a brisk correction of thrombocytopenia usually occurs when high-dose intravenous Ig is given to patients with aHIT
```


# Arxiv factual probe gap examples

Selected the largest 20 7B factual cloze gaps in each direction.
`source_minus_explanations` is final-step `source_only` log-prob minus final-step `with_explanations` log-prob.
Occurrence columns are normalized exact full-target phrase appearances per 1,000 words.

## Source > explanations

### 1. `FeatLLM` probe `296`

- `source_log_prob`: -17.575
- `explanations_log_prob`: -32.724
- `source_minus_explanations`: 15.148
- `target_words`: 8
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", labeling is challenging for practitioners in the real-world finance and healthcare scenarios discussed for \textsf{FeatLLM} because of
```

**Target**

```text
the high human cost or label collection difficulties.
```

**Fact**

```text
According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", labeling is challenging for practitioners in the real-world finance and healthcare scenarios discussed for \textsf{FeatLLM} because of the high human cost or label collection difficulties.
```

### 2. `ByteLatent` probe `349`

- `source_log_prob`: -31.329
- `explanations_log_prob`: -45.083
- `source_minus_explanations`: 13.753
- `target_words`: 12
- `source_occ_per_1k`: 0.115
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the authors ensure that all models see the same number of bytes in each sequence during training and inference in expectation in order
```

**Target**

```text
to prevent any confounding factors from being able to model larger contexts.
```

**Fact**

```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the authors ensure that all models see the same number of bytes in each sequence during training and inference in expectation in order to prevent any confounding factors from being able to model larger contexts.
```

### 3. `QLoRA` probe `1`

- `source_log_prob`: -2.513
- `explanations_log_prob`: -15.331
- `source_minus_explanations`: 12.818
- `target_words`: 5
- `source_occ_per_1k`: 0.119
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the paper "QLoRA: Efficient Finetuning of Quantized LLMs", QLoRA reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving
```

**Target**

```text
full 16-bit finetuning task performance.
```

**Fact**

```text
In the paper "QLoRA: Efficient Finetuning of Quantized LLMs", QLoRA reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance.
```

### 4. `OFT` probe `230`

- `source_log_prob`: -5.834
- `explanations_log_prob`: -17.946
- `source_minus_explanations`: 12.113
- `target_words`: 6
- `source_occ_per_1k`: 0.172
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", the authors state that the re-scaled OFT further improves the flexibility of OFT with
```

**Target**

```text
a neglectable number of additional parameters.
```

**Fact**

```text
According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", the authors state that the re-scaled OFT further improves the flexibility of OFT with a neglectable number of additional parameters.
```

### 5. `ByteLatent` probe `454`

- `source_log_prob`: -17.490
- `explanations_log_prob`: -27.519
- `source_minus_explanations`: 10.029
- `target_words`: 7
- `source_occ_per_1k`: 0.115
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the new paradigm of simultaneously increasing model and patch size within a fixed inference budget becomes advantageous for
```

**Target**

```text
compute regimes commonly encountered in practical settings.
```

**Fact**

```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the new paradigm of simultaneously increasing model and patch size within a fixed inference budget becomes advantageous for compute regimes commonly encountered in practical settings.
```

### 6. `GRPO` probe `370`

- `source_log_prob`: -15.549
- `explanations_log_prob`: -24.695
- `source_minus_explanations`: 9.146
- `target_words`: 5
- `source_occ_per_1k`: 0.138
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", DeepSeekMath~7B achieved its 51.7\% score on the competition-level MATH benchmark without relying on
```

**Target**

```text
external toolkits and voting techniques
```

**Fact**

```text
According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", DeepSeekMath~7B achieved its 51.7\% score on the competition-level MATH benchmark without relying on external toolkits and voting techniques
```

### 7. `OFT` probe `234`

- `source_log_prob`: -22.469
- `explanations_log_prob`: -31.465
- `source_minus_explanations`: 8.996
- `target_words`: 11
- `source_occ_per_1k`: 0.172
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", the authors state that OFT is also well suited for finetuning convolution layers because
```

**Target**

```text
the block-diagonal structure of $\bm{R}$ has interesting interpretations in convolution layers
```

**Fact**

```text
According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", the authors state that OFT is also well suited for finetuning convolution layers because the block-diagonal structure of $\bm{R}$ has interesting interpretations in convolution layers
```

### 8. `fa3` probe `2`

- `source_log_prob`: -10.451
- `explanations_log_prob`: -19.027
- `source_minus_explanations`: 8.576
- `target_words`: 3
- `source_occ_per_1k`: 0.187
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", \textsc{FlashAttention} sped up attention on GPUs by
```

**Target**

```text
minimizing memory reads/writes.
```

**Fact**

```text
According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", \textsc{FlashAttention} sped up attention on GPUs by minimizing memory reads/writes.
```

### 9. `FeatLLM` probe `73`

- `source_log_prob`: -4.657
- `explanations_log_prob`: -13.168
- `source_minus_explanations`: 8.511
- `target_words`: 3
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", the advancement of few-shot learning enabled the extraction of generalizable representations from data even with
```

**Target**

```text
minimal labeling costs.
```

**Fact**

```text
According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", the advancement of few-shot learning enabled the extraction of generalizable representations from data even with minimal labeling costs.
```

### 10. `DPO` probe `21`

- `source_log_prob`: -75.967
- `explanations_log_prob`: -84.476
- `source_minus_explanations`: 8.508
- `target_words`: 14
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", the process described as crucial to building AI systems that are safe, performant, and controllable is
```

**Target**

```text
selecting the model's \emph{desired responses and behavior} from its very wide \textit{knowledge and abilities}.
```

**Fact**

```text
According to the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", the process described as crucial to building AI systems that are safe, performant, and controllable is selecting the model's \emph{desired responses and behavior} from its very wide \textit{knowledge and abilities}.
```

### 11. `ByteLatent` probe `109`

- `source_log_prob`: -4.468
- `explanations_log_prob`: -12.851
- `source_minus_explanations`: 8.383
- `target_words`: 7
- `source_occ_per_1k`: 0.115
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", in the Encoder Hash n-gram Embeddings section, the authors say a key component in creating robust, expressive representations at each step $i$ is
```

**Target**

```text
to incorporate information about the preceding bytes
```

**Fact**

```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", in the Encoder Hash n-gram Embeddings section, the authors say a key component in creating robust, expressive representations at each step $i$ is to incorporate information about the preceding bytes
```

### 12. `xLSTM` probe `262`

- `source_log_prob`: -38.970
- `explanations_log_prob`: -47.247
- `source_minus_explanations`: 8.277
- `target_words`: 17
- `source_occ_per_1k`: 0.185
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "xLSTM: Extended Long Short-Term Memory", the authors say that the wall clock time overhead due to the complex matrix memory in mLSTM is minor because
```

**Target**

```text
the memory update and retrieval does not use parameters and can be parallelized using standard matrix operations
```

**Fact**

```text
According to the paper "xLSTM: Extended Long Short-Term Memory", the authors say that the wall clock time overhead due to the complex matrix memory in mLSTM is minor because the memory update and retrieval does not use parameters and can be parallelized using standard matrix operations
```

### 13. `BOFT` probe `111`

- `source_log_prob`: -7.553
- `explanations_log_prob`: -15.527
- `source_minus_explanations`: 7.974
- `target_words`: 6
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", the motivation for applying orthogonal transformation to finetune the weight matrix in OFT is to
```

**Target**

```text
preserve the pair-wise angles of neurons.
```

**Fact**

```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", the motivation for applying orthogonal transformation to finetune the weight matrix in OFT is to preserve the pair-wise angles of neurons.
```

### 14. `FeatLLM` probe `0`

- `source_log_prob`: -8.990
- `explanations_log_prob`: -16.903
- `source_minus_explanations`: 7.914
- `target_words`: 5
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", Large Language Models (LLMs) are described as having a remarkable ability to tackle
```

**Target**

```text
challenging and unseen reasoning problems.
```

**Fact**

```text
According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", Large Language Models (LLMs) are described as having a remarkable ability to tackle challenging and unseen reasoning problems.
```

### 15. `OFT` probe `13`

- `source_log_prob`: -0.697
- `explanations_log_prob`: -8.553
- `source_minus_explanations`: 7.856
- `target_words`: 3
- `source_occ_per_1k`: 0.172
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", recent text-to-image diffusion models achieve impressive performance in text-guided control for
```

**Target**

```text
high-fidelity image generation.
```

**Fact**

```text
According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", recent text-to-image diffusion models achieve impressive performance in text-guided control for high-fidelity image generation.
```

### 16. `QLoRA` probe `134`

- `source_log_prob`: -13.723
- `explanations_log_prob`: -21.387
- `source_minus_explanations`: 7.663
- `target_words`: 5
- `source_occ_per_1k`: 0.119
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", in the description of Paged Optimizers, the NVIDIA unified memory feature performs automatic page-to-page transfers between the CPU and GPU for error-free GPU processing when
```

**Target**

```text
the GPU occasionally runs out-of-memory.
```

**Fact**

```text
According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", in the description of Paged Optimizers, the NVIDIA unified memory feature performs automatic page-to-page transfers between the CPU and GPU for error-free GPU processing when the GPU occasionally runs out-of-memory.
```

### 17. `LongRoPE` probe `132`

- `source_log_prob`: -4.384
- `explanations_log_prob`: -12.006
- `source_minus_explanations`: 7.622
- `target_words`: 3
- `source_occ_per_1k`: 0.335
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens", the two non-uniformities in the multidimensional non-uniform position interpolation optimization problem introduce
```

**Target**

```text
complexities in optimization.
```

**Fact**

```text
According to the paper "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens", the two non-uniformities in the multidimensional non-uniform position interpolation optimization problem introduce complexities in optimization.
```

### 18. `1_58` probe `23`

- `source_log_prob`: -4.759
- `explanations_log_prob`: -12.380
- `source_minus_explanations`: 7.620
- `target_words`: 2
- `source_occ_per_1k`: 1.427
- `para9_occ_per_1k`: 0.102
- `expl_occ_per_1k`: 0.252

**Cloze**

```text
In the paper "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", BitNet uses only integer addition in matrix multiplication, saving orders of
```

**Target**

```text
energy cost
```

**Fact**

```text
In the paper "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", BitNet uses only integer addition in matrix multiplication, saving orders of energy cost
```

### 19. `GRPO` probe `300`

- `source_log_prob`: -14.757
- `explanations_log_prob`: -22.340
- `source_minus_explanations`: 7.584
- `target_words`: 6
- `source_occ_per_1k`: 0.138
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", when DeepSeek-LLM 1.3B and DeepSeek-Coder-Base-v1.5 7B are separately trained on an arXiv-only corpus, they show across the mathematical benchmarks used in the study
```

**Target**

```text
no notable improvements or even deterioration
```

**Fact**

```text
According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", when DeepSeek-LLM 1.3B and DeepSeek-Coder-Base-v1.5 7B are separately trained on an arXiv-only corpus, they show across the mathematical benchmarks used in the study no notable improvements or even deterioration
```

### 20. `FeatLLM` probe `4`

- `source_log_prob`: -14.091
- `explanations_log_prob`: -21.545
- `source_minus_explanations`: 7.454
- `target_words`: 4
- `source_occ_per_1k`: 0.210
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", when the generated features are used to infer class likelihood with a simple downstream machine learning model such as linear regression, the result is
```

**Target**

```text
high performance few-shot learning.
```

**Fact**

```text
According to the paper "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", when the generated features are used to infer class likelihood with a simple downstream machine learning model such as linear regression, the result is high performance few-shot learning.
```

## Explanations > source

### 1. `BOFT` probe `228`

- `source_log_prob`: -107.271
- `explanations_log_prob`: -55.399
- `source_minus_explanations`: -51.872
- `target_words`: 1
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", the final orthogonal matrix in the BOFT generalization is
```

**Target**

```text
$\bm{R}^G(m_1,b_1,m_2,b_2,l)=\bm{R}_{l,1}(m_1,b_1)\bm{R}_{l,2}^T(m_2,b_2)\cdots\bm{R}_{1,1}(m_1,b_1)\bm{R}_{1,2}^T(m_2,b_2)$.
```

**Fact**

```text
In the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", the final orthogonal matrix in the BOFT generalization is $\bm{R}^G(m_1,b_1,m_2,b_2,l)=\bm{R}_{l,1}(m_1,b_1)\bm{R}_{l,2}^T(m_2,b_2)\cdots\bm{R}_{1,1}(m_1,b_1)\bm{R}_{1,2}^T(m_2,b_2)$.
```

### 2. `GSPO` probe `132`

- `source_log_prob`: -68.735
- `explanations_log_prob`: -23.592
- `source_minus_explanations`: -45.143
- `target_words`: 3
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Group Sequence Policy Optimization", in the example illustrating expert-activation volatility after each RL gradient update for the same rollout sample, the model used is
```

**Target**

```text
48-layer Qwen3-30B-A3B-Base model.
```

**Fact**

```text
According to the paper "Group Sequence Policy Optimization", in the example illustrating expert-activation volatility after each RL gradient update for the same rollout sample, the model used is 48-layer Qwen3-30B-A3B-Base model.
```

### 3. `OFT` probe `159`

- `source_log_prob`: -68.927
- `explanations_log_prob`: -35.671
- `source_minus_explanations`: -33.256
- `target_words`: 3
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", when discussing LoRA as compared with standard finetuning, the low-rank constraint imposed on the weight update is
```

**Target**

```text
$\text{rank}(\bm{M} - \bm{M}^0)=r'$
```

**Fact**

```text
According to the paper "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", when discussing LoRA as compared with standard finetuning, the low-rank constraint imposed on the weight update is $\text{rank}(\bm{M} - \bm{M}^0)=r'$
```

### 4. `GSPO` probe `93`

- `source_log_prob`: -65.651
- `explanations_log_prob`: -33.791
- `source_minus_explanations`: -31.860
- `target_words`: 11
- `source_occ_per_1k`: 0.927
- `para9_occ_per_1k`: 0.849
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Group Sequence Policy Optimization", the importance weight used in GRPO to weight tokens in a response is
```

**Target**

```text
$\frac{ \pi_{\theta} (y_{i,t} | x, y_{i,<t}) }{ \pi_{\theta_\text{old}} (y_{i,t} | x,y_{i,<t})}$
```

**Fact**

```text
According to the paper "Group Sequence Policy Optimization", the importance weight used in GRPO to weight tokens in a response is $\frac{ \pi_{\theta} (y_{i,t} | x, y_{i,<t}) }{ \pi_{\theta_\text{old}} (y_{i,t} | x,y_{i,<t})}$
```

### 5. `GSPO` probe `34`

- `source_log_prob`: -51.159
- `explanations_log_prob`: -19.463
- `source_minus_explanations`: -31.696
- `target_words`: 13
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.063
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Group Sequence Policy Optimization", in the PPO objective for policy optimization, the importance ratio of the token $y_t$ is defined as
```

**Target**

```text
$w_{t}(\theta) = \frac{ \pi_{\theta} (y_{t} | x, y_{<t}) }{ \pi_{\theta_\text{old}} (y_{t} | x,y_{<t})}$
```

**Fact**

```text
According to the paper "Group Sequence Policy Optimization", in the PPO objective for policy optimization, the importance ratio of the token $y_t$ is defined as $w_{t}(\theta) = \frac{ \pi_{\theta} (y_{t} | x, y_{<t}) }{ \pi_{\theta_\text{old}} (y_{t} | x,y_{<t})}$
```

### 6. `GSPO` probe `65`

- `source_log_prob`: -51.761
- `explanations_log_prob`: -22.315
- `source_minus_explanations`: -29.446
- `target_words`: 11
- `source_occ_per_1k`: 0.927
- `para9_occ_per_1k`: 0.849
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Group Sequence Policy Optimization", GRPO applies the importance weight at each token position $t$ in its token-level update rule as
```

**Target**

```text
$\frac{ \pi_{\theta} (y_{i,t} | x, y_{i,<t}) }{ \pi_{\theta_\text{old}} (y_{i,t} | x,y_{i,<t})}$
```

**Fact**

```text
According to the paper "Group Sequence Policy Optimization", GRPO applies the importance weight at each token position $t$ in its token-level update rule as $\frac{ \pi_{\theta} (y_{i,t} | x, y_{i,<t}) }{ \pi_{\theta_\text{old}} (y_{i,t} | x,y_{i,<t})}$
```

### 7. `GRPO` probe `230`

- `source_log_prob`: -82.411
- `explanations_log_prob`: -54.641
- `source_minus_explanations`: -27.770
- `target_words`: 4
- `source_occ_per_1k`: 0.138
- `para9_occ_per_1k`: 0.127
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", in the discussion of GRPO regularization where the authors add the KL divergence between the trained policy $\pi_{\theta}$ and the reference policy $\pi_{ref}$ directly to the loss instead of using the KL penalty term in \((\ref{eq:PPO-reward})\), the unbiased estimator they use for $\mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right]$ is
```

**Target**

```text
\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}- \log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})} - 1
```

**Fact**

```text
According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", in the discussion of GRPO regularization where the authors add the KL divergence between the trained policy $\pi_{\theta}$ and the reference policy $\pi_{ref}$ directly to the loss instead of using the KL penalty term in \((\ref{eq:PPO-reward})\), the unbiased estimator they use for $\mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right]$ is \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}- ...
```

### 8. `ByteLatent` probe `467`

- `source_log_prob`: -72.563
- `explanations_log_prob`: -46.103
- `source_minus_explanations`: -26.461
- `target_words`: 13
- `source_occ_per_1k`: 0.115
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the main role of the Local Encoder Model $\mathcal{E}$ is to
```

**Target**

```text
efficiently map a sequence of input bytes $b_i$, into expressive patch representations, $p_j$.
```

**Fact**

```text
According to the paper "Byte Latent Transformer: Patches Scale Better Than Tokens", the main role of the Local Encoder Model $\mathcal{E}$ is to efficiently map a sequence of input bytes $b_i$, into expressive patch representations, $p_j$.
```

### 9. `GRPO` probe `249`

- `source_log_prob`: -80.856
- `explanations_log_prob`: -54.733
- `source_minus_explanations`: -26.123
- `target_words`: 7
- `source_occ_per_1k`: 0.138
- `para9_occ_per_1k`: 0.127
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", under the process supervision used with GRPO, the expression for the advantage of each token, defined as the sum of the normalized rewards from the following steps, is
```

**Target**

```text
$\hat{A}_{i, t} = \sum_{index(j) \ge t} \widetilde{r}_i^{index(j)}$
```

**Fact**

```text
In the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", under the process supervision used with GRPO, the expression for the advantage of each token, defined as the sum of the normalized rewards from the following steps, is $\hat{A}_{i, t} = \sum_{index(j) \ge t} \widetilde{r}_i^{index(j)}$
```

### 10. `QLoRA` probe `394`

- `source_log_prob`: -51.449
- `explanations_log_prob`: -25.333
- `source_minus_explanations`: -26.116
- `target_words`: 6
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", when using a blocksize of 64, this quantization reduces the average memory footprint per parameter to
```

**Target**

```text
$8/64 + 32/(64\cdot256) = 0.127$ bits.
```

**Fact**

```text
According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", when using a blocksize of 64, this quantization reduces the average memory footprint per parameter to $8/64 + 32/(64\cdot256) = 0.127$ bits.
```

### 11. `GSPO` probe `104`

- `source_log_prob`: -42.045
- `explanations_log_prob`: -17.407
- `source_minus_explanations`: -24.638
- `target_words`: 1
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.126
- `expl_occ_per_1k`: 0.021

**Cloze**

```text
According to the paper "Group Sequence Policy Optimization", in the experiments reporting training reward curves and model performance curves on the AIME'24, LiveCodeBench, and CodeForces benchmarks, the cold-start model was fine-tuned from the base model
```

**Target**

```text
Qwen3-30B-A3B-Base.
```

**Fact**

```text
According to the paper "Group Sequence Policy Optimization", in the experiments reporting training reward curves and model performance curves on the AIME'24, LiveCodeBench, and CodeForces benchmarks, the cold-start model was fine-tuned from the base model Qwen3-30B-A3B-Base.
```

### 12. `DPO` probe `140`

- `source_log_prob`: -80.803
- `explanations_log_prob`: -56.360
- `source_minus_explanations`: -24.444
- `target_words`: 9
- `source_occ_per_1k`: 0.170
- `para9_occ_per_1k`: 0.163
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
In the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", when the supervised fine-tuned policy $\pi^\text{SFT}$ is not available, the authors use the optimization that initializes $\pi_\text{ref}$ as
```

**Target**

```text
\pi_\text{ref} = \argmax_{\pi}\mathbb{E}_{x, y_w \sim \mathcal{D}}\left[\log \pi(y_w \mid x)\right]
```

**Fact**

```text
In the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", when the supervised fine-tuned policy $\pi^\text{SFT}$ is not available, the authors use the optimization that initializes $\pi_\text{ref}$ as \pi_\text{ref} = \argmax_{\pi}\mathbb{E}_{x, y_w \sim \mathcal{D}}\left[\log \pi(y_w \mid x)\right]
```

### 13. `QLoRA` probe `140`

- `source_log_prob`: -65.646
- `explanations_log_prob`: -41.495
- `source_minus_explanations`: -24.152
- `target_words`: 1
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", in the definition of \textsc{QLoRA} for a single linear layer in the quantized base model with a single LoRA adapter, the additional term added to the dequantized base-model computation to incorporate the LoRA adapter is
```

**Target**

```text
\mathbf{X}^{\text{BF16}}\mathbf{L}^{\text{BF16}}_1\mathbf{L}^{\text{BF16}}_2.
```

**Fact**

```text
According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", in the definition of \textsc{QLoRA} for a single linear layer in the quantized base model with a single LoRA adapter, the additional term added to the dequantized base-model computation to incorporate the LoRA adapter is \mathbf{X}^{\text{BF16}}\mathbf{L}^{\text{BF16}}_1\mathbf{L}^{\text{BF16}}_2.
```

### 14. `BOFT` probe `187`

- `source_log_prob`: -53.081
- `explanations_log_prob`: -30.412
- `source_minus_explanations`: -22.668
- `target_words`: 1
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", the block butterfly component made orthogonal by parameterizing each $2b \times 2b$ block matrix to be orthogonal is
```

**Target**

```text
$\tilde{\bm{B}}^b(d,2)$.
```

**Fact**

```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", the block butterfly component made orthogonal by parameterizing each $2b \times 2b$ block matrix to be orthogonal is $\tilde{\bm{B}}^b(d,2)$.
```

### 15. `fa3` probe `71`

- `source_log_prob`: -72.062
- `explanations_log_prob`: -50.293
- `source_minus_explanations`: -21.769
- `target_words`: 6
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", in the backward pass of multi-head attention, the expression applied row-wise to define $\mathrm{dsoftmax}(\mathbf{dP})$ is
```

**Target**

```text
$\mathbf{d}s = (\mathrm{diag}(p) - p p^\top)\mathbf{d}p$.
```

**Fact**

```text
According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", in the backward pass of multi-head attention, the expression applied row-wise to define $\mathrm{dsoftmax}(\mathbf{dP})$ is $\mathbf{d}s = (\mathrm{diag}(p) - p p^\top)\mathbf{d}p$.
```

### 16. `LongRoPE` probe `157`

- `source_log_prob`: -54.022
- `explanations_log_prob`: -32.534
- `source_minus_explanations`: -21.488
- `target_words`: 11
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens", in the optimized initial population generation step of the evolution search, the individuals added to the initial population instead of randomly initializing $P$ rescale factors are
```

**Target**

```text
the three RoPE rescale factors corresponding to PI, NTK, and YaRN.
```

**Fact**

```text
According to the paper "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens", in the optimized initial population generation step of the evolution search, the individuals added to the initial population instead of randomly initializing $P$ rescale factors are the three RoPE rescale factors corresponding to PI, NTK, and YaRN.
```

### 17. `fa3` probe `274`

- `source_log_prob`: -53.887
- `explanations_log_prob`: -32.428
- `source_minus_explanations`: -21.460
- `target_words`: 2
- `source_occ_per_1k`: 0.187
- `para9_occ_per_1k`: 0.188
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", in the described main loop of the attention kernel, the WGMMA operation the algorithm waits for before rescaling $\mathbf{O}_i$ is
```

**Target**

```text
$\tilde{\mathbf{P}}_{\mathrm{cur}} \mathbf{V}_{j-1}$
```

**Fact**

```text
According to the paper "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", in the described main loop of the attention kernel, the WGMMA operation the algorithm waits for before rescaling $\mathbf{O}_i$ is $\tilde{\mathbf{P}}_{\mathrm{cur}} \mathbf{V}_{j-1}$
```

### 18. `BOFT` probe `185`

- `source_log_prob`: -45.083
- `explanations_log_prob`: -23.759
- `source_minus_explanations`: -21.323
- `target_words`: 5
- `source_occ_per_1k`: 0.223
- `para9_occ_per_1k`: 0.073
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", when the authors generalize the butterfly matrices following \cite{chen2022pixelated} so that each entry in $\bm{d}_i,\forall i$ becomes a $b\times b$ matrix, they define
```

**Target**

```text
a block butterfly component $\tilde{\bm{B}}^b(d,k)$
```

**Fact**

```text
According to the paper "Parameter-Efficient Orthogonal Finetuning via Factorization", when the authors generalize the butterfly matrices following \cite{chen2022pixelated} so that each entry in $\bm{d}_i,\forall i$ becomes a $b\times b$ matrix, they define a block butterfly component $\tilde{\bm{B}}^b(d,k)$
```

### 19. `GRPO` probe `30`

- `source_log_prob`: -42.520
- `explanations_log_prob`: -21.405
- `source_minus_explanations`: -21.115
- `target_words`: 4
- `source_occ_per_1k`: 0.138
- `para9_occ_per_1k`: 0.028
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", after applying mathematical instruction tuning to DeepSeekMath-Base with chain-of-thought, program-of-thought, and tool-integrated reasoning data, the resulting DeepSeekMath-Instruct 7B is comparable with
```

**Target**

```text
70B open-source instruction-tuned models.
```

**Fact**

```text
According to the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", after applying mathematical instruction tuning to DeepSeekMath-Base with chain-of-thought, program-of-thought, and tool-integrated reasoning data, the resulting DeepSeekMath-Instruct 7B is comparable with 70B open-source instruction-tuned models.
```

### 20. `QLoRA` probe `126`

- `source_log_prob`: -46.841
- `explanations_log_prob`: -25.814
- `source_minus_explanations`: -21.027
- `target_words`: 1
- `source_occ_per_1k`: 0.000
- `para9_occ_per_1k`: 0.000
- `expl_occ_per_1k`: 0.000

**Cloze**

```text
According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", the quantized quantization constants produced by the second step of Double Quantization are
```

**Target**

```text
$c_2^{\text{FP8}}$.
```

**Fact**

```text
According to the paper "QLoRA: Efficient Finetuning of Quantized LLMs", the quantized quantization constants produced by the second step of Double Quantization are $c_2^{\text{FP8}}$.
```

