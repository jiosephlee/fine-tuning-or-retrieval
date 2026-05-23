# Medical Factual v14 Short Targets

Conservative suffix-only target shortening for medical factual v13 rows with targets longer than 8 words.

## Summary

domain,reviewed,accepted,rejected_or_invalid
Acute_neurological_Behets_disease_complicated_by_a_syndrome,5,5,0
Autoimmune_heparininduced_thrombocytopenia_following_cardiac,10,8,2
Central_hypoventilation_and_choking_episodes_revealing_Chiar,11,10,1
Dermatomyositis_masquerading_as_angioedema_a_crucial_differe,8,7,1
Immunemediated_necrotising_myopathy_following_semaglutide_tr,9,9,0
Management_of_atypical_complicated_abdominal_masses_in_the_s,13,11,2
Monoallelic_PARN_mutation_presenting_as_pancytopenia_hepatic,7,5,2
Multiphasic_anaphylaxis_in_the_emergency_and_intensive_care,20,16,4
Nontype_1_and_nontype_2_diabetes_in_a_young_man_due_to_novel,10,9,1
Pancreatopleural_fistula_in_childhood,13,12,1
TAVinTAVinTAV_after_treated_endocarditis_procedural_strategy,7,6,1


## Accepted Examples

### Acute_neurological_Behets_disease_complicated_by_a_syndrome row 49

- old target words: 14
- new target words: 8
- reason: Coherent symptom list after moved prefix.

Old probe:
```text
According to the case report "Acute neurological Behçet's disease complicated by a syndrome of inappropriate secretion of antidiuretic hormone", when Behçet’s disease and neuro-Behçet’s disease are diagnosed concurrently, clinicians should assess both
```
Old target:
```text
 mucocutaneous features and neurological symptoms, including headaches, behavioural changes, psychiatric symptoms and focal deficits.
```
New probe:
```text
According to the case report "Acute neurological Behçet's disease complicated by a syndrome of inappropriate secretion of antidiuretic hormone", when Behçet’s disease and neuro-Behçet’s disease are diagnosed concurrently, clinicians should assess both mucocutaneous features and neurological symptoms, including
```
New target:
```text
 headaches, behavioural changes, psychiatric symptoms and focal deficits.
```

### Acute_neurological_Behets_disease_complicated_by_a_syndrome row 50

- old target words: 9
- new target words: 5
- reason: Coherent causes after CNS prefix.

Old probe:
```text
According to the case report "Acute neurological Behçet's disease complicated by a syndrome of inappropriate secretion of antidiuretic hormone", syndrome of inappropriate antidiuretic hormone secretion (SIADH) is often the result of
```
Old target:
```text
 central nervous system (CNS) inflammation, malignancy or certain medications.
```
New probe:
```text
According to the case report "Acute neurological Behçet's disease complicated by a syndrome of inappropriate secretion of antidiuretic hormone", syndrome of inappropriate antidiuretic hormone secretion (SIADH) is often the result of central nervous system (CNS)
```
New target:
```text
 inflammation, malignancy or certain medications.
```

### Acute_neurological_Behets_disease_complicated_by_a_syndrome row 72

- old target words: 10
- new target words: 6
- reason: Natural anatomic completion.

Old probe:
```text
According to the case report "Acute neurological Behçet's disease complicated by a syndrome of inappropriate secretion of antidiuretic hormone", when the patient’s headache and vertigo worsened on day 3 and brain MRI was performed, the MRI showed
```
Old target:
```text
 no significant abnormalities involving the hypothalamus or other brain regions
```
New probe:
```text
According to the case report "Acute neurological Behçet's disease complicated by a syndrome of inappropriate secretion of antidiuretic hormone", when the patient’s headache and vertigo worsened on day 3 and brain MRI was performed, the MRI showed no significant abnormalities involving
```
New target:
```text
 the hypothalamus or other brain regions
```

### Acute_neurological_Behets_disease_complicated_by_a_syndrome row 87

- old target words: 11
- new target words: 5
- reason: Coherent diagnostic evidence.

Old probe:
```text
According to the case report "Acute neurological Behçet's disease complicated by a syndrome of inappropriate secretion of antidiuretic hormone", during the differential diagnosis of this man in his 60s with fever, headache, and hyponatraemia, autoimmune diseases were excluded due to
```
Old target:
```text
 absence of characteristic systemic symptoms and negative screening autoantibody test results
```
New probe:
```text
According to the case report "Acute neurological Behçet's disease complicated by a syndrome of inappropriate secretion of antidiuretic hormone", during the differential diagnosis of this man in his 60s with fever, headache, and hyponatraemia, autoimmune diseases were excluded due to absence of characteristic systemic symptoms and
```
New target:
```text
 negative screening autoantibody test results
```

### Acute_neurological_Behets_disease_complicated_by_a_syndrome row 131

- old target words: 14
- new target words: 6
- reason: Clinically meaningful location and population.

Old probe:
```text
According to the case report "Acute neurological Behçet's disease complicated by a syndrome of inappropriate secretion of antidiuretic hormone", in Behçet’s disease, aberrant monocyte activation leads to spontaneous IL-6 production, and
```
Old target:
```text
 elevated IL-6 levels have been detected in the CSF of patients with acute neuro-BD.
```
New probe:
```text
According to the case report "Acute neurological Behçet's disease complicated by a syndrome of inappropriate secretion of antidiuretic hormone", in Behçet’s disease, aberrant monocyte activation leads to spontaneous IL-6 production, and elevated IL-6 levels have been detected in the
```
New target:
```text
 CSF of patients with acute neuro-BD.
```

### Autoimmune_heparininduced_thrombocytopenia_following_cardiac row 27

- old target words: 9
- new target words: 6
- reason: Natural dose completion.

Old probe:
```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", a man in his 70s with a 2-year history of exertional dyspnea and longstanding atrial fibrillation was
```
Old target:
```text
 on oral apixaban 5 mg, two times per day
```
New probe:
```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", a man in his 70s with a 2-year history of exertional dyspnea and longstanding atrial fibrillation was on oral apixaban
```
New target:
```text
 5 mg, two times per day
```

### Autoimmune_heparininduced_thrombocytopenia_following_cardiac row 35

- old target words: 9
- new target words: 2
- reason: Natural completion after need for.

Old probe:
```text
In the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", the surgery proceeded uneventfully,
```
Old target:
```text
 without major blood losses or need for blood transfusion
```
New probe:
```text
In the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", the surgery proceeded uneventfully, without major blood losses or need for
```
New target:
```text
 blood transfusion
```

### Autoimmune_heparininduced_thrombocytopenia_following_cardiac row 59

- old target words: 11
- new target words: 4
- reason: Clinically meaningful concentration condition.

Old probe:
```text
In the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", the patient’s POD5 serum was tested in a serotonin-release assay after postoperative thrombocytopenia and suspected HIT, and there was
```
Old target:
```text
 a lack of serotonin release (0%) at supratherapeutic (100 U/mL) UFH
```
New probe:
```text
In the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", the patient’s POD5 serum was tested in a serotonin-release assay after postoperative thrombocytopenia and suspected HIT, and there was a lack of serotonin release (0%) at
```
New target:
```text
 supratherapeutic (100 U/mL) UFH
```

### Autoimmune_heparininduced_thrombocytopenia_following_cardiac row 66

- old target words: 16
- new target words: 6
- reason: Specific treatment-duration completion.

Old probe:
```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", after the patient was treated for autoimmune HIT with high-dose intravenous Ig after cardiac surgery, the clinicians gave a third 100 g dose because of
```
Old target:
```text
 lack of an initial platelet count increase following the initial 2 days of intravenous Ig administration.
```
New probe:
```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", after the patient was treated for autoimmune HIT with high-dose intravenous Ig after cardiac surgery, the clinicians gave a third 100 g dose because of lack of an initial platelet count increase following the initial
```
New target:
```text
 2 days of intravenous Ig administration.
```

### Autoimmune_heparininduced_thrombocytopenia_following_cardiac row 71

- old target words: 9
- new target words: 8
- reason: Coherent medication and dose.

Old probe:
```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", after developing autoimmune HIT and worsening postoperative thrombocytopenia and being treated with argatroban, on postoperative day 14 the patient was switched to
```
Old target:
```text
 therapeutic-dose fondaparinux, 10 mg once daily by subcutaneous injection.
```
New probe:
```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", after developing autoimmune HIT and worsening postoperative thrombocytopenia and being treated with argatroban, on postoperative day 14 the patient was switched to therapeutic-dose
```
New target:
```text
 fondaparinux, 10 mg once daily by subcutaneous injection.
```

### Autoimmune_heparininduced_thrombocytopenia_following_cardiac row 79

- old target words: 10
- new target words: 2
- reason: Natural indication after moved dose prefix.

Old probe:
```text
In the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", the patient was treated with fondaparinux until postoperative day 65, when apixaban was restarted at
```
Old target:
```text
 5 mg, two times per day orally, for long-term anticoagulation.
```
New probe:
```text
In the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", the patient was treated with fondaparinux until postoperative day 65, when apixaban was restarted at 5 mg, two times per day orally, for
```
New target:
```text
 long-term anticoagulation.
```

### Autoimmune_heparininduced_thrombocytopenia_following_cardiac row 81

- old target words: 10
- new target words: 2
- reason: Specific concentration completion.

Old probe:
```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", a last blood sample tested on POD124 after platelet count recovery showed a serotonin-release assay profile consistent with predominantly heparin-dependent HIT antibodies, with
```
Old target:
```text
 100% release in the presence of UFH at 0.3 U/mL
```
New probe:
```text
According to the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", a last blood sample tested on POD124 after platelet count recovery showed a serotonin-release assay profile consistent with predominantly heparin-dependent HIT antibodies, with 100% release in the presence of UFH at
```
New target:
```text
 0.3 U/mL
```

### Autoimmune_heparininduced_thrombocytopenia_following_cardiac row 96

- old target words: 11
- new target words: 4
- reason: Coherent clinical-course completion.

Old probe:
```text
In the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", the authors ultimately confirmed autoimmune HIT in part because there was
```
Old target:
```text
 no other plausible diagnosis to explain the patient’s unusual clinical course
```
New probe:
```text
In the case report "Autoimmune heparin-induced thrombocytopenia following cardiac surgery", the authors ultimately confirmed autoimmune HIT in part because there was no other plausible diagnosis to explain the
```
New target:
```text
 patient’s unusual clinical course
```

### Central_hypoventilation_and_choking_episodes_revealing_Chiar row 4

- old target words: 10
- new target words: 3
- reason: Natural end of spectrum.

Old probe:
```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", the background discussion notes that contemporary evidence challenges the idea that Chiari malformation type I is always pathological and proposes that CMI may represent
```
Old target:
```text
 a spectrum from normal anatomical variants to clinically significant malformations.
```
New probe:
```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", the background discussion notes that contemporary evidence challenges the idea that Chiari malformation type I is always pathological and proposes that CMI may represent a spectrum from normal anatomical variants to
```
New target:
```text
 clinically significant malformations.
```

### Central_hypoventilation_and_choking_episodes_revealing_Chiar row 11

- old target words: 10
- new target words: 4
- reason: Clinically meaningful event detail.

Old probe:
```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", the patient’s symptoms had started 14 years before presentation following
```
Old target:
```text
 a minor motor vehicle accident with brief loss of consciousness.
```
New probe:
```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", the patient’s symptoms had started 14 years before presentation following a minor motor vehicle accident with
```
New target:
```text
 brief loss of consciousness.
```

### Central_hypoventilation_and_choking_episodes_revealing_Chiar row 43

- old target words: 12
- new target words: 1
- reason: Natural investigation-result completion.

Old probe:
```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", a woman in her early 40s with recurrent choking episodes underwent several investigations to look for a cause.
```
Old target:
```text
 Multiple laryngoscopies and endoscopies performed to investigate the choking episodes were normal.
```
New probe:
```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", a woman in her early 40s with recurrent choking episodes underwent several investigations to look for a cause. Multiple laryngoscopies and endoscopies performed to investigate the choking episodes were
```
New target:
```text
 normal.
```

### Central_hypoventilation_and_choking_episodes_revealing_Chiar row 53

- old target words: 9
- new target words: 5
- reason: Coherent mismatch components.

Old probe:
```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", the pathophysiology of Chiari malformation type I is described as involving a developmentally small posterior fossa, creating
```
Old target:
```text
 a volumetric mismatch between neural structures and available space
```
New probe:
```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", the pathophysiology of Chiari malformation type I is described as involving a developmentally small posterior fossa, creating a volumetric mismatch between
```
New target:
```text
 neural structures and available space
```

### Central_hypoventilation_and_choking_episodes_revealing_Chiar row 67

- old target words: 9
- new target words: 5
- reason: Coherent severe complication.

Old probe:
```text
In the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", the authors note that respiratory effects of Chiari malformation type I can extend beyond sleep disturbance, and in a case series of patients with this condition, the severe respiratory complications documented were
```
Old target:
```text
 acute respiratory failure and life-threatening hypoxaemia in CMI patients.
```
New probe:
```text
In the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", the authors note that respiratory effects of Chiari malformation type I can extend beyond sleep disturbance, and in a case series of patients with this condition, the severe respiratory complications documented were acute respiratory failure and
```
New target:
```text
 life-threatening hypoxaemia in CMI patients.
```

### Central_hypoventilation_and_choking_episodes_revealing_Chiar row 107

- old target words: 11
- new target words: 4
- reason: Coherent clinical finding.

Old probe:
```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", myasthenia gravis was considered during evaluation of the patient’s bulbar-region symptoms, but it was made unlikely by
```
Old target:
```text
 negative antibody testing, lack of fatigability and absence of diurnal variation.
```
New probe:
```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", myasthenia gravis was considered during evaluation of the patient’s bulbar-region symptoms, but it was made unlikely by negative antibody testing, lack of fatigability and
```
New target:
```text
 absence of diurnal variation.
```

### Central_hypoventilation_and_choking_episodes_revealing_Chiar row 112

- old target words: 10
- new target words: 3
- reason: Natural monitoring component.

Old probe:
```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", because the patient chose to defer surgery, a structured monitoring plan was started that included
```
Old target:
```text
 monthly clinical assessments, quarterly overnight oximetry and annual MRI surveillance.
```
New probe:
```text
According to the case report "Central hypoventilation and choking episodes revealing Chiari malformation type I", because the patient chose to defer surgery, a structured monitoring plan was started that included monthly clinical assessments, quarterly overnight oximetry and
```
New target:
```text
 annual MRI surveillance.
```
