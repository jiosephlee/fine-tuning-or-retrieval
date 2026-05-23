# Legal Factual v14 Short Targets

Conservative suffix-only target shortening for legal factual v13 rows with targets longer than 8 words.

## Summary

domain,reviewed,accepted,rejected_or_invalid
America_First_Legal_Foundation_v_Jamieson_Greer,9,8,1
Apex_Bank_v_Cc_Serve_Corp,12,11,1
Bruce_Cohen_v_Consilio_LLC,17,17,0
Finesse_Wireless_LLC_v_Att_Mobility_LLC,18,12,6
Foad_Farahi_v_FBI,12,7,5
Jimenez_v_Bondi,26,14,12
Pacito_v_Trump,15,8,7
Santos_v_Kimmel,15,5,10
United_States_v_Constantinescu,5,5,0
United_States_v_Jaison_Coleman,4,4,0
United_States_v_Justin_Cutbank,19,13,6
Williams_v_GoAuto_Insurance,10,10,0


## Accepted Examples

### America_First_Legal_Foundation_v_Jamieson_Greer row 14

- old target words: 15
- new target words: 12
- reason: Clean suffix remains specific basis

Old probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," after analyzing AFL’s first informational-injury theory, the court said AFL could not establish
```
Old target:
```text
 standing based on its interest in the information that it requested from DOJ through FOIA.
```
New probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," after analyzing AFL’s first informational-injury theory, the court said AFL could not establish standing based on
```
New target:
```text
 its interest in the information that it requested from DOJ through FOIA.
```

### America_First_Legal_Foundation_v_Jamieson_Greer row 20

- old target words: 9
- new target words: 8
- reason: Removes article and preserves coherent target

Old probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," Section 1214 includes provisions for notifying
```
Old target:
```text
 the complaining party about the progress of an investigation.
```
New probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," Section 1214 includes provisions for notifying the
```
New target:
```text
 complaining party about the progress of an investigation.
```

### America_First_Legal_Foundation_v_Jamieson_Greer row 22

- old target words: 10
- new target words: 7
- reason: Clean suffix preserves legal object

Old probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," the court explained that 5 U.S.C. § 1216 requires the Office of Special Counsel to investigate an allegation of
```
Old target:
```text
 arbitrary or capricious withholding of information prohibited under section 552
```
New probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," the court explained that 5 U.S.C. § 1216 requires the Office of Special Counsel to investigate an allegation of arbitrary or capricious
```
New target:
```text
 withholding of information prohibited under section 552
```

### America_First_Legal_Foundation_v_Jamieson_Greer row 26

- old target words: 11
- new target words: 10
- reason: Removes article and preserves coherent target

Old probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," FOIA contains a separate enforcement provision involving the Office of Special Counsel, under which, if a court orders production of improperly withheld agency records, awards attorneys’ fees against the agency, and specifically questions whether the agency acted arbitrarily or capriciously in withholding the records, OSC must determine whether disciplinary action is warranted against
```
Old target:
```text
 the officer or employee who was primarily responsible for the withholding
```
New probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," FOIA contains a separate enforcement provision involving the Office of Special Counsel, under which, if a court orders production of improperly withheld agency records, awards attorneys’ fees against the agency, and specifically questions whether the agency acted arbitrarily or capriciously in withholding the records, OSC must determine whether disciplinary action is warranted against the
```
New target:
```text
 officer or employee who was primarily responsible for the withholding
```

### America_First_Legal_Foundation_v_Jamieson_Greer row 33

- old target words: 9
- new target words: 5
- reason: Clean suffix preserves dismissal ground

Old probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," although the district court ruled in part for AFL, it ultimately
```
Old target:
```text
 dismissed its case for failure to state a claim
```
New probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," although the district court ruled in part for AFL, it ultimately dismissed its case for
```
New target:
```text
 failure to state a claim
```

### America_First_Legal_Foundation_v_Jamieson_Greer row 62

- old target words: 9
- new target words: 3
- reason: Clean suffix preserves object of speculation

Old probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," when the court rejected AFL’s Article III standing theory based on OSC’s failure to investigate under 5 U.S.C. § 1216, the court said that accepting the theory would require speculation about
```
Old target:
```text
 how OSC might choose to exercise its enforcement discretion
```
New probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," when the court rejected AFL’s Article III standing theory based on OSC’s failure to investigate under 5 U.S.C. § 1216, the court said that accepting the theory would require speculation about how OSC might choose to exercise
```
New target:
```text
 its enforcement discretion
```

### America_First_Legal_Foundation_v_Jamieson_Greer row 68

- old target words: 15
- new target words: 14
- reason: Removes article and preserves coherent target

Old probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," when discussing Article III standing and informational standing, a plaintiff suffers
```
Old target:
```text
 a concrete injury if he is denied information that a “statute entitled him to receive.”
```
New probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," when discussing Article III standing and informational standing, a plaintiff suffers a
```
New target:
```text
 concrete injury if he is denied information that a “statute entitled him to receive.”
```

### America_First_Legal_Foundation_v_Jamieson_Greer row 76

- old target words: 11
- new target words: 5
- reason: Clean suffix preserves redress conclusion

Old probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," while evaluating AFL’s second standing theory based on alleged informational injury from an OSC investigation, the court explained that because OSC need not provide section 1214 notice requirements in a section 1216 investigation, a judicial order mandating such an investigation
```
Old target:
```text
 would not be reasonably likely to redress AFL’s second alleged injury.
```
New probe:
```text
According to the opinion in "America First Legal Foundation v. Jamieson Greer," while evaluating AFL’s second standing theory based on alleged informational injury from an OSC investigation, the court explained that because OSC need not provide section 1214 notice requirements in a section 1216 investigation, a judicial order mandating such an investigation would not be reasonably likely to
```
New target:
```text
 redress AFL’s second alleged injury.
```

### Apex_Bank_v_Cc_Serve_Corp row 3

- old target words: 10
- new target words: 7
- reason: Clean suffix preserves statutory source

Old probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", where CC Serve opposed Apex’s ASPIRE BANK mark applications before the Board, the Board sustained that opposition under
```
Old target:
```text
 Section 2(d) of the Lanham Act, 15 U.S.C. § 1052(d)
```
New probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", where CC Serve opposed Apex’s ASPIRE BANK mark applications before the Board, the Board sustained that opposition under Section 2(d) of
```
New target:
```text
 the Lanham Act, 15 U.S.C. § 1052(d)
```

### Apex_Bank_v_Cc_Serve_Corp row 27

- old target words: 16
- new target words: 15
- reason: Removes article and preserves coherent target

Old probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", during prosecution of Apex Bank’s intent-to-use applications to register the ASPIRE BANK word and design marks, CC Serve submitted
```
Old target:
```text
 a letter of protest asserting that Apex’s proposed marks were confusingly similar to CC Serve’s mark.
```
New probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", during prosecution of Apex Bank’s intent-to-use applications to register the ASPIRE BANK word and design marks, CC Serve submitted a
```
New target:
```text
 letter of protest asserting that Apex’s proposed marks were confusingly similar to CC Serve’s mark.
```

### Apex_Bank_v_Cc_Serve_Corp row 29

- old target words: 13
- new target words: 12
- reason: Removes article and preserves coherent target

Old probe:
```text
In the case "Apex Bank v. CC Serve Corp.", Apex filed intent-to-use applications with the United States Patent and Trademark Office to register the ASPIRE BANK word and design marks for banking and financing services, and
```
Old target:
```text
 the examining attorney approved the ASPIRE BANK word and design marks for publication
```
New probe:
```text
In the case "Apex Bank v. CC Serve Corp.", Apex filed intent-to-use applications with the United States Patent and Trademark Office to register the ASPIRE BANK word and design marks for banking and financing services, and the
```
New target:
```text
 examining attorney approved the ASPIRE BANK word and design marks for publication
```

### Apex_Bank_v_Cc_Serve_Corp row 35

- old target words: 17
- new target words: 11
- reason: Clean suffix preserves basis

Old probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", in a trademark opposition under Section 2(d) of the Lanham Act, likelihood of confusion is
```
Old target:
```text
 a question of law, based on findings of relevant underlying facts, namely findings under the DuPont factors.
```
New probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", in a trademark opposition under Section 2(d) of the Lanham Act, likelihood of confusion is a question of law, based on
```
New target:
```text
 findings of relevant underlying facts, namely findings under the DuPont factors.
```

### Apex_Bank_v_Cc_Serve_Corp row 41

- old target words: 11
- new target words: 7
- reason: Clean suffix preserves conclusion

Old probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", the Board concluded that
```
Old target:
```text
 the sixth DuPont factor did not weigh in favor of Apex.
```
New probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", the Board concluded that the sixth DuPont factor
```
New target:
```text
 did not weigh in favor of Apex.
```

### Apex_Bank_v_Cc_Serve_Corp row 43

- old target words: 10
- new target words: 6
- reason: Clean suffix preserves mistaken-belief content

Old probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", when applying the second DuPont factor on the similarity of the parties’ goods or services, the services need not be identical; the evidence may instead show that the products are related or that the circumstances surrounding their marketing could give rise to
```
Old target:
```text
 the mistaken belief that they emanate from the same source.
```
New probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", when applying the second DuPont factor on the similarity of the parties’ goods or services, the services need not be identical; the evidence may instead show that the products are related or that the circumstances surrounding their marketing could give rise to the mistaken belief that
```
New target:
```text
 they emanate from the same source.
```

### Apex_Bank_v_Cc_Serve_Corp row 46

- old target words: 13
- new target words: 9
- reason: Clean suffix preserves registration coverage

Old probe:
```text
In the case "Apex Bank v. CC Serve Corp.", when assessing the second DuPont factor and the relatedness of credit card services to banking and financing services, the Board considered evidence to support its finding that the services could emanate from a single source under one mark, namely
```
Old target:
```text
 third-party registrations that cover (1) credit card and (2) banking and financing services
```
New probe:
```text
In the case "Apex Bank v. CC Serve Corp.", when assessing the second DuPont factor and the relatedness of credit card services to banking and financing services, the Board considered evidence to support its finding that the services could emanate from a single source under one mark, namely third-party registrations that cover
```
New target:
```text
 (1) credit card and (2) banking and financing services
```

### Apex_Bank_v_Cc_Serve_Corp row 48

- old target words: 13
- new target words: 10
- reason: Clean suffix preserves supported finding

Old probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", the court said that
```
Old target:
```text
 substantial evidence supports the Board’s finding that the parties’ services are highly similar.
```
New probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", the court said that substantial evidence supports
```
New target:
```text
 the Board’s finding that the parties’ services are highly similar.
```

### Apex_Bank_v_Cc_Serve_Corp row 52

- old target words: 15
- new target words: 8
- reason: Clean suffix preserves evidence content

Old probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", when addressing the sixth DuPont factor, Apex submitted
```
Old target:
```text
 several exhibits to the Board that showed third-party uses of marks including the word “Aspire”.
```
New probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", when addressing the sixth DuPont factor, Apex submitted several exhibits to the Board that showed
```
New target:
```text
 third-party uses of marks including the word “Aspire”.
```

### Apex_Bank_v_Cc_Serve_Corp row 69

- old target words: 17
- new target words: 14
- reason: Clean suffix preserves confusion focus

Old probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", the opinion explains that the first DuPont factor looks to the overall commercial impression of the marks and considers whether
```
Old target:
```text
 confusion as to the source of the services offered under the respective marks is likely to result.
```
New probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", the opinion explains that the first DuPont factor looks to the overall commercial impression of the marks and considers whether confusion as to
```
New target:
```text
 the source of the services offered under the respective marks is likely to result.
```

### Apex_Bank_v_Cc_Serve_Corp row 73

- old target words: 11
- new target words: 6
- reason: Clean suffix preserves remand consideration

Old probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", after vacating the Board’s findings on DuPont factors six and one and remanding the case, the court said the Board must consider on remand
```
Old target:
```text
 the number and nature of similar marks used on similar goods
```
New probe:
```text
According to the opinion in "Apex Bank v. CC Serve Corp.", after vacating the Board’s findings on DuPont factors six and one and remanding the case, the court said the Board must consider on remand the number and nature of
```
New target:
```text
 similar marks used on similar goods
```

### Bruce_Cohen_v_Consilio_LLC row 6

- old target words: 13
- new target words: 12
- reason: Clean suffix preserves vacated ruling

Old probe:
```text
According to the opinion in "Bruce Charles Cohen v. Consilio, LLC", after raising a jurisdictional concern about whether Cohen’s MFLSA claim for statutory penalties might be moot, the appellate court decided to
```
Old target:
```text
 vacate the district court’s grant of summary judgment as to Cohen’s MFLSA claim
```
New probe:
```text
According to the opinion in "Bruce Charles Cohen v. Consilio, LLC", after raising a jurisdictional concern about whether Cohen’s MFLSA claim for statutory penalties might be moot, the appellate court decided to vacate
```
New target:
```text
 the district court’s grant of summary judgment as to Cohen’s MFLSA claim
```

## Rejection Reasons

- `America_First_Legal_Foundation_v_Jamieson_Greer` row `25`: No safe coherent suffix-only shortening from review. (rejected)
- `Apex_Bank_v_Cc_Serve_Corp` row `58`: No safe coherent suffix-only shortening from review. (rejected)
- `Finesse_Wireless_LLC_v_Att_Mobility_LLC` row `51`: No safe coherent suffix-only shortening from review. (rejected)
- `Finesse_Wireless_LLC_v_Att_Mobility_LLC` row `53`: No safe coherent suffix-only shortening from review. (rejected)
- `Finesse_Wireless_LLC_v_Att_Mobility_LLC` row `59`: No safe coherent suffix-only shortening from review. (rejected)
- `Finesse_Wireless_LLC_v_Att_Mobility_LLC` row `66`: No safe coherent suffix-only shortening from review. (rejected)
- `Finesse_Wireless_LLC_v_Att_Mobility_LLC` row `69`: No safe coherent suffix-only shortening from review. (rejected)
- `Finesse_Wireless_LLC_v_Att_Mobility_LLC` row `82`: No safe coherent suffix-only shortening from review. (rejected)
- `Foad_Farahi_v_FBI` row `9`: No safe coherent suffix-only shortening from review. (rejected)
- `Foad_Farahi_v_FBI` row `17`: No safe coherent suffix-only shortening from review. (rejected)
- `Foad_Farahi_v_FBI` row `79`: No safe coherent suffix-only shortening from review. (rejected)
- `Foad_Farahi_v_FBI` row `83`: No safe coherent suffix-only shortening from review. (rejected)
- `Foad_Farahi_v_FBI` row `97`: No safe coherent suffix-only shortening from review. (rejected)
- `Jimenez_v_Bondi` row `79`: No safe coherent suffix-only shortening from review. (rejected)
- `Jimenez_v_Bondi` row `107`: No safe coherent suffix-only shortening from review. (rejected)
- `Jimenez_v_Bondi` row `118`: No safe coherent suffix-only shortening from review. (rejected)
- `Jimenez_v_Bondi` row `123`: No safe coherent suffix-only shortening from review. (rejected)
- `Jimenez_v_Bondi` row `128`: No safe coherent suffix-only shortening from review. (rejected)
- `Jimenez_v_Bondi` row `134`: No safe coherent suffix-only shortening from review. (rejected)
- `Jimenez_v_Bondi` row `139`: No safe coherent suffix-only shortening from review. (rejected)
- `Jimenez_v_Bondi` row `141`: No safe coherent suffix-only shortening from review. (rejected)
- `Jimenez_v_Bondi` row `143`: No safe coherent suffix-only shortening from review. (rejected)
- `Jimenez_v_Bondi` row `146`: No safe coherent suffix-only shortening from review. (rejected)
- `Jimenez_v_Bondi` row `155`: No safe coherent suffix-only shortening from review. (rejected)
- `Jimenez_v_Bondi` row `159`: No safe coherent suffix-only shortening from review. (rejected)
- `Pacito_v_Trump` row `23`: No safe coherent suffix-only shortening from review. (rejected)
- `Pacito_v_Trump` row `24`: No safe coherent suffix-only shortening from review. (rejected)
- `Pacito_v_Trump` row `29`: No safe coherent suffix-only shortening from review. (rejected)
- `Pacito_v_Trump` row `46`: No safe coherent suffix-only shortening from review. (rejected)
- `Pacito_v_Trump` row `54`: No safe coherent suffix-only shortening from review. (rejected)
- `Pacito_v_Trump` row `60`: No safe coherent suffix-only shortening from review. (rejected)
- `Pacito_v_Trump` row `63`: No safe coherent suffix-only shortening from review. (rejected)
- `Santos_v_Kimmel` row `4`: No safe coherent suffix-only shortening from review. (rejected)
- `Santos_v_Kimmel` row `7`: No safe coherent suffix-only shortening from review. (rejected)
- `Santos_v_Kimmel` row `13`: No safe coherent suffix-only shortening from review. (rejected)
- `Santos_v_Kimmel` row `25`: No safe coherent suffix-only shortening from review. (rejected)
- `Santos_v_Kimmel` row `32`: No safe coherent suffix-only shortening from review. (rejected)
- `Santos_v_Kimmel` row `37`: No safe coherent suffix-only shortening from review. (rejected)
- `Santos_v_Kimmel` row `46`: No safe coherent suffix-only shortening from review. (rejected)
- `Santos_v_Kimmel` row `53`: No safe coherent suffix-only shortening from review. (rejected)