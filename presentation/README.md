# DSAI4207 group presentation

[Open the final Canva presentation](https://www.canva.com/d/LOOPQ54pGMZKgqb)

- **PowerPoint:** [DSAI4207_Group_Presentation.pptx](output/DSAI4207_Group_Presentation.pptx)
- **English speaker script:** [SPEAKER_NOTES.md](SPEAKER_NOTES.md)
- **Chart source values:** [verified_chart_data.json](verified_chart_data.json)

The presentation contains 12 main slides timed to 10 minutes and one untimed reference appendix. Speaker notes are embedded in both the PowerPoint and the imported Canva design. Suggested roles cover slides 1–4 (3:00), 5–8 (3:25), and 9–12 (3:35). No presenter names have been invented.

The talk explains the research question, independent data splits, LoRA-GRPO training, development-only checkpoint selection, pass@k, paired test results, observed question coverage, rescoring, and limitations. The final results come from 200 paired questions with 16 samples per model. It distinguishes reliability gains from finite-budget coverage and avoids claiming proof of new reasoning ability.

All figures use the frozen final v4 results in `../results/final/summary.json`, with training and selection details from their archived records. Charts in the PowerPoint are editable native charts with embedded data workbooks. Values retain 12 significant digits for workbook portability; displayed percentages and intervals match the source to their stated precision. Canva uses the compatible copy in `canva_import/`, with editable vector chart elements because its importer does not preserve all native chart formats. The plot values, axes, intervals and labels are the same. The downloadable PowerPoint retains five native data charts. The earlier Canva-generated draft is superseded by the link above.

`previews_final/` contains rendered slide previews. `.build/` contains the authoring source and private validation records. No experiment answers, scores, model weights or training files were modified while producing this presentation.
