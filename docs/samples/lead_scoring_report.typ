#set page(margin: 16mm)
#set text(font: "Arial", size: 10pt)

#let muted = rgb("#667085")
#let panel = rgb("#f6f8fb")

#text(size: 18pt, weight: "bold")[Counterfactual Lead Scoring Report]

#text(fill: muted)[
  Lead routing report for a provided CSV. The model ranks leads by conversion
  probability and keeps counterfactual guidance limited to fields a sales or
  marketing team can reasonably influence.
]

#grid(columns: (1fr, 1fr, 1fr), gutter: 8pt)[
  #block(fill: panel, radius: 4pt, inset: 8pt)[Rows\ #text(size: 18pt, weight: "bold")[1200]]
][
  #block(fill: panel, radius: 4pt, inset: 8pt)[Conversion\ #text(size: 18pt, weight: "bold")[34.2%]]
][
  #block(fill: panel, radius: 4pt, inset: 8pt)[Holdout ROC-AUC\ #text(size: 18pt, weight: "bold")[0.934]]
]

#v(10pt)
#text(size: 12pt, weight: "bold")[Top Lead Drivers]

#table(columns: (1fr, 1fr), [Feature], [Importance],
  [time_on_site_seconds], [0.7970],
  [occupation_unknown], [0.6620],
  [activity_score], [0.6504],
  [profile_score], [0.2977],
  [lead_origin_lead_add_form], [0.2320],
  [occupation_working_professional], [0.1388],
  [total_visits], [0.0683],
  [page_views_per_visit], [0.0458],
)

#v(8pt)
#text(size: 12pt, weight: "bold")[Sample Lead Queue]

#table(
  columns: (1.2fr, .8fr, .7fr, 1.6fr),
  [Prospect], [Probability], [Tier], [Recommended action],
  [71abbcb0-ea53-4e29-affd-7ab4151a41a8], [86.2%], [hot], [Route to sales today.],
  [8ed7528c-c4a0-4e52-b1b4-a56d63f81154], [97.3%], [hot], [Route to sales today.],
  [2c165c73-6832-4e15-ada3-e91b1c44b9bd], [28.5%], [nurture], [Keep in low-touch nurture.],
  [0481b169-304f-4814-85bf-65af5688b954], [55.1%], [warm], [Send a targeted follow-up within 48 hours.],
  [435cb827-db56-4301-9f5e-14e1be5a9ffd], [39.9%], [nurture], [Keep in low-touch nurture.],
  [f8c8282b-33dd-4558-89c1-9eeb4650b433], [10.3%], [nurture], [Keep in low-touch nurture.],
)

#v(8pt)
#text(size: 12pt, weight: "bold")[Counterfactual Playbook]
- Nurture lead at p=0.29. DiCE counterfactual paths to conversion (behavioral changes only):
- CF #1: time_on_site_seconds: 54.0 -> 1906.0, page_views_per_visit: 6.5 -> 5.2 => probability 0.29 -> 0.78
- CF #2: time_on_site_seconds: 54.0 -> 828.0 => probability 0.29 -> 0.76
- CF #3: time_on_site_seconds: 54.0 -> 1704.0 => probability 0.29 -> 0.81

#v(8pt)
#text(size: 12pt, weight: "bold")[Model Risk Notes]

- Reported ROC-AUC is measured on a held-out split.
- Do not use post-sale fields, manually assigned sales outcomes, or future
  activity columns as model inputs.
- Counterfactual guidance should not change immutable or protected customer attributes.
