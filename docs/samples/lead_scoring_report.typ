#set page(margin: 16mm)
#set text(font: "Arial", size: 10pt)

#text(size: 18pt, weight: "bold")[Counterfactual Lead Scoring Lab]

Model-training report for lead scoring with SHAP/DiCE planning.

#grid(columns: (1fr, 1fr, 1fr), gutter: 8pt)[
  #block(fill: rgb("#f3f6fb"), radius: 4pt, inset: 8pt)[Rows\ #text(size: 18pt, weight: "bold")[1200]]
][
  #block(fill: rgb("#f3f6fb"), radius: 4pt, inset: 8pt)[Conversion\ #text(size: 18pt, weight: "bold")[0.3417]]
][
  #block(fill: rgb("#f3f6fb"), radius: 4pt, inset: 8pt)[ROC-AUC\ #text(size: 18pt, weight: "bold")[0.9338]]
]

#v(10pt)
#text(size: 12pt, weight: "bold")[Top Lead Drivers]

#table(columns: (1fr, 1fr), [Feature], [Importance],
  [time_on_site_seconds], [0.25825],
  [activity_score], [0.20406],
  [occupation_unknown], [0.17521],
  [lead_origin_lead_add_form], [0.15569],
  [profile_score], [0.07021],
  [occupation_working_professional], [0.03312],
  [page_views_per_visit], [0.01879],
  [total_visits], [0.01832],
)

#v(8pt)
#text(size: 12pt, weight: "bold")[Counterfactual Playbook]
- Start counterfactual review with time_on_site_seconds; test whether changing it is actionable.
- Keep immutable fields fixed and test only behaviors a sales or marketing team can influence.
- Use hot leads for direct sales follow-up, warm leads for nurture, and low leads for automation.
