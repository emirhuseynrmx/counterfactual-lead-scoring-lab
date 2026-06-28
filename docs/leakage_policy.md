# Leakage Policy

This repo trains on lead attributes and engagement fields available before conversion.

Allowed feature families:

- lead source and origin
- website visits and page views
- time spent on site
- occupation and city
- activity and profile scores
- email opt-out flag

The model must not use fields that are downstream of conversion:

- payment status after conversion
- sales outcome notes
- closed-won labels
- post-demo revenue
- manually assigned final lead quality tags created after sales review

The target column, `converted`, is used only as the supervised label.
