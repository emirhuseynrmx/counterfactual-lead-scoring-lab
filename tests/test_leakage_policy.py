from counterfactual_lead_scoring_lab.data import FEATURE_COLUMNS


def test_feature_columns_do_not_include_post_conversion_fields() -> None:
    forbidden = {
        "converted",
        "payment_status",
        "closed_won",
        "sales_outcome",
        "post_demo_revenue",
    }

    assert forbidden.isdisjoint(FEATURE_COLUMNS)
