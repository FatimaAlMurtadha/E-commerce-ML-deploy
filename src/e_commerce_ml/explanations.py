"""Human-readable, rule-based explanations for session predictions."""


def explain_prediction(
    probability: float,
    num_clicks: int,
    num_carts: int,
    num_events: int,
    num_unique_items: int,
) -> dict:
    """Translate a model score and observable session signals into guidance.

    The wording describes signals associated with the prediction; it does not
    claim that any individual feature caused the customer outcome.
    """
    if probability >= 0.66:
        risk_level = "high"
        summary = "This session shows strong purchase intent."
        company_action = "Prioritize this customer for helpful follow-up, but avoid unnecessary discounts."
        customer_message = "Your selected products are still available. Complete your order when you are ready."
    elif probability >= 0.33:
        risk_level = "medium"
        summary = "The session shows some purchase intent, but the outcome is uncertain."
        company_action = "Test a personalized reminder or a low-cost benefit such as free shipping."
        customer_message = "Still deciding? Take another look at the products you explored and see what suits you best."
    else:
        risk_level = "low"
        summary = "The session currently shows limited purchase intent."
        company_action = "Do not spend a large incentive yet; improve product discovery and consider a later reminder."
        customer_message = "We found products you may like. Come back anytime to continue exploring."

    reasons = []
    if num_carts > 0:
        reasons.append(f"The customer added {num_carts} item(s) to the cart, a direct intent signal.")
    else:
        reasons.append("No items have been added to the cart yet.")

    if num_clicks >= 10:
        reasons.append(f"The customer made {num_clicks} clicks, showing active browsing.")
    elif num_clicks > 0:
        reasons.append(f"The customer made {num_clicks} click(s), indicating early exploration.")
    else:
        reasons.append("There is no click activity in the session yet.")

    if num_unique_items >= 5:
        reasons.append(f"The customer viewed {num_unique_items} unique items, suggesting broad interest.")
    elif num_unique_items > 0:
        reasons.append(f"The customer viewed {num_unique_items} unique item(s).")

    reasons.append(f"The session contains {num_events} tracked clicks and cart actions in total.")

    return {
        "risk_level": risk_level,
        "summary": summary,
        "reasons": reasons,
        "company_action": company_action,
        "customer_message": customer_message,
    }