"""Shadow comparisons that cannot replace production-selected output."""

def shadow_result(production_output, candidate_output, production_model: str, shadow_model: str) -> dict:
    return {"selected_output": production_output, "shadow_output": candidate_output,
            "metadata": {"production_model": production_model, "shadow_model": shadow_model, "production_selected": True}}
