REGISTRY={("M","FT"):(3.280839895,"value * 3.280839895"),("MM","IN"):(1/25.4,"value / 25.4"),("PERC","V/V"):(.01,"value * 0.01")}

def conversion_proposal(source_unit,destination_unit,depth_start,depth_end):
    key=(source_unit.upper(),destination_unit.upper())
    if key not in REGISTRY:raise ValueError(f"No controlled conversion from {source_unit} to {destination_unit}")
    factor,formula=REGISTRY[key]
    return {"source_unit":key[0],"destination_unit":key[1],"factor":factor,"formula":formula,"converted_interval":[depth_start,depth_end],"original_values_preserved":True,"automatic":False,"requires_review":True}

def apply_reviewed_conversion(values,proposal,reviewed=False):
    if not reviewed:raise PermissionError("Unit conversion requires explicit review")
    return values*proposal["factor"]
