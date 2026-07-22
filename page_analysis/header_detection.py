from __future__ import annotations
import difflib,re
MNEMONICS={"GR":"Gamma Ray","SP":"Spontaneous Potential","CALI":"Caliper","ILD":"Induction Deep","ILM":"Induction Medium","LLD":"Deep Laterolog","LLS":"Shallow Laterolog","RHOB":"Bulk Density","NPHI":"Neutron Porosity","DT":"Sonic Transit Time","PEF":"Photoelectric Factor","RES":"Resistivity","RXO":"Flushed Zone Resistivity"}
UNITS={"OHMM":"OHMM","OHM M":"OHMM","OHM-M":"OHMM","API UNITS":"API","API":"API","G/CC":"G/C3","G/C3":"G/C3","USEC/FT":"US/F","US/FT":"US/F","INCHES":"IN","IN":"IN","FEET":"FT","FT":"FT","METERS":"M"}
def normalize_mnemonic(raw,threshold=.72):
    cleaned=re.sub(r"[^A-Z0-9]","",str(raw).upper())
    if cleaned in MNEMONICS: match=cleaned; confidence=1.
    else:
        options=difflib.get_close_matches(cleaned,MNEMONICS,1,threshold); match=options[0] if options else cleaned; confidence=.8 if options else .35
    return {"raw_name":raw,"normalized_mnemonic":match,"display_name":MNEMONICS.get(match,raw),"normalization_confidence":confidence,"preserved_unknown":match not in MNEMONICS}
def normalize_unit(raw):
    normalized=UNITS.get(re.sub(r"\s+"," ",str(raw).upper()).strip())
    return {"raw_unit":raw,"normalized_unit":normalized or str(raw).upper(),"confidence":.95 if normalized else .35,"source":"header_ocr"}
