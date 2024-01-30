APPLICATION_NAME = "NormalizeAndExport"


PHASES_DIRECTORY_MAPPING = {
  "MS": "mid-systolic",
  "MD": "mid-diastolic",
  "ES": "end-systolic",
  "ED": "end-diastolic",
  "CS": "custom-systolic",
  "CT": "custom-transition",
  "CD": "custom-diastolic"
}


LANDMARK_LABELS = {
  "mitral": ['PM', 'A', 'AL', 'P'],
  "tricuspid": ['A', 'L', 'P', 'S'],
  "lavv": ['ALC', 'SIC', 'PMC'],
  "cavc": ['MP', 'L', 'MA', 'R']
}

LEAFLET_ORDER = {
  # NB: deprecated
  "mitral": ['anterior', 'posterior'],
  "tricuspid": ['anterior', 'posterior', 'septal'],
  "cavc": ['superior', 'right', 'inferior', 'left']
}


LEAFLET_ORDER_CODE_VALUES = {
  "mitral": ["sh-leaflet-mv-a", "sh-leaflet-mv-p"],
  "tricuspid": ["sh-leaflet-tcv-a", "sh-leaflet-tcv-p", "sh-leaflet-tcv-s"],
  "cavc": ["sh-leaflet-cavc-sb", "sh-leaflet-cavc-rm", "sh-leaflet-cavc-ib", "sh-leaflet-cavc-lm"],
  "lavv": ["sh-leaflet-cavc-lm", "sh-leaflet-cavc-sb", "sh-leaflet-cavc-ib"]
}


LEAFLET_ORDER_CODE_MEANINGS = {
  "mitral": ["mitral anterior leaflet", "mitral posterior leaflet"],
  "tricuspid": ["tricuspid anterior leaflet", "tricuspid posterior leaflet", "tricuspid septal leaflet"],
  "cavc": ["superior bridging leaflet", "right mural leaflet", "inferior bridging leaflet", "left mural leaflet"],
  "lavv": ["left mural leaflet", "superior bridging leaflet", "inferior bridging leaflet"]
}


def getLandmarkLabelsDefinition(valveType):
  try:
    return LANDMARK_LABELS[valveType.lower()]
  except KeyError:
    raise ValueError("valve type %s not supported" % valveType)