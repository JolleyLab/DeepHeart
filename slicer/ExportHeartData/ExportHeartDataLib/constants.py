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
  "tricuspid": ["sh-leaflet-tv-a", "sh-leaflet-tv-p", "sh-leaflet-tv-s"],
  "cavc": ["sh-leaflet-cavc-sb", "sh-leaflet-cavc-rm", "sh-leaflet-cavc-ib", "sh-leaflet-cavc-lm"],
  "lavv": ["sh-leaflet-lavv-lm", "sh-leaflet-lavv-sb", "sh-leaflet-lavv-ib"]
}


LEAFLET_ORDER_CODE_MEANINGS = {
  "mitral": ["mitral anterior leaflet", "mitral posterior leaflet"],
  "tricuspid": ["tricuspid anterior leaflet", "tricuspid posterior leaflet", "tricuspid septal leaflet"],
  "cavc": ["cavc superior bridging leaflet", "cavc right mural leaflet", "cavc inferior bridging leaflet", "cavc left mural leaflet"],
  "lavv": ["lavv left mural leaflet", "lavv superior bridging leaflet", "lavv inferior bridging leaflet"]
}


def getLandmarkLabelsDefinition(valveType):
  try:
    return LANDMARK_LABELS[valveType.lower()]
  except KeyError:
    raise ValueError("valve type %s not supported" % valveType)