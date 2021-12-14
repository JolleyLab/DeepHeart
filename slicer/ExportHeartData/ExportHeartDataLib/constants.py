APPLICATION_NAME = "NormalizeAndExport"


PHASES_DIRECTORY_MAPPING = {
  "MS": "mid-systolic",
  "MD": "mid-diastolic",
  "ES": "end-systolic",
  "ED": "end-diastolic"
}


LANDMARK_LABELS = {
  "mitral": ['PM', 'A', 'AL', 'P'],
  "tricuspid": ['A', 'L', 'P', 'S'],
  "cavc": ['MP', 'L', 'MA', 'R']
}


LEAFLET_ORDER = {
  "mitral": ['anterior', 'posterior'],
  "tricuspid": ['anterior', 'posterior', 'septal'],
  "cavc": ['superior', 'right', 'inferior', 'left']
}


def getLandmarkLabelsDefinition(valveType):
  try:
    return LANDMARK_LABELS[valveType.lower()]
  except KeyError:
    raise ValueError("valve type %s not supported" % valveType)