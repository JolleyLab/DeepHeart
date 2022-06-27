from slicer.ScriptedLoadableModule import *

class ExportHeartData(ScriptedLoadableModule):
  """
  This class is the 'hook' for slicer to detect and recognize the plugin
  as a loadable scripted module
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    parent.title = "ExportHeartData"
    parent.categories = ["Cardiac"]
    parent.dependencies = ["ValveAnnulusAnalysis"]
    parent.contributors = ["Christian Herz (CHOP), Andras Lasso (PerkLab), Matt Jolley (UPenn)"]
    parent.hidden = True
    parent.helpText = """
    ExportHeartData module normalization and export of valve structures in preparation for deep learning
    """
    parent.acknowledgementText = """
    This file was originally developed by Christian Herz (CHOP).
    """