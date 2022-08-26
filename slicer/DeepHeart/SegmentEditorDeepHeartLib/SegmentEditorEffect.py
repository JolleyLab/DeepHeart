import qt, ctk, vtk
from SegmentEditorEffects import *
from slicer.ScriptedLoadableModule import *
import slicer
import os
import numpy as np
import logging
from collections import OrderedDict
from tempfile import TemporaryDirectory, NamedTemporaryFile

import HeartValveLib
from MONAILabel import MONAILabelLogic
from MONAILabelLib import MONAILabelClient
from ExportHeartDataLib.export import Exporter, ExportItem

import time


PROGRESS_VALUES = {
  0: "%p%: Initializing",
  25: "%p%: Data Preparation",
  50: "%p%: Sending Data",
  75: "%p%: Running Inference",
  100: "%p%: Importing Results"
}

PARAM_DEFAULTS = {

}


class SegmentEditorEffect(AbstractScriptedSegmentEditorEffect):
  """This effect uses Watershed algorithm to partition the input volume"""

  @property
  def serverUrl(self):
      serverUrl = self.ui.serverComboBox.currentText
      return serverUrl

  def __init__(self, scriptedEffect):
    scriptedEffect.name = 'DeepHeart'
    scriptedEffect.perSegment = False # this effect operates on all segments at once (not on a single selected segment)
    scriptedEffect.requireSegments = False # this effect requires segment(s) existing in the segmentation
    AbstractScriptedSegmentEditorEffect.__init__(self, scriptedEffect)

    if (slicer.app.majorVersion >= 5) or (slicer.app.majorVersion >= 4 and slicer.app.minorVersion >= 11):
      scriptedEffect.requireSegments = False

    self.moduleName = "SegmentEditorDeepHeart"
    self.logic = DeepHeartLogic()

  def resourcePath(self, filename):
    scriptedModulesPath = os.path.dirname(slicer.util.modulePath(self.moduleName))
    return os.path.join(scriptedModulesPath, 'Resources', filename)

  def clone(self):
    # It should not be necessary to modify this method
    import qSlicerSegmentationsEditorEffectsPythonQt as effects
    clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
    clonedEffect.setPythonSource(__file__.replace('\\','/'))
    return clonedEffect

  def icon(self, name="DeepHeart.png"):
    # It should not be necessary to modify this method
    iconPath = self.resourcePath(f"Icons/{name}")
    if os.path.exists(iconPath):
      return qt.QIcon(iconPath)
    return qt.QIcon()

  def helpText(self):
    return """
    The DeepHeart SegmentEditor effect provides access to rapid segmentation of heart valve leaflets.
    ----------------------------------------
    Based on MONAILabel (client/server)
    Depends on SlicerHeart and MONAILabel extension
    """

  def setupOptionsFrame(self):
    uiWidget = slicer.util.loadUI(self.resourcePath(f"UI/{self.moduleName}.ui"))
    self.scriptedEffect.addOptionsWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)
    self.ui.refreshServerInfoButton.setIcon(self.icon("refresh-icon.png"))

    settings = qt.QSettings()
    self.ui.dhServerComboBox.currentText = settings.value("DeepHeart/serverUrl", "http://127.0.0.1:8000")
    self.ui.dhProgressBar.hide()
    self.ui.statusLabel.hide()

    self.ui.refreshServerInfoButton.clicked.connect(self.onClickFetchInfo)
    self.ui.dhServerComboBox.connect("currentIndexChanged(int)", self.onClickFetchInfo)
    self.ui.dhModelSelector.connect("currentIndexChanged(int)", self.onSegmentationModelSelected)
    self.ui.runButton.connect("clicked(bool)", self.onClickSegmentation)

    self._heartValveSelection = dict()
    self.updateServerUrlGUIFromSettings()

  def initializeParameterNode(self):
    if self._parameterNode is not None:
      self._parameterNode = \
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self.setParameterNode(self.logic.getParameterNode())

  def setParameterNode(self, inputParameterNode):
    self._parameterNode = inputParameterNode

    if self._parameterNode is not None:
      self.logic.setDefaultParameters(inputParameterNode)
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    self.updateGUIFromParameterNode()

  def onClickFetchInfo(self):
    self.saveServerUrl()
    self.ui.dhAppComboBox.clear()
    self.ui.dhModelSelector.clear()
    self.ui.runButton.setEnabled(False)

    serverUrl = self.ui.dhServerComboBox.currentText
    info = self.logic.fetchInfo(serverUrl)
    if not info:
      return

    self.ui.dhAppComboBox.addItem(info.get("name", ""))

    from HeartValveLib.helpers import getValveModelForSegmentationNode
    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
    valveModel = getValveModelForSegmentationNode(segmentationNode)

    if not valveModel:
      self.ui.statusLabel.plainText = """
      No associated HeartValve (SlicerHeart) could be found for selected segmentation node.
      Please use the SlicerHeart infrastructure to be able to continue. 
      """
      self.ui.statusLabel.show()
      qt.QTimer.singleShot(10000, lambda: self.ui.statusLabel.hide())
    else:
      self._updateModelSelector(self.ui.dhModelSelector, "DeepHeartSegmentation", valveModel.getValveType())

  def saveServerUrl(self):
    settings = qt.QSettings()
    serverUrl = self.ui.dhServerComboBox.currentText
    settings.setValue("DeepHeart/serverUrl", serverUrl)
    self._updateServerHistory(serverUrl)
    
    self.updateServerUrlGUIFromSettings()

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()

    # TODO: update info here

    self._parameterNode.EndModify(wasModified)

  def _updateServerHistory(self, serverUrl):
    settings = qt.QSettings()
    serverUrlHistory = settings.value("DeepHeart/serverUrlHistory")
    if serverUrlHistory:
      serverUrlHistory = serverUrlHistory.split(";")
    else:
      serverUrlHistory = []
    try:
      serverUrlHistory.remove(serverUrl)
    except ValueError:
      pass
    serverUrlHistory.insert(0, serverUrl)
    serverUrlHistory = serverUrlHistory[:10]  # keep up to first 10 elements
    settings.setValue("DeepHeart/serverUrlHistory", ";".join(serverUrlHistory))

  def updateServerUrlGUIFromSettings(self):
    # Save current server URL to the top of history
    settings = qt.QSettings()
    serverUrlHistory = settings.value("DeepHeart/serverUrlHistory")

    wasBlocked = self.ui.dhServerComboBox.blockSignals(True)
    self.ui.dhServerComboBox.clear()
    if serverUrlHistory:
      self.ui.dhServerComboBox.addItems(serverUrlHistory.split(";"))
    self.ui.dhServerComboBox.setCurrentText(settings.value("DeepHeart/serverUrl"))
    self.ui.dhServerComboBox.blockSignals(wasBlocked)

  def createCursor(self, widget):
    # Turn off effect-specific cursor for this effect
    return slicer.util.mainWindow().cursor

  def updateGUIFromMRML(self):
    pass

  def updateMRMLFromGUI(self):
    pass

  def updateProgress(self, value):
    self.ui.dhProgressBar.setValue(value)
    self.ui.dhProgressBar.setStyleSheet(
     """
      QdhProgressBar {
        text-align: center;
      }
      QdhProgressBar::chunk {
        background-color: qlineargradient(x0: 0, x2: 1, stop: 0 orange, stop:1 green )
      }
      """
    )

    self.ui.dhProgressBar.setFormat(PROGRESS_VALUES[value])
    slicer.app.processEvents()
    if value == 100:
      self.ui.dhProgressBar.hide()
    else:
      self.ui.dhProgressBar.show()

  def onSegmentationModelSelected(self):
    segmentationModelIndex = self.ui.dhModelSelector.currentIndex
    self.ui.runButton.setEnabled(self.ui.dhModelSelector.itemText(segmentationModelIndex) != "")

    # hide all selectors
    for modelName in self._heartValveSelection.keys():
      labels, selectors = self._heartValveSelection[modelName]
      for label, selector in zip(labels, selectors):
        label.hide()
        selector.hide()

    modelName = self.ui.dhModelSelector.currentText
    if not modelName:
      return

    modelConfig = self.logic.models[modelName]["config"]
    cardiacPhases = modelConfig["model_attributes"]["cardiac_phase_frames"]
    valveType = modelConfig["model_attributes"]["valve_type"]
    try:
      labels, selectors = self._heartValveSelection[modelName]
      for label, selector in zip(labels, selectors):
        label.show()
        selector.show()
    except KeyError:
      labels = []
      selectors = []
      for cardiacPhase in cardiacPhases:
        valveLabel = qt.QLabel(f"{cardiacPhase} Phase")
        valveSelector = self._createHeartValveNodeSelector()
        labels.append(valveLabel)
        selectors.append(valveSelector)
        self.ui.valveSelectionFrame.layout().addRow(valveLabel, valveSelector)
        self._heartValveSelection[modelName] = (labels, selectors)

    for idx, (label, selector) in enumerate(zip(labels, selectors)):
      if selector.currentNode() is not None:
        continue
      from HeartValveLib.helpers import getSpecificHeartValveModelNodesMatchingPhaseAndType
      valveModels = getSpecificHeartValveModelNodesMatchingPhaseAndType([cardiacPhases[idx]], valveType)
      if valveModels:
        selector.setCurrentNode(valveModels[0].heartValveNode)

  def _createHeartValveNodeSelector(self):
    valveSelector = slicer.qMRMLNodeComboBox()
    valveSelector.nodeTypes = ["vtkMRMLScriptedModuleNode"]
    valveSelector.setNodeTypeLabel("HeartValve", "vtkMRMLScriptedModuleNode")
    valveSelector.addAttribute("vtkMRMLScriptedModuleNode", "ModuleName", "HeartValve")
    valveSelector.addEnabled = False
    valveSelector.removeEnabled = True
    valveSelector.noneEnabled = True
    valveSelector.showHidden = True  # scripted module nodes are hidden by default
    valveSelector.renameEnabled = True
    valveSelector.setMRMLScene(slicer.mrmlScene)
    valveSelector.setToolTip("Select heart valve")
    return valveSelector

  def onClickSegmentation(self):
    try:
      import nibabel as nib
    except ImportError:
      slicer.util.pip_install("nibabel")

    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
    modelName = self.ui.dhModelSelector.currentText
    serverUrl = self.ui.dhServerComboBox.currentText

    from HeartValveLib.helpers import getValveModelForSegmentationNode
    valveModel = getValveModelForSegmentationNode(segmentationNode)
    if not valveModel:
      logging.warning("No associated HeartValve (SlicerHeart) could be found for selected segmentation node")
      return

    _, selectors = self._heartValveSelection[modelName]
    heartValves = [selector.currentNode() for selector in selectors]
    if any(hv is None for hv in heartValves):
      slicer.util.errorDisplay("Missing phases. Please check the inputs.")
      self.updateProgress(100)
      return

    from HeartValveLib.helpers import getValvePhaseShortName
    modelConfig = self.logic.models[modelName]["config"]
    cardiacPhases = modelConfig["model_attributes"]["cardiac_phase_frames"]
    differentPhaseSelected = []
    for idx, selector in enumerate(selectors):
    # check if selected phase is right phase
      vm = HeartValveLib.HeartValves.getValveModel(selector.currentNode())
      selectedPhaseShortname = getValvePhaseShortName(vm)
      differentPhaseSelected.append(selectedPhaseShortname != cardiacPhases[idx])

    if any(different is True for different in differentPhaseSelected):
      if not slicer.util.confirmYesNoDisplay(
              "Found mismatch between assigned phases and selected heart valve nodes.\n\n"
              "Do you want to continue with the currently selected heart valve(s)?"):
        return

    self.updateProgress(0)

    with TemporaryDirectory() as temp_dir:
      result_file = None
      try:
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        self.updateProgress(25)

        image_in = self.logic.preprocessSceneData(heartValves, modelName, temp_dir)
        result_file = self.logic.infer(image_in, serverUrl, modelName, progressCallback=self.updateProgress)
        if result_file:
          labelNode = slicer.util.loadLabelVolume(str(result_file))

          tlogic = slicer.modules.terminologies.logic()
          terminologyName = tlogic.LoadTerminologyFromFile(HeartValveLib.getTerminologyFile())
          terminologyEntry = slicer.vtkSlicerTerminologyEntry()
          terminologyEntry.SetTerminologyContextName(terminologyName)

          segmentation = segmentationNode.GetSegmentation()
          numberOfExistingSegments = segmentation.GetNumberOfSegments()
          slicer.vtkSlicerSegmentationsModuleLogic.ImportLabelmapToSegmentationNode(labelNode,
                                                                                    segmentationNode,
                                                                                    terminologyName)
          slicer.mrmlScene.RemoveNode(labelNode)

          numberOfAddedSegments = segmentation.GetNumberOfSegments() - numberOfExistingSegments
          addedSegmentIds = [
            segmentation.GetNthSegmentID(numberOfExistingSegments + i) for i in range(numberOfAddedSegments)
          ]
          model = self.logic.models[modelName]
          for segmentId, segmentName in zip(addedSegmentIds, model["labels"]):
            segment = segmentation.GetSegment(segmentId)
            segment.SetName(segmentName)
            segType = getSegmentTerminologyByName(terminologyName, segmentName)
            if not segType:
              logging.info(f"No terminology entry found for segment with name {segmentName}. Using default colors.")
            segment.SetColor(np.array(segType.GetRecommendedDisplayRGBValue()) / 255.0)

            # TODO: apply SlicerHeart terminology!
            tagName = slicer.vtkSegment.GetTerminologyEntryTagName()

      except Exception as exc:
        import traceback
        traceback.print_exc()
        slicer.util.errorDisplay(
          "Failed to run inference in MONAI Label Server", detailedText=traceback.format_exc()
        )
        self.updateProgress(100)
      finally:
        qt.QApplication.restoreOverrideCursor()
        if result_file and os.path.exists(result_file):
          os.unlink(result_file)

  def _updateModelSelector(self, selector, modelType, valveType):
      self.ui.statusLabel.plainText = ''
      wasSelectorBlocked = selector.blockSignals(True)
      selector.clear()
      num_eligible = 0
      for model_name, model in self.logic.models.items():
          if model["type"] == modelType and model["valve_type"] == valveType:
              selector.addItem(model_name)
              selector.setItemData(selector.count - 1, model["description"], qt.Qt.ToolTipRole)
              num_eligible += 1
      selector.blockSignals(wasSelectorBlocked)
      self.onSegmentationModelSelected()
      if not num_eligible:
        msg = f"No eligible models were found for current valve type: {valveType}.\t\n"
      else:
        msg = f"Found {num_eligible} eligible models for current valve type: {valveType}.\t\n"
      msg += "-----------------------------------------------------\t\n"
      msg += f"Total Models Available:  {len(self.logic.models)}\t\n"
      msg += "-----------------------------------------------------\t\n"

      self.ui.statusLabel.plainText = msg
      self.ui.statusLabel.show()
      qt.QTimer.singleShot(10000, lambda: self.ui.statusLabel.hide())


class DeepHeartLogic(ScriptedLoadableModuleLogic):

  @staticmethod
  def setDefaultParameters(parameterNode, defaults=PARAM_DEFAULTS):
    for paramName, paramDefaultValue in defaults.items():
      if not parameterNode.GetParameter(paramName):
        parameterNode.SetParameter(paramName, str(paramDefaultValue))

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self.logic = MONAILabelLogic()
    self.models = OrderedDict()

  def fetchInfo(self, serverUrl):
    self.models = OrderedDict()
    try:
      start = time.time()
      self.logic.setServer(serverUrl)
      info = self.logic.info()
      logging.info("Time consumed by fetch info: {0:3.1f}".format(time.time() - start))
      self._updateModels(info["models"])
      return info
    except Exception as exc:
      print(exc)
      import traceback
      slicer.util.errorDisplay(
        "Failed to fetch models from remote server. "
        "Make sure server address is correct and <server_uri>/info/ "
        "is accessible in browser",
        detailedText=traceback.format_exc(),
      )

  def _updateModels(self, models):
    self.models.clear()
    model_count = {}
    for k, v in models.items():
      model_type = v.get("type", "segmentation")
      model_count[model_type] = model_count.get(model_type, 0) + 1

      logging.debug("{} = {}".format(k, model_type))
      self.models[k] = v

  def infer(self, image_in, serverUrl, modelName, progressCallback):
    progressCallback(50)
    client = MONAILabelClient(server_url=serverUrl)
    sessionId = client.create_session(image_in)["session_id"]

    progressCallback(75)
    result_file, params = client.infer(model=modelName,
                                       image_in=image_in,
                                       params={},
                                       session_id=sessionId)

    progressCallback(100)

    return result_file

  def preprocessSceneData(self, heartValves, modelName, temp_dir):
    """
    :param heartValves: heart valve model to use as reference frame
    :param modelConfig: model configuration as stored on the MONAI server
    :param temp_dir: directory to temporarily export data to
    :return: prepared inference model input volume (4D)
    """
    valveModels = [HeartValveLib.HeartValves.getValveModel(hv) for hv in heartValves]

    start = time.time()
    modelConfig = self.models[modelName]["config"]
    exportSettings = modelConfig["model_attributes"]
    exported_dict = getInferenceExport(valveModels, output_directory=temp_dir, **exportSettings)

    volumes = [exported_dict[key][0] for key in modelConfig["export_keys"]]
    image_in = _stackVolumes(volumes, temp_dir)
    logging.info("Time consumed to preprocess data: {0:3.1f}".format(time.time() - start))
    return image_in


def getInferenceExport(valveModels,
                       valve_type,
                       cardiac_phase_frames,
                       output_directory=None,
                       voxel_spacing=None,
                       volume_dimensions=None,
                       landmark_labels=None,
                       landmark_label_phases=None,
                       annulus_contour_label=True,
                       annulus_phases=None):

  """
     :return: returning a dictionary with the output types and file paths
  """
  try:
    from ExportHeartDataLib.summary import ExportSummary
    ExportItem.setExportSummarizer(ExportSummary())
    ExportItem.prefix = "temp"

    from pathlib import Path
    ExportItem.outputDir = Path(output_directory)

    from HeartValveLib.helpers import getFirstValveModelNodeMatchingPhaseAndType, getValvePhaseShortName
    from ExportHeartDataLib.utils import cloneMRMLNode
    referenceValveModel = valveModels[0]
    ExportItem.probeToRasTransform = cloneMRMLNode(referenceValveModel.getProbeToRasTransformNode())
    Exporter.initializeReferenceVolumeNode(referenceValveModel, voxel_spacing, volume_dimensions)

    assert ExportItem.requiredAttributesSet()

    from ExportHeartDataLib.items import Annulus, PhaseFrame, LandmarkLabels
    for valveModel, phaseShortName in zip(valveModels, cardiac_phase_frames):
      verifyAndRunItemExport(PhaseFrame(valveModel, valveModel is referenceValveModel, phase=phaseShortName))
      if annulus_contour_label and phaseShortName in annulus_phases:
        verifyAndRunItemExport(Annulus(valveModel, asLabel=True, phase=phaseShortName))
      if landmark_labels and phaseShortName in landmark_label_phases:
        verifyAndRunItemExport(LandmarkLabels(valveModel, landmark_labels, phase=phaseShortName))
  except Exception as exc:
    raise exc
  finally:
    ExportItem.cleanup()

  return ExportItem.exportSummarizer.get_summary()


def verifyAndRunItemExport(item):
  valid, message = item.verify()
  if valid:
    item()
  else:
    raise ValueError(f"{item.__class__.__name__}: {message}")


def _stackVolumes(volumes: list, out_dir: str):
  import nibabel as nib
  affine = nib.load(volumes[0]).affine
  dtype = np.float32
  data = np.stack([nib.load(path).get_fdata().astype(dtype) for path in volumes])
  img = nib.Nifti1Image(data, affine)
  in_file = NamedTemporaryFile(suffix=".nii.gz", dir=out_dir).name
  nib.save(img, in_file)
  return in_file


def getSegmentTerminologyByName(terminologyName, name):
  tlogic = slicer.modules.terminologies.logic()
  cat = slicer.vtkSlicerTerminologyCategory()
  tlogic.GetNthCategoryInTerminology(terminologyName, 0, cat)
  segType = slicer.vtkSlicerTerminologyType()
  for idx in range(tlogic.GetNumberOfTypesInTerminologyCategory(terminologyName, cat)):
    tlogic.GetNthTypeInTerminologyCategory(terminologyName, cat, idx, segType)
    if segType.GetCodeMeaning() == name:
      return segType
  raise None