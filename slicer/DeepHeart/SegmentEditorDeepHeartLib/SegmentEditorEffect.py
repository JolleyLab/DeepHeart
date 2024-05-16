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
  0: "%p%: Idle",
  10: "%p%: Initializing",
  25: "%p%: Data Preparation",
  50: "%p%: Sending Data",
  75: "%p%: Running Inference",
  90: "%p%: Importing Results",
  100: "%p%: Done"
}


APPLICATION_NAME = 'DeepHeart'


class SegmentEditorEffect(AbstractScriptedSegmentEditorEffect):
  """This effect uses Watershed algorithm to partition the input volume"""

  @property
  def serverUrl(self):
      serverUrl = self.ui.serverComboBox.currentText
      return serverUrl

  def __init__(self, scriptedEffect):
    scriptedEffect.name = APPLICATION_NAME
    scriptedEffect.perSegment = False # this effect operates on all segments at once (not on a single selected segment)
    scriptedEffect.requireSegments = False # this effect requires segment(s) existing in the segmentation
    AbstractScriptedSegmentEditorEffect.__init__(self, scriptedEffect)

    if (slicer.app.majorVersion >= 5) or (slicer.app.majorVersion >= 4 and slicer.app.minorVersion >= 11):
      scriptedEffect.requireSegments = False

    self.moduleName = "SegmentEditorDeepHeart"
    self.logic = DeepHeartLogic(progressCallback=self.updateProgress)

  def resourcePath(self, filename):
    scriptedModulesPath = os.path.dirname(slicer.util.modulePath(self.moduleName))
    return os.path.join(scriptedModulesPath, 'Resources', filename)

  def clone(self):
    # It should not be necessary to modify this method
    import qSlicerSegmentationsEditorEffectsPythonQt as effects
    clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
    clonedEffect.setPythonSource(__file__.replace('\\','/'))
    return clonedEffect

  def icon(self, name=f"{APPLICATION_NAME}.png"):
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
    self.ui.dhServerComboBox.currentText = settings.value(f"{APPLICATION_NAME}/serverUrl", "http://reslnjolleyweb01.research.chop.edu:8894")

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

    self.ui.refreshServerInfoButton.clicked.connect(self.onClickFetchInfo)
    self.ui.dhServerComboBox.connect("currentIndexChanged(int)", self.onClickFetchInfo)
    self.ui.dhModelSelector.connect("currentIndexChanged(int)", self.onSegmentationModelSelected)
    self.ui.runButton.connect("clicked(bool)", self.onClickSegmentation)

    self._heartValveSelection = dict()
    self.updateServerUrlGUIFromSettings()

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
    else:
      self._updateModelSelector(self.ui.dhModelSelector, "DeepHeartSegmentation", valveModel.getValveType())

  def saveServerUrl(self):
    settings = qt.QSettings()
    serverUrl = self.ui.dhServerComboBox.currentText
    settings.setValue(f"{APPLICATION_NAME}/serverUrl", serverUrl)
    self._updateServerHistory(serverUrl)
    
    self.updateServerUrlGUIFromSettings()

  def _updateServerHistory(self, serverUrl):
    settings = qt.QSettings()
    serverUrlHistory = settings.value(f"{APPLICATION_NAME}/serverUrlHistory")
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
    settings.setValue(f"{APPLICATION_NAME}/serverUrlHistory", ";".join(serverUrlHistory))

  def updateServerUrlGUIFromSettings(self):
    # Save current server URL to the top of history
    settings = qt.QSettings()
    serverUrlHistory = settings.value(f"{APPLICATION_NAME}/serverUrlHistory")

    wasBlocked = self.ui.dhServerComboBox.blockSignals(True)
    self.ui.dhServerComboBox.clear()
    if serverUrlHistory:
      self.ui.dhServerComboBox.addItems(serverUrlHistory.split(";"))
    self.ui.dhServerComboBox.setCurrentText(settings.value(f"{APPLICATION_NAME}/serverUrl"))
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
    self.ui.dhProgressBar.setFormat(PROGRESS_VALUES[value])
    if value == 100: # reset progressbar after 10sec
      qt.QTimer.singleShot(10000, lambda: self.ui.dhProgressBar.reset())
    slicer.app.processEvents()

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
      self.updateProgress(0)
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

    self.updateProgress(10)

    with TemporaryDirectory(dir=slicer.app.temporaryPath) as temp_dir:
      result_file = None
      try:
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        result_file = self.logic.infer(heartValves, serverUrl, modelName, temp_dir)
        self.updateProgress(90)
        self.importInferenceResults(modelName, result_file, segmentationNode)
        self.updateProgress(100)
      except Exception as exc:
        import traceback
        traceback.print_exc()
        slicer.util.errorDisplay(
          "Failed to run inference in MONAI Label Server", detailedText=traceback.format_exc()
        )
        self.updateProgress(0)
      finally:
        qt.QApplication.restoreOverrideCursor()
        if result_file and os.path.exists(result_file):
          os.unlink(result_file)

  def importInferenceResults(self, modelName, result_file, segmentationNode):
    labelNode = slicer.util.loadLabelVolume(str(result_file))
    terminologyName = slicer.modules.terminologies.logic().LoadTerminologyFromFile(HeartValveLib.getTerminologyFile())
    segmentation = segmentationNode.GetSegmentation()
    numberOfExistingSegments = segmentation.GetNumberOfSegments()
    slicer.vtkSlicerSegmentationsModuleLogic.ImportLabelmapToSegmentationNode(
      labelNode, segmentationNode, terminologyName
    )
    slicer.mrmlScene.RemoveNode(labelNode)
    numberOfAddedSegments = segmentation.GetNumberOfSegments() - numberOfExistingSegments
    addedSegmentIds = [
      segmentation.GetNthSegmentID(numberOfExistingSegments + i) for i in range(numberOfAddedSegments)
    ]
    model = self.logic.models[modelName]
    for segmentId, segmentName in zip(addedSegmentIds, model["labels"]):
      segment = segmentation.GetSegment(segmentId)
      segment.SetName(segmentName)

      segmentTerminologyTag, segType = getTerminologyTagAndTypeBySegmentName(terminologyName, segmentName)
      if segmentTerminologyTag:
        segment.SetTag(segment.GetTerminologyEntryTagName(), segmentTerminologyTag)
        segment.SetColor(np.array(segType.GetRecommendedDisplayRGBValue()) / 255.0)
      else:
        logging.info(f"No terminology entry found for segment with name {segmentName}. Using default colors.")

  def _updateModelSelector(self, selector, modelType, valveType):
    self.ui.statusLabel.plainText = ''
    wasSelectorBlocked = selector.blockSignals(True)
    selector.clear()
    for model_name, model in self.logic.models.items():
      if model["type"] == modelType and model["valve_type"] == valveType:
        selector.addItem(model_name)
        selector.setItemData(selector.count - 1, model["description"], qt.Qt.ToolTipRole)
    selector.blockSignals(wasSelectorBlocked)
    self.onSegmentationModelSelected()
    if not selector.count:
      msg = f"No eligible models were found for current valve type: {valveType}.\t\n"
    else:
      msg = f"Found {selector.count} eligible models for current valve type: {valveType}.\t\n"
    msg += "-----------------------------------------------------\t\n"
    msg += f"Total Models Available:  {len(self.logic.models)}\t\n"
    msg += "-----------------------------------------------------\t\n"
    self.ui.statusLabel.plainText = msg


class DeepHeartLogic(ScriptedLoadableModuleLogic):

  def __init__(self, progressCallback=None):
    ScriptedLoadableModuleLogic.__init__(self)
    self.monaiLabelLogic = MONAILabelLogic()
    self.models = OrderedDict()
    self.progressCallback = progressCallback

  def fetchInfo(self, serverUrl):
    self.models = OrderedDict()
    try:
      start = time.time()
      self.monaiLabelLogic.setServer(serverUrl)
      info = self.monaiLabelLogic.info()
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

  def infer(self, heartValves, serverUrl, modelName, temp_dir):
    self.progressCallback(25)

    image_in = self.preprocessSceneData(heartValves, modelName, temp_dir)
    self.progressCallback(50)
    client = MONAILabelClient(server_url=serverUrl)
    sessionId = client.create_session(image_in)["session_id"]

    self.progressCallback(75)
    result_file, params = client.infer(model=modelName,
                                       image_id=image_in,
                                       params={},
                                       session_id=sessionId)
    return result_file

  def preprocessSceneData(self, heartValves, modelName, temp_dir):
    """
    :param heartValves: heart valve model to use as reference frame
    :param modelName: name of the selected model
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


def getTerminologyTagAndTypeBySegmentName(terminologyName, name):
  """ Returns serialized terminology tag and in addition the terminology type object for color """
  tlogic = slicer.modules.terminologies.logic()
  category = slicer.vtkSlicerTerminologyCategory()
  for categoryIdx in range(tlogic.GetNumberOfCategoriesInTerminology(terminologyName)):
    tlogic.GetNthCategoryInTerminology(terminologyName, categoryIdx, category)
    termType = slicer.vtkSlicerTerminologyType()
    for typeIdx in range(tlogic.GetNumberOfTypesInTerminologyCategory(terminologyName, category)):
      tlogic.GetNthTypeInTerminologyCategory(terminologyName, category, typeIdx, termType)
      if termType.GetCodeMeaning() == name:
        segmentTerminologyTag = tlogic.SerializeTerminologyEntry(
          terminologyName,
          category.GetCodeValue(), category.GetCodingSchemeDesignator(), category.GetCodeMeaning(),
          termType.GetCodeValue(), termType.GetCodingSchemeDesignator(), termType.GetCodeMeaning(),
          "", "", "", "", "", "", "", "", "", ""
        )
        return segmentTerminologyTag, termType
  return None, None
