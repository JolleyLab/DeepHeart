import slicer

from ExportHeartDataLib.constants import PHASES_DIRECTORY_MAPPING
from ExportHeartDataLib.base import ExportItem
from ExportHeartDataLib.utils import getSegmentationFromAnnulusContourNode, getLabelFromLandmarkPosition, \
  cloneMRMLNode, getResampledScalarVolume
import ExportHeartDataLib.segmentation_utils as SegmentationHelper


class Landmarks(ExportItem):

  LANDMARKS_OUTPUT_DIR_NAME = "landmarks"

  def __init__(self, valveModel):
    super(Landmarks, self).__init__(valveModel)

  def __call__(self):
    logger = self.getLogger()
    landmarksNode = self._valveModel.getAnnulusLabelsMarkupNode()
    outputDirectory = self.outputDir / self.LANDMARKS_OUTPUT_DIR_NAME
    outputFile = outputDirectory / f"{self.prefix}.fcsv"
    outputDirectory.mkdir(exist_ok=True)
    logger.debug(f"Saving landmarks to {outputFile}: {self.saveNode(landmarksNode, outputFile, 'landmarks')}")


class LandmarkLabels(ExportItem):

  def __init__(self, valveModel, landmarkLabels):
    super(LandmarkLabels, self).__init__(valveModel)
    self.landmarkLabels = landmarkLabels if landmarkLabels else []

  def __call__(self):
    logger = self.getLogger()
    for landmarkLabel in self.landmarkLabels:
      landmarkPosition = self._valveModel.getAnnulusMarkupPositionByLabel(landmarkLabel)
      if landmarkPosition is None:
        logger.debug(f"Couldn't find landmark label {landmarkLabel}")
        continue
      labelNode = getLabelFromLandmarkPosition(landmarkLabel, landmarkPosition, self.referenceVolumeNode)
      folderName = f"{PHASES_DIRECTORY_MAPPING[self.phase]}-{landmarkLabel}"
      outputDirectory = self.outputDir / folderName
      outputDirectory.mkdir(exist_ok=True)
      outputFile = outputDirectory / f"{self.prefix}.nii.gz"
      logger.debug(f"Saving landmark {landmarkLabel} to {outputFile}: "
                   f"{self.saveNode(labelNode, outputFile, folderName)}")


class Segmentation(ExportItem):

  SEGMENTATION_OUTPUT_DIR_NAME = "segmentation"

  def __init__(self, valveModel, oneFilePerSegment):
    super(Segmentation, self).__init__(valveModel)
    self._oneFilePerSegment = oneFilePerSegment

  def __call__(self):
    segmentationNode = cloneMRMLNode(self._valveModel.getLeafletSegmentationNode())
    segmentationNode.SetAndObserveTransformNodeID(None)
    SegmentationHelper.deleteNonLeafletSegments(segmentationNode)

    if len(SegmentationHelper.getAllSegmentIDs(segmentationNode)) == 0 or \
       SegmentationHelper.hasEmptySegments(segmentationNode):
      self.getLogger().debug(f"Segmentation for phase {self.phase} empty. Skipping.")
      return
    SegmentationHelper.checkAndSortSegments(segmentationNode, self._valveModel.getValveType())
    # Smoothing
    SegmentationHelper.postProcessSegmentation(segmentationNode, removeIslands=True) #, smoothingFactor=0.3)
    folderName = f"{PHASES_DIRECTORY_MAPPING[self.phase]}-{self.SEGMENTATION_OUTPUT_DIR_NAME}"
    outputDirectory = self.outputDir / folderName
    outputDirectory.mkdir(parents=True, exist_ok=True)
    if self._oneFilePerSegment:
      self._saveSegmentsIntoSeparateFiles(segmentationNode, outputDirectory)
    else:
      self._saveSegmentsIntoSingleFile(segmentationNode, outputDirectory)

  def _saveSegmentsIntoSeparateFiles(self, segmentationNode, outputDirectory):
    segmentationsLogic = slicer.modules.segmentations.logic()
    for segmentID in SegmentationHelper.getAllSegmentIDs(segmentationNode):
      labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
      SegmentationHelper.showOnlySegmentWithSegmentID(segmentationNode, segmentID)
      segmentationsLogic.ExportVisibleSegmentsToLabelmapNode(segmentationNode, labelNode, self.referenceVolumeNode)
      segmentName = segmentationNode.GetSegmentation().GetSegment(segmentID).GetName()
      filename = f"{self.prefix}_{segmentName.replace(' ', '_')}.nii.gz"
      self.saveNode(labelNode, outputDirectory / filename, f"{PHASES_DIRECTORY_MAPPING[self.phase]}-segmentation")

  def _saveSegmentsIntoSingleFile(self, segmentationNode, outputDirectory):
    SegmentationHelper.showOnlySegmentsWithKeywordInName(segmentationNode, keyword="leaflet")
    labelNode = SegmentationHelper.createLabelNodeFromVisibleSegments(segmentationNode, self.referenceVolumeNode,
                                                                      "LeafletSegmentation")
    self.saveNode(labelNode, outputDirectory / f"{self.prefix}.nii.gz",
                  f"{PHASES_DIRECTORY_MAPPING[self.phase]}-segmentation")


class Annulus(ExportItem):

  ANNULI_OUTPUT_DIR_NAME = "annulus"
  SUPPORTED_FORMATS = [".vtk", ".nrrd", ".nii.gz"]

  class AnnulusExportFailed(Exception):
    pass

  def __init__(self, valveModel, asLabel=False, asModel=False):
    super(Annulus, self).__init__(valveModel)
    self._outputFormats = []
    if asLabel is True:
      self._outputFormats.append(".nii.gz")
    if asModel is True:
      self._outputFormats.append(".vtk")

  def __call__(self):
    folderName = f"{PHASES_DIRECTORY_MAPPING[self.phase]}-{self.ANNULI_OUTPUT_DIR_NAME}"
    outputDirectory = self.outputDir / folderName
    outputDirectory.mkdir(parents=True, exist_ok=True)
    for outputFormat in self._outputFormats:
      if outputFormat == ".vtk":
        node = self.getAnnulusModel()
      else:
        node = self.getAnnulusLabel()
      outputFile = outputDirectory / f"{self.prefix}{outputFormat}"
      self.getLogger().debug(f"Exporting annulus to {outputFile}")
      self.saveNode(node, outputFile, f"{PHASES_DIRECTORY_MAPPING[self.phase]}-{self.ANNULI_OUTPUT_DIR_NAME}")

  def getAnnulusModel(self):
    return self._valveModel.getAnnulusContourModelNode()

  def getAnnulusLabel(self):
    segNode = getSegmentationFromAnnulusContourNode(self._valveModel, self.referenceVolumeNode)
    if not segNode:
      raise self.AnnulusExportFailed()
    node = SegmentationHelper.createLabelNodeFromVisibleSegments(segNode, self.referenceVolumeNode, "Annulus")
    return node


class PhaseFrame(ExportItem):

  PHASE_FRAME_OUTPUT_DIR_NAME = "images"

  class NoAssociatedFrameNumberFound(Exception):
    pass

  class NoSequenceBrowserNodeFound(Exception):
    pass

  @classmethod
  def getExportReadyFrameVolume(cls, frameNumber, referenceVolumeNode, valveModel):
    import numpy as np
    numFrames = cls.getNumberOfSequenceFrames(valveModel)
    cls.setSequenceFrameNumber(valveModel, np.clip(frameNumber, 0, numFrames))
    volumeNode = cloneMRMLNode(valveModel.getValveVolumeNode())
    return getResampledScalarVolume(volumeNode, referenceVolumeNode)

  @classmethod
  def getAssociatedFrameNumber(cls, valveModel):
    frameNumber = valveModel.getValveVolumeSequenceIndex()
    if frameNumber == -1:
      raise cls.NoAssociatedFrameNumberFound(f"No associated frame number found for {valveModel.getCardiacCyclePhase()}")
    return frameNumber

  @classmethod
  def setSequenceFrameNumber(cls, valveModel, frameNumber):
    valveVolume = valveModel.getValveVolumeNode()
    seqBrowser = cls.getSequenceBrowserNode(valveVolume)
    seqBrowser.SetSelectedItemNumber(1)
    seqBrowser.SetSelectedItemNumber(frameNumber)

  @classmethod
  def getNumberOfSequenceFrames(cls, valveModel):
    valveVolume = valveModel.getValveVolumeNode()
    seqBrowser = cls.getSequenceBrowserNode(valveVolume)
    return seqBrowser.GetNumberOfItems()

  @classmethod
  def getSequenceBrowserNode(cls, masterOutputNode):
    browserNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLSequenceBrowserNode')
    browserNodes.UnRegister(None)
    for idx in range(browserNodes.GetNumberOfItems()):
      browserNode = browserNodes.GetItemAsObject(idx)
      if browserNode.GetProxyNode(browserNode.GetMasterSequenceNode()) is masterOutputNode:
        browserNode.SetIndexDisplayMode(browserNode.IndexDisplayAsIndex)
        return browserNode
    raise cls.NoSequenceBrowserNodeFound()

  def __init__(self, valveModel, isReferenceFrame=False):
    super(PhaseFrame, self).__init__(valveModel)
    self._isReferenceVolume = isReferenceFrame

  def getVolumeFrame(self):
    if self._isReferenceVolume:
      return cloneMRMLNode(self.referenceVolumeNode)
    else:
      frameNumber = self.getAssociatedFrameNumber(self._valveModel)
      return self.getExportReadyFrameVolume(frameNumber, self.referenceVolumeNode, self._valveModel)

  def __call__(self):
    volumeNode = self.getVolumeFrame()
    if self.outputDir is not None:
      folderName = f"{PHASES_DIRECTORY_MAPPING[self.phase]}-{self.PHASE_FRAME_OUTPUT_DIR_NAME}"
      outputDirectory = self.outputDir / folderName
      outputDirectory.mkdir(parents=True, exist_ok=True)
      outputFile = outputDirectory / f"{self.prefix}.nii.gz"
      self.getLogger().debug(f"Exporting phase {self.phase} to {outputFile}")
      self.saveNode(volumeNode, outputFile, folderName)
    return volumeNode


class AdditionalFrames(ExportItem):

  def __init__(self, phase, frameRange):
    super(AdditionalFrames, self).__init__(phase)
    self._additionalFrameRange = frameRange

  def __call__(self):
    segmentedFrameNumber = PhaseFrame.getAssociatedFrameNumber(self._valveModel)
    self._additionalFrameRange[1] += 1
    for additionalFrame in range(*self._additionalFrameRange):
      additionalFrameNumber = segmentedFrameNumber + additionalFrame
      if additionalFrameNumber == segmentedFrameNumber:
        continue
      folderName = "{}-images{:+d}".format(PHASES_DIRECTORY_MAPPING[self.phase], additionalFrame)
      volumeNode = PhaseFrame.getExportReadyFrameVolume(additionalFrameNumber, self.referenceVolumeNode,
                                                        self._valveModel)
      frame_directory = self.outputDir / folderName
      frame_directory.mkdir(parents=True, exist_ok=True)
      self.saveNode(volumeNode, str(frame_directory / f"{self.prefix}.nii.gz"), folderName)
