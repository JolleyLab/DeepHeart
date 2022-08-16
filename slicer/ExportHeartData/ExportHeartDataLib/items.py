import slicer

from ExportHeartDataLib.constants import PHASES_DIRECTORY_MAPPING
from ExportHeartDataLib.base import ExportItem
from ExportHeartDataLib.utils import getSegmentationFromAnnulusContourNode, getLabelFromLandmarkPosition, \
  cloneMRMLNode, getResampledScalarVolume
import ExportHeartDataLib.segmentation_utils as SegmentationHelper
from HeartValveLib.helpers import getValvePhaseShortName


class Landmarks(ExportItem):

  def verify(self):
    valid = self._valveModel.getAnnulusLabelsMarkupNode() is not None
    return valid, \
           None if valid else f"No Annulus Labels Markup Node found for phase {getValvePhaseShortName(self._valveModel)}"

  def __init__(self, valveModel, phase=None):
    super(Landmarks, self).__init__(valveModel, phase)

  def __call__(self):
    logger = self.getLogger()
    landmarksNode = cloneMRMLNode(self._valveModel.getAnnulusLabelsMarkupNode())
    folderName = f"{PHASES_DIRECTORY_MAPPING[self.phase]}-landmarks"
    outputDirectory = self.outputDir / folderName
    outputFile = outputDirectory / f"{self.prefix}_f{self.getAssociatedFrameNumber(self._valveModel)}.fcsv"
    outputDirectory.mkdir(exist_ok=True)
    logger.debug(f"Saving landmarks to {outputFile}: {self.saveNode(landmarksNode, outputFile, 'landmarks')}")
    slicer.mrmlScene.RemoveNode(landmarksNode)


class LandmarkLabels(ExportItem):

  def verify(self):
    positions = self._valveModel.getAnnulusMarkupPositionsByLabels(self.landmarkLabels)
    if all(pos is not None for pos in positions):
      return True, None
    else:
      message = f"Found annulus landmark positions for phase {getValvePhaseShortName(self._valveModel)} are: " \
                f"{[f'{a}={b}' for a,b in zip(self.landmarkLabels, positions)]}"
      return False, message

  def __init__(self, valveModel, landmarkLabels, phase=None):
    super(LandmarkLabels, self).__init__(valveModel, phase)
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
      outputFile = outputDirectory / f"{self.prefix}_f{self.getAssociatedFrameNumber(self._valveModel)}.nii.gz"
      logger.debug(f"Saving landmark {landmarkLabel} to {outputFile}: "
                   f"{self.saveNode(labelNode, outputFile, folderName)}")
      slicer.mrmlScene.RemoveNode(labelNode)


class Segmentation(ExportItem):

  SEGMENTATION_OUTPUT_DIR_NAME = "segmentation"

  def verify(self):
    valid = self._valveModel.getLeafletSegmentationNode() is not None
    return valid, None if valid else "No leaflet segmentation node could be found " \
                                     "for phase {getValvePhaseShortName(self._valveModel)}"

  def __init__(self, valveModel, oneFilePerSegment, phase=None):
    super(Segmentation, self).__init__(valveModel, phase)
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
    slicer.mrmlScene.RemoveNode(segmentationNode)

  def _saveSegmentsIntoSeparateFiles(self, segmentationNode, outputDirectory):
    segmentationsLogic = slicer.modules.segmentations.logic()
    for segmentID in SegmentationHelper.getAllSegmentIDs(segmentationNode):
      labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
      SegmentationHelper.showOnlySegmentWithSegmentID(segmentationNode, segmentID)
      segmentationsLogic.ExportVisibleSegmentsToLabelmapNode(segmentationNode, labelNode, self.referenceVolumeNode)
      segmentName = segmentationNode.GetSegmentation().GetSegment(segmentID).GetName()
      filename = f"{self.prefix}_f{self.getAssociatedFrameNumber(self._valveModel)}_{segmentName.replace(' ', '_')}.nii.gz"
      self.saveNode(labelNode, outputDirectory / filename, f"{PHASES_DIRECTORY_MAPPING[self.phase]}-segmentation")

  def _saveSegmentsIntoSingleFile(self, segmentationNode, outputDirectory):
    SegmentationHelper.showOnlySegmentsWithKeywordInName(segmentationNode, keyword="leaflet")
    labelNode = SegmentationHelper.createLabelNodeFromVisibleSegments(segmentationNode, self.referenceVolumeNode,
                                                                      "LeafletSegmentation")
    self.saveNode(labelNode, outputDirectory / f"{self.prefix}_f{self.getAssociatedFrameNumber(self._valveModel)}.nii.gz",
                  f"{PHASES_DIRECTORY_MAPPING[self.phase]}-segmentation")


class Annulus(ExportItem):

  ANNULI_OUTPUT_DIR_NAME = "annulus"
  SUPPORTED_FORMATS = [".vtk", ".nrrd", ".nii.gz"]

  class AnnulusExportFailed(Exception):
    pass

  def verify(self):
    valid = self._valveModel.getAnnulusContourModelNode() is not None
    return valid, None if valid else "No annulus contour model node could be found " \
                                     "for phase {getValvePhaseShortName(self._valveModel)}"

  def __init__(self, valveModel, asLabel=False, asModel=False, phase=None):
    super(Annulus, self).__init__(valveModel, phase)
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
        node = cloneMRMLNode(self._valveModel.getAnnulusContourModelNode())
      else:
        node = self.getAnnulusLabel()
      outputFile = outputDirectory / f"{self.prefix}_f{self.getAssociatedFrameNumber(self._valveModel)}{outputFormat}"
      self.getLogger().debug(f"Exporting annulus to {outputFile}")
      self.saveNode(node, outputFile, f"{PHASES_DIRECTORY_MAPPING[self.phase]}-{self.ANNULI_OUTPUT_DIR_NAME}")
      slicer.mrmlScene.RemoveNode(node)

  def getAnnulusLabel(self):
    segNode = getSegmentationFromAnnulusContourNode(self._valveModel, self.referenceVolumeNode)
    if not segNode:
      raise self.AnnulusExportFailed()
    node = SegmentationHelper.createLabelNodeFromVisibleSegments(segNode, self.referenceVolumeNode, "Annulus")
    slicer.mrmlScene.RemoveNode(segNode)
    return node


class PhaseFrame(ExportItem):

  PHASE_FRAME_OUTPUT_DIR_NAME = "images"

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

  def verify(self):
    try:
      self.getAssociatedFrameNumber(self._valveModel)
    except Exception as exc:
      self.getLogger().error(exc)
      return False, exc
    return True, None

  def __init__(self, valveModel, isReferenceFrame=False, phase=None):
    super(PhaseFrame, self).__init__(valveModel, phase)
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
      outputFile = outputDirectory / f"{self.prefix}_f{self.getAssociatedFrameNumber(self._valveModel)}.nii.gz"
      self.getLogger().debug(f"Exporting phase {self.phase} to {outputFile}")
      self.saveNode(volumeNode, outputFile, folderName)
    slicer.mrmlScene.RemoveNode(volumeNode)


class AdditionalFrames(ExportItem):

  def verify(self):
    associatedFrameNumber = PhaseFrame.getAssociatedFrameNumber(self._valveModel)
    numFrames = PhaseFrame.getNumberOfSequenceFrames(self._valveModel)
    lowEnd = associatedFrameNumber + self._additionalFrameRange[0]
    upperEnd = associatedFrameNumber + self._additionalFrameRange[1] + 1
    valid = lowEnd >= 0 and upperEnd < (numFrames -1)
    if not valid:
      message = f"Additional frame range invalid: \n " \
                f"ValveModel associated frame: {associatedFrameNumber}\n" \
                f"Number of frames in volume sequence: {numFrames}\n" \
                f"Requested frames: {lowEnd} to {upperEnd}"
      self.getLogger().warn(message)
      return False, message
    return True, None


  def __init__(self, valveModel, frameRange, phase=None):
    super(AdditionalFrames, self).__init__(valveModel, phase)
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
      self.saveNode(volumeNode, str(frame_directory / f"{self.prefix}_f{self.getAssociatedFrameNumber(self._valveModel)}.nii.gz"), folderName)
      slicer.mrmlScene.RemoveNode(volumeNode)
