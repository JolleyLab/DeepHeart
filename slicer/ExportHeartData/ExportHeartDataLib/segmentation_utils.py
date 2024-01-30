import logging

from collections import OrderedDict

import slicer
import vtk
import numpy as np

from ExportHeartDataLib.constants import LEAFLET_ORDER, LEAFLET_ORDER_CODE_VALUES, LEAFLET_ORDER_CODE_MEANINGS
from ExportHeartDataLib.exceptions import MissingSegmentsError, MultipleSegmentsForCodeValueError


from typing import Optional, List, Dict


def getAllSegmentNames(segmentationNode):
  return [segment.GetName() for segment in getAllSegments(segmentationNode)]


def getAllSegments(segmentationNode):
  segmentation = segmentationNode.GetSegmentation()
  return [segmentation.GetSegment(segmentID) for segmentID in getAllSegmentIDs(segmentationNode)]


def getAllSegmentIDs(segmentationNode):
  segmentIDs = vtk.vtkStringArray()
  segmentation = segmentationNode.GetSegmentation()
  segmentation.GetSegmentIDs(segmentIDs)
  return [segmentIDs.GetValue(idx) for idx in range(segmentIDs.GetNumberOfValues())]


def hideAllSegments(segmentationNode):
  for segmentID in getAllSegmentIDs(segmentationNode):
    segmentationNode.GetDisplayNode().SetSegmentVisibility(segmentID, False)


def showAllSegments(segmentationNode):
  for segmentID in getAllSegmentIDs(segmentationNode):
    segmentationNode.GetDisplayNode().SetSegmentVisibility(segmentID, True)


def showOnlySegmentsWithKeywordInName(segmentationNode, keyword):
  hideAllSegments(segmentationNode)
  for segment, segmentID in zip(getAllSegments(segmentationNode), getAllSegmentIDs(segmentationNode)):
    if not keyword in segment.GetName():
      continue
    segmentationNode.GetDisplayNode().SetSegmentVisibility(segmentID, True)


def showOnlySegmentWithSegmentID(segmentationNode, segmentID):
  hideAllSegments(segmentationNode)
  segmentationNode.GetDisplayNode().SetSegmentVisibility(segmentID, True)


def getSegmentCenter(valveModel, keyword="leaflet"):
  segmentationNode = valveModel.getLeafletSegmentationNode()
  segmentIDs = filter(lambda s: keyword, getAllSegmentIDs(segmentationNode))
  return np.mean([np.array(segmentationNode.GetSegmentCenter(sID)) for sID in segmentIDs], axis=0)


def createLabelNodeFromVisibleSegments(segmentationNode, referenceVolumeNode, labelNodeName):
  labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", labelNodeName)
  segmentationsLogic = slicer.modules.segmentations.logic()
  segmentationsLogic.ExportVisibleSegmentsToLabelmapNode(segmentationNode, labelNode, referenceVolumeNode)
  return labelNode


def deleteNonLeafletSegments(segmentationNode, keyword="leaflet"):
  segmentation = segmentationNode.GetSegmentation()
  for segment in getAllSegments(segmentationNode):
    if keyword in segment.GetName():
      continue
    logging.info(f"Removing segment with name: {segment.GetName()}")
    segmentation.RemoveSegment(segment)


def getNewSegmentationNode(masterVolumeNode):
  segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
  segmentationNode.CreateDefaultDisplayNodes()
  segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(masterVolumeNode)
  return segmentationNode


def hasEmptySegments(segmentationNode):
  for segmentID in getAllSegmentIDs(segmentationNode):
    if not getVoxelCount(segmentationNode, segmentID):
      return True
  return False


def getVoxelCount(segmentationNode, segmentID):
  segmentLabelmap = segmentationNode.GetBinaryLabelmapInternalRepresentation(segmentID)

  # We need to know exactly the value of the segment voxels, apply threshold to make force the selected label value
  labelValue = 1
  backgroundValue = 0
  thresh = vtk.vtkImageThreshold()
  thresh.SetInputData(segmentLabelmap)
  thresh.ThresholdByLower(0)
  thresh.SetInValue(backgroundValue)
  thresh.SetOutValue(labelValue)
  thresh.SetOutputScalarType(vtk.VTK_UNSIGNED_CHAR)
  thresh.Update()

  #  Use binary labelmap as a stencil
  stencil = vtk.vtkImageToImageStencil()
  stencil.SetInputData(thresh.GetOutput())
  stencil.ThresholdByUpper(labelValue)
  stencil.Update()

  stat = vtk.vtkImageAccumulate()
  stat.SetInputData(thresh.GetOutput())
  stat.SetStencilData(stencil.GetOutput())
  stat.Update()

  return stat.GetVoxelCount()


def getLeafletOrderDefinition(valveType):
  try:
    return LEAFLET_ORDER[valveType.lower()]
  except KeyError:
    raise ValueError("valve type %s not supported " % valveType)


def checkAndSortSegments(segmentationNode, valveType):
  expectedOrder = getLeafletOrderDefinition(valveType)
  segmentIDs = getAllSegmentIDs(segmentationNode)
  segmentNames = getAllSegmentNames(segmentationNode)
  if not isSorted(expectedOrder, segmentIDs) or not isSorted(expectedOrder, segmentNames):
    logging.info("Leaflet names don't match up with segment IDs ")
    sortSegments(segmentationNode, valveType)


def sortSegments(segmentationNode, valveType):
  expectedOrder = getLeafletOrderDefinition(valveType)
  segmentation = segmentationNode.GetSegmentation()
  segmentInfos = getSortedSegmentInfos(segmentationNode, expectedOrder)
  newSegmentIDs, segments = getSortedSegmentsAndIDs(segmentationNode, segmentInfos, valveType)
  segmentation.RemoveAllSegments()
  for newSegmentID, segment in zip(newSegmentIDs, segments):
    segmentation.AddSegment(segment, newSegmentID)


def getSortedSegmentInfos(segmentationNode, expectedOrder):
  segmentNames = getAllSegmentNames(segmentationNode)
  orderedSegmentNames = list()
  for location in expectedOrder:
    segmentName = getFirstMatchingListElement(segmentNames, location)
    if not segmentName:
      raise ValueError(f"Cannot find segment with name {location}. Following segments are available: {segmentNames}")
    orderedSegmentNames.append((segmentName, location))
  return orderedSegmentNames


def getSortedSegmentsAndIDs(segmentationNode, segmentInfos, valveType):
  segmentation = segmentationNode.GetSegmentation()
  newSegmentIDs = list()
  segments = list()
  for segmentName, loc in segmentInfos:
    segmentID = segmentation.GetSegmentIdBySegmentName(segmentName)
    newSegmentID = "{}_{}_leaflet".format(valveType, loc)
    newSegmentIDs.append(newSegmentID)
    segments.append(segmentation.GetSegment(segmentID))
  return newSegmentIDs, segments


def postProcessSegmentation(segmentationNode, removeIslands=True, smoothingFactor=None):
  from SegmentEditorEffects import JOINT_TAUBIN

  # Create segment editor to get access to effects
  segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
  segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
  segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
  segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
  segmentEditorWidget.setSegmentationNode(segmentationNode)

  if smoothingFactor is not None:
    logging.info("Joint smoothing segments")
    assert type(smoothingFactor) is float
    segmentEditorWidget.setActiveEffectByName("Smoothing")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("SmoothingMethod", JOINT_TAUBIN)
    effect.setParameter("JointTaubinSmoothingFactor", smoothingFactor)
    effect.self().onApply()

  if removeIslands:
    from SegmentEditorEffects.SegmentEditorIslandsEffect import KEEP_LARGEST_ISLAND
    logging.info("Removing islands")
    segmentEditorWidget.setActiveEffectByName("Islands")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation", KEEP_LARGEST_ISLAND)
    showAllSegments(segmentationNode)
    for segID in getAllSegmentIDs(segmentationNode):
      logging.info(f"Setting current segment ID: {segID}")
      segmentEditorWidget.setCurrentSegmentID(segID)
      effect.self().onApply()

  segmentEditorWidget = None
  slicer.mrmlScene.RemoveNode(segmentEditorNode)


def calculateValveVolume(segmentationNode):
  stats = computeSegmentStatistics(segmentationNode)
  leafletVolumeMm3 = 0.0
  for segmentID, segmentName in zip(getAllSegmentIDs(segmentationNode), getAllSegmentNames(segmentationNode)):
    if any("leaflet" in attr for attr in [segmentID, segmentName]):
      if not getVoxelCount(segmentationNode, segmentID):
        continue
      leafletVolumeMm3 += stats[segmentID, 'ClosedSurfaceSegmentStatisticsPlugin.volume_mm3']
  return leafletVolumeMm3


def computeSegmentStatistics(segmentationNode):
  from SegmentStatistics import SegmentStatisticsLogic
  segmentStaticsLogic = SegmentStatisticsLogic()
  showAllSegments(segmentationNode)
  segmentStaticsLogic.getParameterNode().SetParameter("Segmentation", segmentationNode.GetID())
  segmentStaticsLogic.computeStatistics()
  return segmentStaticsLogic.getStatistics()


def getNewSegmentationNodeFromModel(valveVolume, modelNode):
  segmentationNode = getNewSegmentationNode(valveVolume)
  segmentationsLogic = slicer.modules.segmentations.logic()
  segmentationsLogic.ImportModelToSegmentationNode(modelNode, segmentationNode)
  return segmentationNode


def getFirstMatchingListElement(elements : list, keyword : str) -> Optional[str]:
  """ Returns first element with the keyword in it

  :param elements: list of strings
  :param keyword:
  :return: None if none was found, otherwise the first matching element
  """
  for elem in elements:
    if keyword in elem:
      return elem
  return None


def isSorted(expectedOrder : list, currentOrder : list) -> bool:
  """ returns if the current list of strings has the expected order of elements

  :param expectedOrder: list of keywords as expected in the specific element
  :param currentOrder: list of strings to check for order
  :return: true if ordered, otherwise false
  """
  return all(expectedOrder[i] in currentOrder[i] for i in range(len(expectedOrder)))


def getSegmentIDsMatchingValveType(segmentationNode: slicer.vtkMRMLSegmentationNode, valveType: str):
  """ get a list of segment ids that match terminology codes required by the provided valve type

  :param segmentationNode: segmentation node to retrieve matching segment ids from
  :param valveType: valve type to use for retrieving required segment codes
  :return: list of segment ids
  """
  requiredCodeValues = LEAFLET_ORDER_CODE_VALUES[valveType]
  requiredCodeMeanings = LEAFLET_ORDER_CODE_MEANINGS[valveType]
  requiredCodes = {codeValue: codeMeaning for codeValue, codeMeaning in zip(requiredCodeValues, requiredCodeMeanings)}
  return getSegmentIDsMatchingTerminologyCodes(segmentationNode, requiredCodes)


def getSegmentIDsMatchingTerminologyCodes(segmentationNode: slicer.vtkMRMLSegmentationNode, requiredCodes: Dict[str,str]):
  """ Retrieves a list of terminology matching segment ids in requested order

  :param segmentationNode: segmentation node to retrieve matching segment ids from
  :param requiredCodes: dict of codeValue:codeMeaning
  :return: list of segment ids if successful
  """
  allCodeValues = getAllTerminologyTypeCodeValues(segmentationNode)
  filteredCodeValues = []
  for codeValue, codeMeaning in requiredCodes.items():
    if not codeValue in allCodeValues:
      raise MissingSegmentsError(f"Could not find requested segment with assigned terminology {codeMeaning} ({codeValue})")
    segmentIDs = allCodeValues[codeValue]
    if len(segmentIDs) > 1:
      raise MultipleSegmentsForCodeValueError(
        f"Found multiple segments ({segmentIDs}) with same terminology {codeMeaning} ({codeValue}) assigned."
      )
    filteredCodeValues.append(segmentIDs[0])

  return filteredCodeValues


def getAllTerminologyTypeCodeValues(segmentationNode: slicer.vtkMRMLSegmentationNode) -> OrderedDict[str, List]:
  """ iterates over all segments and returns a dictionary of code value and a list of segment ids (if multiple)

  the same code value can be assigned to multiple segments

  :return: dict of codeValue: [segment id]
  """
  codeValues = OrderedDict()
  for segment, segmentID in zip(getAllSegments(segmentationNode), getAllSegmentIDs(segmentationNode)):
    codeValue = getTerminologyTypeCodeValue(segment)
    if not codeValue in codeValues:
      codeValues[codeValue] = []
    codeValues[codeValue].append(segmentID)
  return codeValues


def getTerminologyTypeCodeValue(segment: slicer.vtkSegment) -> str:
  terminologyEntry = getTerminologyEntryFromSegment(segment)
  return terminologyEntry.GetTypeObject().GetCodeValue()


def getTerminologyTypeCodeMeaning(segmentationNode: slicer.vtkMRMLSegmentationNode, segmentId: str) -> str:
  segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
  terminologyEntry = getTerminologyEntryFromSegment(segment)
  return terminologyEntry.GetTypeObject().GetCodeMeaning()


def getTerminologyEntryFromSegment(segment: slicer.vtkSegment):
  """ Read terminology saved for provided segment into vtkSlicerTerminologyEntry

  :param segment: segment to retrieve terminology entry from
  :return: vtkSlicerTerminologyEntry
  """
  terminologyEntry = slicer.vtkSlicerTerminologyEntry()
  tag = vtk.mutable("")
  segment.GetTag(segment.GetTerminologyEntryTagName(), tag)
  terminologyLogic = slicer.modules.terminologies.logic()
  terminologyLogic.DeserializeTerminologyEntry(tag, terminologyEntry)
  return terminologyEntry
