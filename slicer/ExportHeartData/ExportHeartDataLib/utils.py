import numpy as np
import vtk

from typing import Optional


def generateDirectoryName(valve_type:str=None, desired_volume_dimensions:list=None, minimum_valve_voxel_height:int=None):
  directory = "DL_DATA"
  if desired_volume_dimensions:
    assert type(desired_volume_dimensions) is list
    directory = "{}_{}".format(directory, "_".join([str(res) for res in desired_volume_dimensions]))
  if minimum_valve_voxel_height:
    assert type(minimum_valve_voxel_height) is int
    directory = "{}_{}_vox_min".format(directory, minimum_valve_voxel_height)
  if valve_type:
    assert type(valve_type) is str
    directory = "{}_{}".format(directory, valve_type)
  return directory


def getRandomDirectoryName():
  import datetime
  import random
  import string
  start_date = datetime.datetime.now().strftime('%m%d%Y')
  start_time = datetime.datetime.now().strftime('%H%M%S')
  random_hash = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
  return f"temp_{start_date}_{start_time}_{random_hash}"


def getValveThickness(valveModel):
  import HeartValveLib
  import ExportHeartDataLib.segmentation_utils as SegmentationHelper
  annulusPoints = valveModel.annulusContourCurve.getInterpolatedPointsAsArray()
  [planePosition, planeNormal] = valveModel.getAnnulusContourPlane()
  [annulusPointsProjected, _, _] = HeartValveLib.getPointsProjectedToPlane(annulusPoints, planePosition, planeNormal)
  annulusAreaPolyData = createPolyDataFromPolygon(annulusPointsProjected.T)
  annulusAreaMm2 = getSurfaceArea(annulusAreaPolyData)
  leafletVolumeMm3 = SegmentationHelper.calculateValveVolume(valveModel.getLeafletSegmentationNode())
  return leafletVolumeMm3 / annulusAreaMm2 / 2 # dividing by 2 because the estimate with annulus area is not precise


def createPolyDataFromPolygon(pointPositions):
  """ Create a polygon model (2D) with a tube around its circumference

  :param pointPositions: points defining the polygon, each row defines a point
  :return: MRML model node
  """
  numberOfPoints = pointPositions.shape[0]

  # Points
  polygonPolyData = vtk.vtkPolyData()
  points = vtk.vtkPoints()
  points.SetNumberOfPoints(numberOfPoints)
  for pointIndex in range(numberOfPoints):
    points.SetPoint(pointIndex, pointPositions[pointIndex])
  polygonPolyData.SetPoints(points)

  # Polygon cell
  polygon = vtk.vtkPolygon()
  polygonPointIds = polygon.GetPointIds()
  polygonPointIds.SetNumberOfIds(numberOfPoints + 1)  # +1 for the closing segment
  for pointIndex in range(numberOfPoints):
    polygonPointIds.SetId(pointIndex, pointIndex)
  polygonPointIds.SetId(numberOfPoints, 0)  # closing segment

  # Cells
  polygons = vtk.vtkCellArray()
  polygons.InsertNextCell(polygon)
  polygonPolyData.SetPolys(polygons)

  # Polygon may be non-convex, so we have to triangulate to allow correct rendering
  # and surface computation
  triangulator = vtk.vtkTriangleFilter()
  triangulator.SetInputData(polygonPolyData)
  triangulator.Update()

  return triangulator.GetOutput()


def computePhaseMetrics(phase):
  # TODO: is this needed?
  from HeartValveLib.helpers import getHeartValveMeasurementNode
  measurementNode = getHeartValveMeasurementNode(phase)
  import ValveQuantification
  valveQuantificationLogic = ValveQuantification.ValveQuantificationLogic()
  valveQuantificationLogic.computeMetrics(measurementNode)


def getSurfaceArea(polydata):
  properties = vtk.vtkMassProperties()
  triangulator = vtk.vtkTriangleFilter()
  triangulator.PassLinesOff()
  triangulator.SetInputData(polydata)
  triangulator.Update()
  properties.SetInputConnection(triangulator.GetOutputPort())
  return properties.GetSurfaceArea()


def getSegmentationFromAnnulusContourNode(valveModel, referenceVolume):
  import slicer
  import ExportHeartDataLib.segmentation_utils as SegmentationHelper
  annulusModelNode = cloneMRMLNode(valveModel.getAnnulusContourModelNode())
  annulusModelNode.SetAndObserveTransformNodeID(None)
  segmentationNode = SegmentationHelper.getNewSegmentationNodeFromModel(referenceVolume, annulusModelNode)
  slicer.mrmlScene.RemoveNode(annulusModelNode)
  return segmentationNode


def getLabelFromLandmarkPosition(name, pos, referenceVolumeNode):
  import random
  import slicer
  import ExportHeartDataLib.segmentation_utils as SegmentationHelper

  segNode = SegmentationHelper.getNewSegmentationNode(referenceVolumeNode)
  sphereSegment = slicer.vtkSegment()
  sphereSegment.SetName(name)
  sphereSegment.SetColor(random.uniform(0.0,1.0), random.uniform(0.0,1.0), random.uniform(0.0,1.0))
  sphere = vtk.vtkSphereSource()
  sphere.SetCenter(*pos)
  sphere.SetRadius(1)
  sphere.Update()
  spherePolyData = sphere.GetOutput()
  sphereSegment.AddRepresentation(
    slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName(),
    spherePolyData)
  segNode.GetSegmentation().AddSegment(sphereSegment)
  labelNode = SegmentationHelper.createLabelNodeFromVisibleSegments(segNode, referenceVolumeNode, name)
  slicer.mrmlScene.RemoveNode(segNode)
  return labelNode


def cloneMRMLNode(node):
  import slicer
  shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
  itemIDToClone = shNode.GetItemByDataNode(node)
  clonedItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, itemIDToClone)
  return shNode.GetItemDataNode(clonedItemID)


def getVTKMatrixFromArray(array):
  matrix = vtk.vtkMatrix4x4()
  for i in range(len(array)):
    for j in range(len(array[i])):
      matrix.SetElement(i, j, array[i, j])
  return matrix


def getResampledScalarVolume(inputVolume, referenceVolume, interpolation="Linear"):
  """ resampling input volume based on provided reference volume

  :param inputVolume: input volume to be resampled
  :param referenceVolume: reference volume to use
  :param interpolation: interpolation Linear or NearestNeighbor
  :return: vtkMRMLScalarVolumeNode
  """
  import slicer
  outputVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
  params = {'inputVolume': inputVolume,
            'referenceVolume': referenceVolume,
            'outputVolume': outputVolume,
            'interpolationMode': interpolation}
  # Exporter.LOGGER.debug('About to run BRAINSResample CLI with those params: %s' % params)
  slicer.cli.run(slicer.modules.brainsresample, None, params, wait_for_completion=True)
  return outputVolume


def getFinalVoxelSpacing(desiredVoxelSpacing : float=None,
                         minimumValveSegmentationHeightVxl : float=None,
                         valveModel=None) -> Optional[float]:
  """ Depending on provided parameters, a final voxel spacing will be provided to satisfy e.g. min valve height

  :param desiredVoxelSpacing: maximum voxel spacing
  :param minimumValveSegmentationHeightVxl: minimum valve segmentation height in voxels
  :param valveModel: SlicerHeart valve model node
  :return:
  """
  if minimumValveSegmentationHeightVxl is not None:
    assert valveModel is not None
    thickness = getValveThickness(valveModel)
    if desiredVoxelSpacing is not None:
      # figure out which pixel spacing is smaller (the desired one or the minimum required to satisfy voxel height)
      return np.min([thickness / minimumValveSegmentationHeightVxl, desiredVoxelSpacing])
    else:
      return thickness / minimumValveSegmentationHeightVxl
  elif desiredVoxelSpacing is not None:
    # only maximum voxel spacing was provided
    return desiredVoxelSpacing
  return None


def getVolumeCloneWithProperties(volumeNode,
                                 volumeDimensions: list,
                                 voxelSpacing=Optional[list]):
  """ Clone volume node with the options of setting properties

  :param volumeNode: volume node to use as template
  :param volumeDimensions: list or numpy array
  :param voxelSpacing: isotropic spacing in mm between voxels
  :return:
  """
  import slicer
  import logging
  from ExportHeartDataLib.constants import APPLICATION_NAME
  volumesLogic = slicer.modules.volumes.logic()
  clonedVolume = volumesLogic.CloneVolume(slicer.mrmlScene, volumeNode, volumeNode.GetName() + "_reference")
  if voxelSpacing:
    if type(voxelSpacing) is float:
      voxelSpacing = [voxelSpacing] * 3
    clonedVolume.SetSpacing(voxelSpacing)
  imageData = clonedVolume.GetImageData()
  if volumeDimensions:
    logging.getLogger(APPLICATION_NAME).debug(f"Setting dimensions of exported image data to {volumeDimensions}")
    imageData.SetDimensions(volumeDimensions)
  imageData.AllocateScalars(imageData.GetScalarType(), 1)
  return clonedVolume


from dataclasses import dataclass
from HeartValveLib.ValveModel import ValveModel

@dataclass
class ValveDataClass:
  valveModel: ValveModel
  phase: str
  suffix: str = ""


def getValveModelsWithPhaseNames(phaseNames, valveType):
  """ get all valve models matching the requested phase names
    :param phaseNames: list of strings (e.g. ["MS-1", "MS", "MS+1"])
    :param valveType:

    :return: list of ValveDataClass objects
  """
  from HeartValveLib.helpers import getAllHeartValveModelsForValveType
  valveModels = getAllHeartValveModelsForValveType(valveType)
  cache = {}
  vms = []
  for shortName in phaseNames:
    shortName = shortName.replace(" ", "").strip() # strip white spaces e.g. when 'MS + 1 '
    prefix, incrementor = getPhaseShortNameAndNumber(shortName)
    try:
      valveModel = cache[prefix]
    except KeyError:
      valveModel = getFirstValveModelMatchingShortName(valveModels, prefix)
      cache[prefix] = valveModel

    if int(incrementor) != 0:
      from HeartValveLib.helpers import getFirstValveModelNodeMatchingSequenceIndexAndValveType
      valveModel = getFirstValveModelNodeMatchingSequenceIndexAndValveType(
          valveModel.getValveVolumeSequenceIndex() + int(incrementor), valveModel.getValveType())
    vms.append(ValveDataClass(valveModel, prefix, incrementor if int(incrementor) != 0 else ""))
  return vms


def getFirstValveModelMatchingShortName(valveModels, shortname):
  from HeartValveLib.helpers import getValvePhaseShortName
  for valveModel in valveModels:
    if getValvePhaseShortName(valveModel) == shortname:
      return valveModel
  raise ValueError(f"Could not find valve with cardiac cycle phase {shortname}")


def getPhaseShortNameAndNumber(shortName):
  """ returns phase shortname and any additional number

  e.g.
    MS+1 would return MS, +1
    MS returns MS, 0
    MS - 1 returns MS, -1

  :param shortName:
  :return:
  """
  for op in ["-", "+"]:
    try:
      prefix, suffix = shortName.split(op)
      return prefix, op+suffix
    except ValueError:
      continue
  return shortName, "0"
