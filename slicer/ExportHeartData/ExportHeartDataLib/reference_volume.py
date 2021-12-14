import numpy as np
import vtk

import slicer

from ExportHeartDataLib.constants import getLandmarkLabelsDefinition
from ExportHeartDataLib.utils import getVTKMatrixFromArray, getResampledScalarVolume, getVolumeCloneWithProperties
from ExportHeartDataLib.items import PhaseFrame


def getNormalizedReferenceVolumeNode(valveModel, volumeDimensions, voxelSpacing):
  frameNumber = PhaseFrame.getAssociatedFrameNumber(valveModel)
  numFrames = PhaseFrame.getNumberOfSequenceFrames(valveModel)
  PhaseFrame.setSequenceFrameNumber(valveModel, np.clip(frameNumber, 0, numFrames))

  volumeNode = valveModel.getValveVolumeNode()
  referenceVolumeNode = getVolumeCloneWithProperties(volumeNode, volumeDimensions, voxelSpacing)
  valveIJKToProbe = _getValveIJKtoProbeTransform(valveModel, referenceVolumeNode)
  referenceVolumeNode.SetIJKToRASMatrix(valveIJKToProbe)
  normalizedVolume = getResampledScalarVolume(volumeNode, referenceVolumeNode)
  return normalizedVolume


def _getValveIJKtoProbeTransform(valveModel, referenceVolumeNode):
  requiredLandmarks = getLandmarkLabelsDefinition(valveModel.getValveType())
  landmarks = valveModel.getAnnulusMarkupPositionsByLabels(requiredLandmarks)
  assert all(lm is not None for lm in landmarks), f"Missing landmarks: {requiredLandmarks}, {landmarks}"
  valveToProbe = _getValveToProbeTransform(np.array(landmarks))
  valveToValveIJK = _getValveToValveIJKTransform(referenceVolumeNode)
  return _getValveIJKToProbe(valveToProbe, valveToValveIJK)


def _getValveToProbeTransform(landmarks: np.array) -> np.array:
  """ Compute transform that translates and rotates annulus landmarks to a normalized position and orientation:
  - ORIGIN: landmarks' centroid
  - X-axis: from origin to first landmark
  - Y-axis: cross product of Z-axis and X-axis
  - Z-axis: cross product of X-axis and valve origin to second landmark

  The resulting transform only includes translation and rotation (no scaling).

  :param landmarks: annulus quadrant landmarks of the valve
  :return:

  """
  valveOriginProbe = np.mean(landmarks, axis=0)

  xValveAxisProbe = landmarks[0] - valveOriginProbe
  xValveAxisProbe = xValveAxisProbe / np.linalg.norm(xValveAxisProbe)

  valvePlaneAxisProbe = landmarks[1] - valveOriginProbe
  valvePlaneAxisProbe = valvePlaneAxisProbe / np.linalg.norm(valvePlaneAxisProbe)

  zValveAxisProbe = np.cross(xValveAxisProbe, valvePlaneAxisProbe)
  zValveAxisProbe = zValveAxisProbe / np.linalg.norm(zValveAxisProbe)

  yValveAxisProbe = np.cross(zValveAxisProbe, xValveAxisProbe)
  yValveAxisProbe = yValveAxisProbe / np.linalg.norm(yValveAxisProbe)

  valveToProbe = np.eye(4)
  valveToProbe[0:3, 0] = xValveAxisProbe
  valveToProbe[0:3, 1] = yValveAxisProbe
  valveToProbe[0:3, 2] = zValveAxisProbe
  valveToProbe[0:3, 3] = valveOriginProbe
  return valveToProbe


def _getValveToValveIJKTransform(referenceVolume: slicer.vtkMRMLScalarVolumeNode) -> np.array:
  """ valve will be centered in IJK volume

  :param referenceVolume: slicer.vtkMRMLScalarVolumeNode
  :return:
  """
  spacing = referenceVolume.GetSpacing()
  dim = referenceVolume.GetImageData().GetDimensions()
  valveToValveIJK = np.array([[1 / spacing[0], 0, 0, dim[0] / 2],
                              [0, 1 / spacing[1], 0, dim[1] / 2],
                              [0, 0, 1 / spacing[2], dim[2] / 2],
                              [0, 0, 0, 1]])
  return valveToValveIJK


def _getValveIJKToProbe(valveToProbe: np.array, valveToValveIJK: np.array) -> vtk.vtkMatrix4x4:
  """ Get transform from valve IJK coordinate system to probe coordinate system

  :param valveToProbe:
  :param valveToValveIJK:
  :return:
  """
  valveIjkToProbe = vtk.vtkMatrix4x4()
  vtk.vtkMatrix4x4.Multiply4x4(getVTKMatrixFromArray(valveToProbe),
                               getVTKMatrixFromArray(np.linalg.inv(valveToValveIJK)),
                               valveIjkToProbe)
  return valveIjkToProbe