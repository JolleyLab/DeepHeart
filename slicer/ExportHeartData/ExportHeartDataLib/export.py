import os
import vtk
import shutil
import logging
from pathlib import Path
from collections import deque

import slicer

from ExportHeartDataLib.constants import APPLICATION_NAME
from ExportHeartDataLib.items import Segmentation, Annulus, PhaseFrame, AdditionalFrames, Landmarks, LandmarkLabels
from ExportHeartDataLib.utils import cloneMRMLNode, computePhaseMetrics, getRandomDirectoryName, getFinalVoxelSpacing
from ExportHeartDataLib.base import ExportBuilder, ExportItem
from ExportHeartDataLib.summary import ExportSummary
import ExportHeartDataLib.reference_volume as ReferenceVolume

import HeartValveLib
from HeartValveLib.helpers import getSpecificHeartValveModelNodes, getFirstValveModelNodeMatchingPhaseAndType, \
  isMRBFile, getAllFilesWithExtension

from typing import Optional


class Exporter(object):

  @property
  def valveType(self):
    return self._valveType

  @valveType.setter
  def valveType(self, value):
    validTypes = HeartValveLib.VALVE_TYPE_PRESETS.keys()
    if not value in validTypes:
      raise ValueError(f"Valve type {value} not supported. Supported types are: {validTypes}")
    self._valveType = value

  @property
  def outputDir(self):
    return self._outputDir

  @outputDir.setter
  def outputDir(self, value):
    self._outputDir = Path(value) if value else (Path(slicer.app.temporaryPath) / getRandomDirectoryName())
    Path(self._outputDir).mkdir(exist_ok=True, parents=True)
    ExportItem.outputDir = self._outputDir

  @property
  def logger(self):
    if not hasattr(self, "_logger"):
      self._logger = logging.getLogger(APPLICATION_NAME)
      if self.saveLogs:
        fh = logging.FileHandler(self.outputDir / f"{APPLICATION_NAME}.log")
        fh.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)
    return self._logger

  def __init__(self,
               valve_type : str,
               phases: Optional[list],
               input_data=None,
               output_directory=None,
               voxel_spacing=None,
               volume_dimensions=None,
               minimum_valve_voxel_height=None,
               additional_frame_range=None,
               export_landmarks=False,
               export_segmentation=False,
               one_file_per_segment=False,
               landmark_labels=None,
               annulus_contour_label=False,
               annulus_contour_model=False,
               run_quantification=False,
               save_log_file=False):
    """
    :param input_data: single MRB file, directory consisting of MRB files, or MRMLScene (currently open)
    """
    self.valveType = valve_type
    self.inputData = input_data
    self.phases = phases
    self.outputDir = output_directory
    self.voxelSpacing = voxel_spacing
    self.volumeDimensions = volume_dimensions

    self.minValveHeightVxl = minimum_valve_voxel_height
    self.additionalFrameRange = additional_frame_range
    self.annulusLabels = landmark_labels
    self.exportLandmarks = export_landmarks
    self.exportSegmentation = export_segmentation                # TODO: it might make sense to provide specific phases
    self.oneFilePerSegment = one_file_per_segment
    self.exportAnnulusContourLabel = annulus_contour_label     # TODO: it might make sense to provide specific phases
    self.exportAnnulusContourModel = annulus_contour_model     # TODO: it might make sense to provide specific phases
    self.runQuantification = run_quantification

    self.saveLogs = save_log_file
    ExportItem.setExportSummarizer(ExportSummary())

  def export(self):
    """
    :return: returning a dictionary with the output types and file paths
    """
    newNodes = []

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeAdded(caller, event, node):
        newNodes.append(node)

    self.logger.debug("Adding mrmlScene observer for detecting new mrml nodes during export for later clean up.")
    mrmlNodeObserver = slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, onNodeAdded)

    try:
      if self.inputData is None or self.inputData is slicer.mrmlScene:
        self._exportFromMRMLScene()
      elif isMRBFile(self.inputData):
        self._exportSingleMRBFile(mrbFile=self.inputData)
      elif os.path.isdir(self.inputData):
        self._exportMRBDirectory()
    except Exception as exc:
      self.logger.exception(exc)
    finally:
      slicer.mrmlScene.RemoveObserver(mrmlNodeObserver)
      logFilePath = slicer.app.errorLogModel().filePath
      shutil.copy(logFilePath, self.outputDir / Path(logFilePath).name)
      deque(map(slicer.mrmlScene.RemoveNode, newNodes))
    return ExportItem.exportSummarizer.get_summary()

  def _exportMRBDirectory(self):
    for mrb in getAllFilesWithExtension(self.inputData, '.mrb'):
      self._exportSingleMRBFile(mrb)

  def _exportSingleMRBFile(self, mrbFile):
    self.logger.info("Processing %s" % mrbFile)
    self.loadScene(mrbFile)
    ExportItem.prefix = os.path.basename(mrbFile).split("_")[0]
    self._processScene()
    slicer.mrmlScene.Clear(0)

  def _exportFromMRMLScene(self):
    ExportItem.prefix = "temp"
    self._processScene()

  def _processScene(self):
    try:
      if self.runQuantification:
        computePhaseMetrics(self.phases[0])
      self._composeAndRunExport()
    except Exception as exc:
      self.logger.info(f"Exception occurred while processing {ExportItem.prefix}")
      import traceback
      traceback.print_exc()
      self.logger.exception(exc)

  def loadScene(self, mrbFile):
    try:
      slicer.app.ioManager().loadScene(mrbFile)
    except Exception as exc:
      self.logger.warning(exc)

  def _composeAndRunExport(self):
    mainValveModel = getFirstValveModelNodeMatchingPhaseAndType(self.phases[0], self._valveType)
    ExportItem.probeToRasTransform = cloneMRMLNode(mainValveModel.getProbeToRasTransformNode())
    self.initializeReferenceVolumeNode(mainValveModel)

    assert ExportItem.requiredAttributesSet()

    # TODO: ExportItems should clean up whatever was added during export
    self._builder = ExportBuilder()

    if self.additionalFrameRange:
      print(f"Adding addition frame range {self.additionalFrameRange}")
      self._builder.add_export_item(AdditionalFrames(mainValveModel, self.additionalFrameRange))

    for valveModel in getSpecificHeartValveModelNodes(self.phases):
      self._builder.add_export_item(PhaseFrame(valveModel, valveModel is mainValveModel))
      self._builder.add_export_item(Annulus(valveModel, self.exportAnnulusContourLabel, self.exportAnnulusContourModel))

      if self.annulusLabels:
        self._builder.add_export_item(LandmarkLabels(valveModel, self.annulusLabels))

      if self.exportLandmarks:
        self._builder.add_export_item(Landmarks(valveModel))

      if self.exportSegmentation:
        self._builder.add_export_item(Segmentation(valveModel, self.oneFilePerSegment))

    self._builder.export()

  def checkRequirements(self):
    satisfied = True
    messages = []

    for phase in self.phases:
      # load scene?
      pass

    # TODO: implement
    # iterate over all user inputs
    # e.g. start with making sure that all required phases are available

    return satisfied, messages

  def initializeReferenceVolumeNode(self, valveModel):
    voxelSpacing = getFinalVoxelSpacing(self.voxelSpacing, self.minValveHeightVxl, valveModel)
    if voxelSpacing is not None and not type(voxelSpacing) is list:
      voxelSpacing = [voxelSpacing] * 3
    ExportItem.referenceVolumeNode = \
      ReferenceVolume.getNormalizedReferenceVolumeNode(valveModel, self.volumeDimensions, voxelSpacing)