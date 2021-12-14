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
from HeartValveLib.helpers import getSpecificHeartValveModelNodesMatchingPhaseAndType, getFirstValveModelNodeMatchingPhaseAndType, \
  isMRBFile, getAllFilesWithExtension, getValvePhaseShortName

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
               cardiac_phase_frames: Optional[list],
               input_data=None,
               output_directory=None,
               voxel_spacing=None,
               volume_dimensions=None,
               export_segmentation=False,
               one_file_per_segment=False,
               segmentation_phases=None,
               minimum_valve_voxel_height=None,
               export_landmarks=False,
               landmark_phases=None,
               landmark_labels=None,
               landmark_label_phases=None,
               annulus_contour_label=False,
               annulus_contour_model=False,
               annulus_phases=None,
               additional_frame_range=None,
               run_quantification=False,
               save_log_file=False):
    """
    :param input_data: single MRB file, directory consisting of MRB files, or MRMLScene (currently open)
    """
    self.valveType = valve_type
    self.inputData = input_data

    self.cardiacPhases = cardiac_phase_frames

    self.outputDir = output_directory
    self.voxelSpacing = voxel_spacing
    self.volumeDimensions = volume_dimensions
    self.minValveHeightVxl = minimum_valve_voxel_height
    self.exportSegmentation = export_segmentation
    self.oneFilePerSegment = one_file_per_segment
    self.segmentationPhases = segmentation_phases if segmentation_phases else self.cardiacPhases

    self.annulusLabels = landmark_labels
    self.annulusLabelPhases = landmark_label_phases if landmark_label_phases else self.cardiacPhases

    self.exportLandmarks = export_landmarks
    self.landmarkPhases = landmark_phases if landmark_phases else self.cardiacPhases

    self.exportAnnulusContourLabel = annulus_contour_label
    self.exportAnnulusContourModel = annulus_contour_model
    self.annulusPhases = annulus_phases if annulus_phases else self.cardiacPhases

    self.additionalFrameRange = additional_frame_range
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
        computePhaseMetrics(self.cardiacPhases[0])
      self._composeExport()
      satisfied, messages = self._builder.requirementsSatisfied()
      if satisfied:
        self._builder.export()
      else:
        self.logger.error(messages)
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

  def _composeExport(self):
    referenceValveModel = getFirstValveModelNodeMatchingPhaseAndType(self.cardiacPhases[0], self._valveType)
    ExportItem.probeToRasTransform = cloneMRMLNode(referenceValveModel.getProbeToRasTransformNode())
    self.initializeReferenceVolumeNode(referenceValveModel)

    assert ExportItem.requiredAttributesSet()

    # TODO: ExportItems should clean up whatever was added during export
    self._builder = ExportBuilder()

    if self.additionalFrameRange:
      print(f"Adding addition frame range {self.additionalFrameRange} around {getValvePhaseShortName(referenceValveModel)}")
      self._builder.add_export_item(AdditionalFrames(referenceValveModel, self.additionalFrameRange))

    for valveModel in self._getRequestedValveModels():
      self._builder.add_export_item(PhaseFrame(valveModel, valveModel is referenceValveModel))
      phaseShortName = getValvePhaseShortName(valveModel)

      if (self.exportAnnulusContourLabel or self.exportAnnulusContourLabel) and phaseShortName in self.annulusPhases:
        self._builder.add_export_item(Annulus(valveModel, self.exportAnnulusContourLabel, self.exportAnnulusContourModel))
      if self.annulusLabels and phaseShortName in self.annulusLabelPhases:
        self._builder.add_export_item(LandmarkLabels(valveModel, self.annulusLabels))
      if self.exportLandmarks and phaseShortName in self.landmarkPhases:
        self._builder.add_export_item(Landmarks(valveModel))
      if self.exportSegmentation and phaseShortName in self.segmentationPhases:
        self._builder.add_export_item(Segmentation(valveModel, self.oneFilePerSegment))

  def _getRequestedValveModels(self):
    valveModels = getSpecificHeartValveModelNodesMatchingPhaseAndType(self.cardiacPhases, self._valveType)
    if len(valveModels) != len(self.cardiacPhases):
      raise ValueError(
        f"""
          Couldn't find valve models for cardiac phases {self.cardiacPhases} or found multiple in cardiac phase.
          Found valve models with phases (in order): {[getValvePhaseShortName(m) for m in valveModels]}
        """
      )
    return valveModels

  def initializeReferenceVolumeNode(self, valveModel):
    voxelSpacing = getFinalVoxelSpacing(self.voxelSpacing, self.minValveHeightVxl, valveModel)
    if voxelSpacing is not None and not type(voxelSpacing) is list:
      voxelSpacing = [voxelSpacing] * 3
    ExportItem.referenceVolumeNode = \
      ReferenceVolume.getNormalizedReferenceVolumeNode(valveModel, self.volumeDimensions, voxelSpacing)