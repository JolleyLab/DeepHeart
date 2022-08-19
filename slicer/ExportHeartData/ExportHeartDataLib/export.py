import os
import shutil
import logging
from pathlib import Path

import slicer

from ExportHeartDataLib.constants import APPLICATION_NAME
from ExportHeartDataLib.items import Segmentation, Annulus, PhaseFrame, Landmarks, LandmarkLabels
from ExportHeartDataLib.utils import cloneMRMLNode, computePhaseMetrics, getRandomDirectoryName, getFinalVoxelSpacing
from ExportHeartDataLib.utils import getValveModelsWithPhaseNames
from ExportHeartDataLib.base import ExportBuilder, ExportItem
from ExportHeartDataLib.summary import ExportSummary

import HeartValveLib
from HeartValveLib.helpers import (
  getSpecificHeartValveModelNodesMatchingPhaseAndType,
  getFirstValveModelNodeMatchingPhaseAndType,
  isMRBFile,
  getAllFilesWithExtension,
  getValvePhaseShortName
)

from typing import Optional


class Exporter(object):

  @staticmethod
  def initializeReferenceVolumeNode(valveModel, voxelSpacing, volumeDimensions):
    import ExportHeartDataLib.reference_volume as ReferenceVolume
    if voxelSpacing is not None and not type(voxelSpacing) is list:
      voxelSpacing = [voxelSpacing] * 3
    ExportItem.referenceVolumeNode = \
      ReferenceVolume.getNormalizedReferenceVolumeNode(valveModel, volumeDimensions, voxelSpacing)

  @staticmethod
  def getRequestedValveModels(cardiacPhases, valveType):
    valveModels = getSpecificHeartValveModelNodesMatchingPhaseAndType(cardiacPhases, valveType)
    if len(valveModels) != len(cardiacPhases):
      raise ValueError(
        f"""
            Couldn't find valve models for cardiac phases {cardiacPhases} or found multiple in cardiac phase.
            Found valve models with phases (in order): {[getValvePhaseShortName(m) for m in valveModels]}
          """
      )
    return valveModels

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
               reference_phase : str,
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
               run_quantification=False,
               save_log_file=False):
    """
    :param input_data: single MRB file, directory consisting of MRB files, or MRMLScene (currently open)
    """
    self.valveType = valve_type
    self.inputData = input_data

    self.referencePhase = reference_phase
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

    self.runQuantification = run_quantification

    self.saveLogs = save_log_file
    ExportItem.setExportSummarizer(ExportSummary())

  def export(self):
    """
    :return: returning a dictionary with the output types and file paths
    """
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
      logFilePath = slicer.app.errorLogModel().filePath
      shutil.copy(logFilePath, self.outputDir / Path(logFilePath).name)
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
        computePhaseMetrics(self.referencePhase)
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
    except:
      ExportItem.cleanup()

  def _composeExport(self):
    referenceValveModel = getFirstValveModelNodeMatchingPhaseAndType(self.referencePhase, self._valveType)
    ExportItem.probeToRasTransform = cloneMRMLNode(referenceValveModel.getProbeToRasTransformNode())
    voxelSpacing = getFinalVoxelSpacing(self.voxelSpacing, self.minValveHeightVxl, referenceValveModel)
    self.initializeReferenceVolumeNode(referenceValveModel, voxelSpacing, self.volumeDimensions)

    assert ExportItem.requiredAttributesSet()

    self._builder = ExportBuilder()

    self._addCardiacFramesToExport(referenceValveModel)

    if self.exportAnnulusContourLabel or self.exportAnnulusContourModel:
      self._addAnnulusToExport()

    if self.annulusLabels:
      self._addLandmarkLabelsToExport()

    if self.exportLandmarks:
      self._addLandmarksToExport()

    if self.exportSegmentation:
      self._addSegmentationToExport()

  def _addCardiacFramesToExport(self, referenceValveModel):
    for valveDataClass in getValveModelsWithPhaseNames(self.cardiacPhases, self.valveType):
      self._builder.add_export_item(
        PhaseFrame(valveDataClass.valveModel,
                   valveDataClass.valveModel is referenceValveModel,
                   phase=valveDataClass.phase,
                   suffix=valveDataClass.suffix)
      )

  def _addAnnulusToExport(self):
    for valveDataClass in getValveModelsWithPhaseNames(self.annulusPhases, self.valveType):
      self._builder.add_export_item(
        Annulus(valveDataClass.valveModel,
                self.exportAnnulusContourLabel,
                self.exportAnnulusContourModel,
                phase=valveDataClass.phase,
                suffix=valveDataClass.suffix)
      )

  def _addLandmarkLabelsToExport(self):
    for valveDataClass in getValveModelsWithPhaseNames(self.annulusLabelPhases, self.valveType):
      self._builder.add_export_item(
        LandmarkLabels(valveDataClass.valveModel,
                       self.annulusLabels,
                       phase=valveDataClass.phase,
                       suffix=valveDataClass.suffix)
      )

  def _addLandmarksToExport(self):
    for valveDataClass in getValveModelsWithPhaseNames(self.landmarkPhases, self.valveType):
      self._builder.add_export_item(
        Landmarks(valveDataClass.valveModel,
                  phase=valveDataClass.phase,
                  suffix=valveDataClass.suffix)
      )

  def _addSegmentationToExport(self):
    for valveDataClass in getValveModelsWithPhaseNames(self.segmentationPhases, self.valveType):
      self._builder.add_export_item(
        Segmentation(valveDataClass.valveModel,
                     self.oneFilePerSegment,
                     phase=valveDataClass.phase,
                     suffix=valveDataClass.suffix)
      )

  def loadScene(self, mrbFile):
    try:
      slicer.app.ioManager().loadScene(mrbFile)
    except Exception as exc:
      self.logger.warning(exc)