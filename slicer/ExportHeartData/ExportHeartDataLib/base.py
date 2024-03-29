from abc import ABC, abstractmethod
import logging
import slicer
from ExportHeartDataLib.constants import APPLICATION_NAME
from ExportHeartDataLib.summary import ExportSummary
from ExportHeartDataLib.exceptions import NoAssociatedFrameNumberFound


class ExportBuilder(object):

  def __init__(self):
    self._items = []

  def add_export_item(self, item) -> None:
    assert isinstance(item, ExportItem)
    self._items.append(item)

  def export(self):
    for item in self._items:
      item()

  def requirementsSatisfied(self):
    satisfied = []
    messages = []
    for item in self._items:
      valid, message = item.verify()
      satisfied.append(valid)
      if message:
        messages.append(message)
    return all(satisfied), messages


class ExportItem(ABC):

  exportSummarizer = None
  prefix = None
  outputDir = None
  referenceVolumeNode = None
  probeToRasTransform = None

  @classmethod
  def getAssociatedFrameNumber(cls, valveModel):
    frameNumber = valveModel.getValveVolumeSequenceIndex()
    if frameNumber == -1:
      raise NoAssociatedFrameNumberFound(f"No associated frame number found for {valveModel.getCardiacCyclePhase()}")
    return frameNumber

  @classmethod
  def cleanup(cls):
    if cls.probeToRasTransform is not None:
      slicer.mrmlScene.RemoveNode(cls.probeToRasTransform)
      cls.probeToRasTransform = None
    if cls.referenceVolumeNode is not None:
      slicer.mrmlScene.RemoveNode(cls.referenceVolumeNode)
      cls.referenceVolumeNode = None

  @classmethod
  def requiredAttributesSet(cls):
    return cls.outputDir is not None and \
           cls.referenceVolumeNode is not None and \
           cls.probeToRasTransform is not None and \
           cls.prefix is not None

  @property
  def phase(self):
    assert self._valveModel is not None
    if self._phase:
      return self._phase
    from HeartValveLib.helpers import getValvePhaseShortName
    return getValvePhaseShortName(self._valveModel)

  @staticmethod
  def getLogger():
    return logging.getLogger(APPLICATION_NAME)

  @classmethod
  def setExportSummarizer(cls, summarizer):
    assert isinstance(summarizer, ExportSummary)
    cls.exportSummarizer = summarizer

  @classmethod
  def saveNode(cls, node, outputFile, kind=None):
    node.SetAndObserveTransformNodeID(cls.probeToRasTransform.GetID())
    slicer.modules.transforms.logic().hardenTransform(node)
    if cls.exportSummarizer:
      assert kind is not None
      cls.exportSummarizer.add_export_item(kind, outputFile)
    return slicer.util.saveNode(node, str(outputFile))

  def __init__(self, valveModel, phase=None, suffix=""):
    self._valveModel = valveModel
    self._phase = phase
    self._suffix = suffix

  @abstractmethod
  def __call__(self):
    # before calling this, needs to check if necessary attributes were set
    pass

  @abstractmethod
  def verify(self):
    pass