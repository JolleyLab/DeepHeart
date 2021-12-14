

class ExportSummary(object):

  def __init__(self):
    self._export_info = dict()

  def add_export_item(self, name, path):
    if not name in self._export_info.keys():
      self._export_info[name] = []
    self._export_info[name].append(str(path))

  def get_summary(self):
    return self._export_info