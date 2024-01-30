#-----------------------------------------------------------------------------
set(MODULE_NAME ExportHeartData)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ExportHeartDataLib/__init__
  ExportHeartDataLib/base
  ExportHeartDataLib/exceptions
  ExportHeartDataLib/constants
  ExportHeartDataLib/export
  ExportHeartDataLib/items
  ExportHeartDataLib/reference_volume
  ExportHeartDataLib/segmentation_utils
  ExportHeartDataLib/summary
  ExportHeartDataLib/utils
  Testing/ExportHeartDataTest
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  WITH_GENERIC_TESTS
  )