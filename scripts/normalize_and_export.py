"""

usage: Slicer --python-script normalize_and_export.py -i {inputDir} -o {outputDir}

"""

import os
import sys
import argparse


def main(argv):
  parser = argparse.ArgumentParser(description="Normalize and export data from heart valves")
  parser.add_argument("-i", "--input_data", metavar="PATH", required=True,
                      help="input data (single .mrb file or a output directory consisting of .mrb files)")
  parser.add_argument("-o", "--output_directory", metavar="PATH", required=True,
                      help="data output directory")
  parser.add_argument('-dim', "--volume_dimensions", metavar='INT', type=int, nargs=3,
                      help='desired dimensions of the output volumes')
  parser.add_argument("-vx", "--voxel_spacing", metavar="FLOAT", type=float, help="desired voxel spacing to be used")
  parser.add_argument("-vh", "--minimum_valve_voxel_height", metavar="HEIGHT", type=int, help="min valve voxel height")
  parser.add_argument("-ph", "--cardiac_phase_frames", metavar="PHASE_SHORTNAME", type=str, nargs="+", default=["MS"],
                      help="cardiac phases which will be exported")
  parser.add_argument("-rp", "--reference_phase", metavar="STR", type=str,
                      help="phase to be used as reference phase frame")
  parser.add_argument("-am", "--annulus_contour_model", action='store_true', help="export annulus contour model (.vtk)")
  parser.add_argument("-al", "--annulus_contour_label", action='store_true', help="annulus contour label (.nii.gz")
  parser.add_argument("-l", "--export_annulus_landmarks", action='store_true', help="export annulus landmarks (.fcsv)")
  parser.add_argument("-lp", "--landmark_phases", metavar="PHASE_SHORTNAME", type=str, nargs="+",
                      help="cardiac phases from which landmarks get exported")
  parser.add_argument("-ap", "--annulus_phases", metavar="PHASE_SHORTNAME", type=str, nargs="+",
                      help="cardiac phases from which annulus specific data (labels, contour, landmarks)")
  parser.add_argument("-ll", "--landmark_labels", metavar="LANDMARK_LABELS", type=str, nargs="+",
                      help="export specific annulus landmark(s) as labels to sub-directory equivalent to landmark name")
  parser.add_argument("-llp", "--landmark_label_phases", metavar="PHASE_SHORTNAME", type=str, nargs="+",
                      help="cardiac phases from which landmarks labels will get exported")
  parser.add_argument('-t', "--valve_type", metavar='STRING', type=str,
                      help='valve type of the input data (used for sub-directory creation)')
  parser.add_argument("-qf", "--run_quantification", action='store_true', help="run quantification before exporting")
  parser.add_argument("-d", "--debug", action='store_true', help="run python debugger upon Slicer start")
  parser.add_argument("-s", "--export_segmentations", action='store_true',
                      help="Export segmentation. By default, all segments are saved into a single segmentation file")
  parser.add_argument("-sf", "--one_file_per_segment", action='store_true',
                      help="Split segments into separate segmentation files")
  parser.add_argument("-sp", "--segmentation_phases", metavar="PHASE_SHORTNAME", type=str, nargs="+",
                      help="cardiac phases from which segmentations should be exported")
  args = parser.parse_args(argv)

  import slicer

  if args.debug:
    slicer.app.layoutManager().selectModule("PyDevRemoteDebug")
    w = slicer.modules.PyDevRemoteDebugWidget
    w.connectButton.click()

  from ExportHeartDataLib.utils import generateDirectoryName
  output_directory = generateDirectoryName(args.valve_type, args.volume_dimensions, args.minimum_valve_voxel_height)
  output_directory = os.path.join(args.output_directory, output_directory)

  from ExportHeartDataLib.export import Exporter
  exporter = Exporter(valve_type=args.valve_type,
                      input_data=args.input_data,
                      output_directory=output_directory,
                      reference_phase=args.reference_phase,
                      cardiac_phase_frames=args.cardiac_phase_frames,
                      voxel_spacing=args.voxel_spacing,
                      volume_dimensions=args.volume_dimensions,
                      export_segmentation=args.export_segmentations or args.one_file_per_segment,
                      one_file_per_segment=args.one_file_per_segment,
                      segmentation_phases=args.segmentation_phases,
                      minimum_valve_voxel_height=args.minimum_valve_voxel_height,
                      landmark_labels=args.landmark_labels,
                      export_landmarks=args.export_annulus_landmarks,
                      landmark_phases=args.landmark_phases,
                      annulus_contour_label=args.annulus_contour_label,
                      annulus_contour_model=args.annulus_contour_model,
                      annulus_phases=args.cardiac_phase_frames,
                      landmark_label_phases=args.landmark_label_phases,
                      run_quantification=args.run_quantification)
  exporter.export()

  sys.exit(0)


if __name__ == "__main__":
  main(sys.argv[1:])


