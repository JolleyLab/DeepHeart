for filename in $1/*.pth; do
  python test.py -r $filename
done
