for f in ./*.tar.gz; do
  tar -xvzf "$f" &
done
wait