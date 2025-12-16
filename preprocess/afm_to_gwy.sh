mkdir -p /home/david/Documents/ml_projects/autoencoder/data/data_gwy

for f in /home/david/Documents/ml_projects/autoencoder/data/data_afm/*.AFM; do
  base="$(basename "$f" .AFM)"
  gwyddion --convert-to-gwy="/home/david/Documents/ml_projects/autoencoder/data/data_gwy/${base}.gwy" "$f"
done
