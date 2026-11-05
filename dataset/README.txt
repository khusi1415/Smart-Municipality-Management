Put training images here (50-150 per class is a good start), then run:  python ai/train.py

dataset/
  garbage/         photos of garbage piles / overflowing dustbins
  pothole/         photos of potholes / damaged roads
  street_light/    photos of broken or dark street lights / poles
  water_leakage/   photos of leaking / burst pipes, water on road

Sources: Kaggle datasets, Google Images (for study use), or your own phone photos.
After training, ai/model.pkl is created and the app starts using image AI automatically.
