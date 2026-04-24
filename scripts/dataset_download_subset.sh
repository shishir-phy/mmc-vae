#!/bin/bash


mkdir -p datasets

## song lyrics dataset download
wget -c https://storage.googleapis.com/kaggle-data-sets/2805070/4840139/compressed/song_lyrics.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260409%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260409T204525Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=b229510528190a726c157777eb1555ccad7e846a8b449b96c779d4f1a4a54ccb5ffe4ebdba2722c6be6feb410dc93f290883091d5149bf85a75b2b2303e87e3e2973f9adce0e62dec9376e64358db5f049f92cc3c004924ba985205fa70c1509700153812fd54e539d4ef0d05d5f05dc21d55af9ca89acee5f7a9692e6045502e8e34cfc61dfab56a845def3c586c4919f6464ffdfd8ae3f5cfb227eb5cbaf30cc7bb1409567f5cd09659d693349687e5e277ae57f6db63ffc265c1a0cde9d3cb516aaf0c1e684fe430fe28598f3a6186db137c31483becad7100c79689773e6cbfebcd3431e75630f3b3e7118d3efaf680e20fcc3d0e9aab1cf40e781d35fbf -O datasets/song_lyrics.csv.zip

## unzip the dataset
unzip datasets/song_lyrics.csv.zip

## check which languages are present
python scripts/count_languages.py

## Create subsets: bn:300, en:300
python scripts/create_subset.py
