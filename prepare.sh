set -x
set -euo pipefail

download_model_dir="./fastspeech2-en-ljspeech"
model_hub_name="facebook/fastspeech2-en-ljspeech"

function download_model() {
  python ./Download_model.py \
    --model_name ${model_hub_name} \
    --local_dir ./${download_model_dir}
}

function download_nltk_data() {
  rm -rf ./3rdparty
  mkdir -p ./3rdparty/nltk/corpora
  mkdir -p ./3rdparty/nltk/taggers

  cd ./3rdparty/nltk/corpora
  wget -c https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/cmudict.zip
  ls
  unzip ./cmudict.zip
  cd -

  cd ./3rdparty/nltk/taggers
  wget -c https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip
  unzip ./averaged_perceptron_tagger.zip
  cd -
}

# step1. download huggingface model
download_model

# step2. download nltk data
download_nltk_data
