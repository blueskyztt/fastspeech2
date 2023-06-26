set -x
set -euo pipefail

model_name="fastspeech2-en-ljspeech"
model_path="./fastspeech2-en-ljspeech"

export JAVA_HOME=$(echo ${JAVA_HOME})
export TMPDIR="$(pwd)/temp"
export TEMP="$(pwd)/temp"
mkdir -p ${TMPDIR}

function create_mar_file() {
  handle_file="./handler.py"

  rm -rf logs
  temp_dir=$(mktemp -d)
  echo ${temp_dir}
  ln -s $(pwd)/${model_path} ${temp_dir}
  ln -s $(pwd)/3rdparty ${temp_dir}

  mkdir -p ./model_store
  torch-model-archiver \
    --model-name ${model_name} \
    --version 1.0 \
    --handler ${handle_file} \
    --export-path ./model_store \
    --extra-files ${temp_dir} \
    --force
}

create_mar_file
echo "create mar file finished."