# TTS FastSpeech2

[FastSpeech2](https://arxiv.org/pdf/2006.04558.pdf) is a deep learning model based on the Transformer structure, which
can directly generate speech from text in speech synthesis. FastSpeech2 was proposed by Microsoft Asia Research
Institute and Microsoft Azure Speech Team jointly with Zhejiang University.

FastSpeech2 is a text-to-speech synthesis model, an improved version
Quick voice. FastSpeech2 improves the training method of FastSpeech and improves the training speed and accuracy.
Compared with FastSpeech, FastSpeech2's speech synthesis is faster and can generate higher quality speech.

The model we will deploy fastspeech2-en-ljspeech can only convert English input text into speech.

## 1. Create a new Notebook Server

Create a new Notebook Server on the Kubeflow on vSphere platform.

- You can create your own custom image or use an image published by us here:

`projects.registry.vmware.com/models/notebook/hf-inference-deploy@sha256:8c5960ce436881f37336b12556d7a661ea20e4dbfe9ac193516cf384daa51c19`

- set 1 CPUs, 2GB memory for this Notebook Server.

## 2. Connect to the Notebook Server

Open a Terminal window. Pull the code of this project by running

`git clone https://github.com/blueskyztt/fastspeech2.git`

## 3. Download model

- (Option1) Download the model from https://huggingface.co/facebook/fastspeech2-en-ljspeech/tree/main and put all model
  files
  in `./fastspeech2-en-ljspeech` directory which under the current directory.

- (Option2) Alternatively, you can download the model with the script `Download_model.py`, an example is
  in `prepare.sh`. Note this script also does the work in Part4.2. If you excuted `bash prepare.sh`, you can skip the
  Part4.2 because Part4.2 is also finished by this script.

Finally, The directory structure is as follows:

```text
./fastspeech2-en-ljspeech
├── README.md
├── config.yaml
├── fbank_mfa_gcmvn_stats.npz
├── hifigan.bin
├── hifigan.json
├── pytorch_model.pt
├── run_fast_speech_2.py
└── vocab.txt
```

## 4. Preparation

### 4.1 Python requirements

Install the python packages necessary for the service, listed in `requirements.txt`.

```shell
pip install -r ./requirements.txt
```

### 4.2 NLTK files

- (Option1) The model prediction requires nltk's `cmudict.zip` and `averaged_perceptron_tagger.zip`. Due to the unstable
  network, manually download these compressed packages by yourself.
  Download archives from https://github.com/nltk/nltk_data/tree/gh-pages/packages and put them in `3rdparty/nltk` under
  the current directory.

- (Option2) Alternatively, you can also do this part with the script `prepare.sh`, just with below command.

```shell
bash prepare.sh
```

In short, you get the following directories and files finally.

```text
3rdparty
└── nltk
    ├── corpora
    │   ├── cmudict
    │   │   ├── README
    │   │   └── cmudict
    │   └── cmudict.zip
    └── taggers
        ├── averaged_perceptron_tagger
        │   └── averaged_perceptron_tagger.pickle
        └── averaged_perceptron_tagger.zip
```

Now, current directory should contain these files and directories:

```text
├── Download_model.py
├── prepare.sh
├── 3rdparty
├── README.md
├── client.py
├── create_mar.sh
├── fastspeech2-en-ljspeech
├── handler.py
├── requirements.txt
└── start_ts.sh
```

## 5. Create TorchServe Model Archiver File

```shell
bash ./create_mar.sh
```

After waiting for a while, we got the mar file needed for the service. Then we can start our service.

## 6. Start TorchServe

Now you can start service with below command.

```shell
bash ./start_ts.sh
```

## 7. Test Service

Request the service in the terminal, execute

```shell
python ./lr_client.py
```

When you see 'Successfully generated output_client.wav', indicating that the requested service is successful. In the
current directory, you can find the file `output_client.wav`. Open `output_client.wav` and you can hear the sound
generated by the model.

