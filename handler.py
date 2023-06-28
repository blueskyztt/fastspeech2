#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
import time
from pathlib import Path

import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.models.text_to_speech import TTSHubInterface
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class fastspeech2Handler(BaseHandler):
    def __init__(self):
        super().__init__()

        self.model_name = "fastspeech2-en-ljspeech"

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = (
            "cuda"
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest
        logger.info("========properties:" + json.dumps(properties, ensure_ascii=False))
        logger.info("========manifest:" + json.dumps(self.manifest, ensure_ascii=False))
        model_dir = os.path.join(properties.get("model_dir"), self.model_name)
        logger.info(f"========model_dir:{model_dir}")

        self._prepare_nltk()
        self._init_model(model_dir)

        self.initialized = True

    def preprocess(self, data):
        data = data[0]["body"].decode("utf-8")
        data = json.loads(data)
        logger.info(f"data:{data}")
        return data

    def inference(self, data, *args, **kwargs):
        logger.info(f"====torch.cuda.is_available():{torch.cuda.is_available()}")
        with torch.no_grad():
            result = self.generate(text=data)
        return result

    def postprocess(self, data):
        return data

    def handle(self, data, context):
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        data = self.preprocess(data)
        output = self.inference(data)
        output = self.postprocess(output)

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        return output

    def _init_model(self, model_dir):
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            model_dir,
            arg_overrides={"vocoder": "hifigan", "fp16": False}
        )
        TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
        generator = task.build_generator(models, cfg)

        self.models = models
        self.cfg = cfg
        self.task = task
        self.generator = generator

    def _prepare_nltk(self):
        home_path = os.environ['HOME']
        assert home_path is not None
        nltk_data_path = f"{home_path}/nltk_data"
        if not os.path.isdir(nltk_data_path):
            os.makedirs(nltk_data_path)
        cmd = f"cp -f -r ./3rdparty/nltk/* {nltk_data_path}"
        logger.info(f"cmd:{cmd}")
        os.system(cmd)

    def generate(self, text):
        task = self.task
        model = self.models[0]
        generator = self.generator

        sample = TTSHubInterface.get_model_input(task, text)
        wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
        wav = wav.numpy().tolist()
        ret = {"wav": wav, "rate": rate}
        ret = [ret]
        return ret


def load_model_ensemble_and_task_from_hf_hub(model_dir, arg_overrides=None):
    cache_dir = model_dir
    _arg_overrides = arg_overrides or {}
    _arg_overrides["data"] = cache_dir
    return load_model_ensemble_and_task(
        [p.as_posix() for p in Path(cache_dir).glob("*.pt")],
        arg_overrides=_arg_overrides,
    )
