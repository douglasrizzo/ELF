# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this is a simpler version of run.py, but it also doesn't work

import argparse
import os
import sys
from datetime import datetime

from rlpytorch import (ArgsProvider, ModelInterface, ModelLoader, Sampler,
                       Trainer)
from rlpytorch.model_loader import load_module
from rlpytorch.runner.multi_process import MultiProcessRun
from rlpytorch.runner.single_process import SingleProcessRun
from rlpytorch.stats import RewardCount, WinRate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    use_multi_process = int(os.environ.get("multi_process", 0))

    sampler = Sampler()
    trainer = Trainer()
    game = load_module(os.environ["game"]).Loader()
    runner = MultiProcessRun() if use_multi_process else SingleProcessRun()
    model_file = load_module(os.environ["model_file"])
    model_class, method_class = model_file.Models[os.environ["model"]]

    model_loader = ModelLoader(model_class)
    method = method_class()

    args_providers = [sampler, trainer, game, runner, model_loader, method]

    all_args = ArgsProvider.Load(parser, args_providers)

    GC = game.initialize()
    # GC.setup_gpu(0)
    all_args.method_class = method_class

    model = model_loader.load_model(GC.params)
    mi = ModelInterface()
    mi.add_model("model", model)
    # , opt=True, params={"lr": 0.001}
    mi.add_model("actor", model, copy=True, cuda=False)
    # method.set_model_interface(mi)

    trainer.setup(sampler=sampler, mi=mi, rl_method=method)

    def train_and_update(sel, sel_gpu, reply):
        trainer.train(sel, sel_gpu, reply)

        # if trainer.just_updated and eval_process is not None:
        #     eval_process.update_model("actor", mi["actor"])

    GC.reg_callback("train", train_and_update)
    GC.reg_callback("actor", trainer.actor)
    runner.setup(GC, episode_summary=trainer.episode_summary, episode_start=trainer.episode_start)

    runner.run()
