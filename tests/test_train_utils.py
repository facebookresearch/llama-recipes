# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from unittest.mock import patch
import pytest

import torch

import os
import shutil

from llama_recipes.utils.train_utils import train

TEMP_OUTPUT_DIR = os.getcwd() + "/tmp"

@pytest.fixture(scope="session")
def temp_output_dir():
    # Create the directory during the session-level setup
    temp_output_dir = "tmp"
    os.mkdir(os.path.join(os.getcwd(), temp_output_dir))
    yield temp_output_dir
    # Delete the directory during the session-level teardown
    shutil.rmtree(temp_output_dir)

@pytest.fixture(scope="session")
def path_to_generate():
    path_to_generate = "some/random/path"
    yield path_to_generate
    
    shutil.rmtree(path_to_generate)


@patch("llama_recipes.utils.train_utils.MemoryTrace")
@patch("llama_recipes.utils.train_utils.nullcontext")
@patch("llama_recipes.utils.train_utils.torch.cuda.amp.GradScaler")
@patch("llama_recipes.utils.train_utils.torch.cuda.amp.autocast")
def test_gradient_accumulation(autocast, scaler, nullcontext, mem_trace, mocker):

    model = mocker.MagicMock(name="model")
    model().loss.__truediv__().detach.return_value = torch.tensor(1)
    mock_tensor = mocker.MagicMock(name="tensor")
    batch = {"input": mock_tensor}
    train_dataloader = [batch, batch, batch, batch, batch]
    eval_dataloader = None
    tokenizer = mocker.MagicMock()
    optimizer = mocker.MagicMock()
    lr_scheduler = mocker.MagicMock()
    gradient_accumulation_steps = 1
    train_config = mocker.MagicMock()
    train_config.enable_fsdp = False
    train_config.use_fp16 = False
    train_config.run_validation = False
    train_config.gradient_clipping = False
    train_config.save_metrics = False

    train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        lr_scheduler,
        gradient_accumulation_steps,
        train_config,
    )

    assert optimizer.zero_grad.call_count == 5
    optimizer.zero_grad.reset_mock()

    assert nullcontext.call_count == 5
    nullcontext.reset_mock()

    assert autocast.call_count == 0

    gradient_accumulation_steps = 2
    train_config.use_fp16 = True
    train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        lr_scheduler,
        gradient_accumulation_steps,
        train_config,
    )
    assert optimizer.zero_grad.call_count == 3
    assert nullcontext.call_count == 0
    assert autocast.call_count == 5

def test_save_to_json_file_exists(temp_output_dir, mocker):
    model = mocker.MagicMock(name="model")
    model().loss.__truediv__().detach.return_value = torch.tensor(1)
    mock_tensor = mocker.MagicMock(name="tensor")
    batch = {"input": mock_tensor}
    train_dataloader = [batch, batch, batch, batch, batch]
    eval_dataloader = None
    tokenizer = mocker.MagicMock()
    optimizer = mocker.MagicMock()
    lr_scheduler = mocker.MagicMock()
    gradient_accumulation_steps = 1
    train_config = mocker.MagicMock()
    train_config.enable_fsdp = False
    train_config.use_fp16 = False
    train_config.run_validation = False
    train_config.gradient_clipping = False
    train_config.save_metrics = True
    train_config.output_dir = temp_output_dir

    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        lr_scheduler,
        gradient_accumulation_steps,
        train_config,
        local_rank=0
    )

    assert results["metrics_filename"] not in ["", None]
    assert os.path.isfile(results["metrics_filename"])

def test_save_to_json_filen_name_local_rank_0(temp_output_dir, mocker):
    model = mocker.MagicMock(name="model")
    model().loss.__truediv__().detach.return_value = torch.tensor(1)
    mock_tensor = mocker.MagicMock(name="tensor")
    batch = {"input": mock_tensor}
    train_dataloader = [batch, batch, batch, batch, batch]
    eval_dataloader = None
    tokenizer = mocker.MagicMock()
    optimizer = mocker.MagicMock()
    lr_scheduler = mocker.MagicMock()
    gradient_accumulation_steps = 1
    train_config = mocker.MagicMock()
    train_config.enable_fsdp = False
    train_config.use_fp16 = False
    train_config.run_validation = False
    train_config.gradient_clipping = False
    train_config.save_metrics = True
    train_config.output_dir = temp_output_dir

    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        lr_scheduler,
        gradient_accumulation_steps,
        train_config
    )

    assert "None" not in results["metrics_filename"]
    assert "metrics_data_0" in results["metrics_filename"]

def test_save_to_json_filen_name_local_rank_1(temp_output_dir, mocker):
    model = mocker.MagicMock(name="model")
    model().loss.__truediv__().detach.return_value = torch.tensor(1)
    mock_tensor = mocker.MagicMock(name="tensor")
    batch = {"input": mock_tensor}
    train_dataloader = [batch, batch, batch, batch, batch]
    eval_dataloader = None
    tokenizer = mocker.MagicMock()
    optimizer = mocker.MagicMock()
    lr_scheduler = mocker.MagicMock()
    gradient_accumulation_steps = 1
    train_config = mocker.MagicMock()
    train_config.enable_fsdp = False
    train_config.use_fp16 = False
    train_config.run_validation = False
    train_config.gradient_clipping = False
    train_config.save_metrics = True
    train_config.output_dir = temp_output_dir

    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        lr_scheduler,
        gradient_accumulation_steps,
        train_config,
        local_rank=1
    )

    assert "None" not in results["metrics_filename"]
    assert "metrics_data_1" in results["metrics_filename"]

def test_save_to_json_folder_exists(path_to_generate, mocker):
    model = mocker.MagicMock(name="model")
    model().loss.__truediv__().detach.return_value = torch.tensor(1)
    mock_tensor = mocker.MagicMock(name="tensor")
    batch = {"input": mock_tensor}
    train_dataloader = [batch, batch, batch, batch, batch]
    eval_dataloader = None
    tokenizer = mocker.MagicMock()
    optimizer = mocker.MagicMock()
    lr_scheduler = mocker.MagicMock()
    gradient_accumulation_steps = 1
    train_config = mocker.MagicMock()
    train_config.enable_fsdp = False
    train_config.use_fp16 = False
    train_config.run_validation = False
    train_config.gradient_clipping = False
    train_config.save_metrics = True
    train_config.output_dir = path_to_generate

    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        lr_scheduler,
        gradient_accumulation_steps,
        train_config
    )
    
    assert os.path.isfile(results["metrics_filename"])


