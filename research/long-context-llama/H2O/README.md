## Run Llama with H2O for long context inference

The following example runs inference of Llama-2-7b on XSUM summarization tasks. We're using `--enable_h2o_generation` to enable H2O algorithm that only keeps heavy-hitter and the local KV pairs. Use `--num_heavy_hitter_tokens` to decide the number of heavy-hitter KV pairs and `--num_window_length `for the KV cache size. The number of local KV pairs equals num_window_length - num_heavy_hitter_tokens. Also, use --enable_position_rolling to enable position rolling in the KV cache size that assign the positions in the KV cache instead of the ones in original sequences. Enabling postional rolling is important when sequence length exceeds the pretrained context windows, e.g., 4K in Llama-2.

```
python run_summarization.py
--input-path data/summarization/xsum.jsonl \ # 5-shot inference on 1000 samples from XSUM dataset. Other option: cnn_dailymail.jsonl.
--output-path summarization_output/xsum_h2o.jsonl \ # output path for generated sequence
--model-name meta-llama/Llama-2-7b-hf \ # models
--enable_h2o_generation 
```
