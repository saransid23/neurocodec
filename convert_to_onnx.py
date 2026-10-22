"""
NeuroCodec: ONNX Export & Edge AI Optimization
Exports NeuroCodec components (Encoder+Importance, Decoder, and End-to-End)
to ONNX format with dynamic batch and sequence axes, followed by ONNX Runtime CPU benchmarking.
"""

import argparse
import os
import time
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml

from neurocodec import NeuroCodec


class OnnxEncoderWrapper(nn.Module):
    """
    ONNX-friendly wrapper combining AudioEncoder and ImportanceNetwork.
    """

    def __init__(self, model: NeuroCodec):
        super().__init__()
        self.encoder = model.encoder
        self.importance_net = model.importance_net

    def forward(self, audio: torch.Tensor):
        z = self.encoder(audio)
        mask, counts, scores = self.importance_net(z)
        return z, scores, mask


class OnnxDecoderWrapper(nn.Module):
    """
    ONNX-friendly wrapper for AudioDecoder.
    """

    def __init__(self, model: NeuroCodec):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, z_q: torch.Tensor):
        return self.decoder(z_q)


class OnnxEndToEndWrapper(nn.Module):
    """
    ONNX-friendly end-to-end model for edge deployment.
    """

    def __init__(self, model: NeuroCodec):
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor):
        out = self.model(audio)
        return out["reconstructed"], out["importance_scores"]


def export_and_benchmark(
    checkpoint_path: str = None,
    config_path: str = "config.yaml",
    output_dir: str = "onnx_models",
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load config and instantiate model
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    audio_cfg = config.get("audio", {})
    model_cfg = config.get("model", {})
    sample_rate = audio_cfg.get("sample_rate", 16000)

    model = NeuroCodec(
        sample_rate=sample_rate,
        latent_dim=model_cfg.get("encoder", {}).get("latent_dim", 512),
        hidden_dim=model_cfg.get("encoder", {}).get("hidden_dim", 64),
        num_quantizers=model_cfg.get("rvq", {}).get("num_quantizers", 8),
        codebook_size=model_cfg.get("rvq", {}).get("codebook_size", 1024),
        encoder_strides=tuple(model_cfg.get("encoder", {}).get("strides", [2, 4, 5, 8])),
        decoder_strides=tuple(model_cfg.get("decoder", {}).get("strides", [8, 5, 4, 2])),
    )
    model.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Loaded weights from {checkpoint_path}")

    # 2. Export End-to-End Codec
    e2e_path = os.path.join(output_dir, "neurocodec_e2e.onnx")
    enc_path = os.path.join(output_dir, "neurocodec_encoder.onnx")
    dec_path = os.path.join(output_dir, "neurocodec_decoder.onnx")

    dummy_audio = torch.randn(1, 1, 16000, dtype=torch.float32)

    # Export Encoder + Importance
    print(f"Exporting NeuroCodec Encoder to ONNX: {enc_path}...")
    enc_model = OnnxEncoderWrapper(model)
    torch.onnx.export(
        enc_model,
        dummy_audio,
        enc_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["audio_input"],
        output_names=["latent", "importance_scores", "quantizer_mask"],
        dynamic_axes={
            "audio_input": {0: "batch_size", 2: "num_samples"},
            "latent": {0: "batch_size", 2: "num_frames"},
            "importance_scores": {0: "batch_size", 1: "num_frames"},
            "quantizer_mask": {0: "batch_size", 2: "num_frames"},
        },
        dynamo=False,
    )

    # Export Decoder
    dummy_latent = torch.randn(1, 512, 50, dtype=torch.float32)
    print(f"Exporting NeuroCodec Decoder to ONNX: {dec_path}...")
    dec_model = OnnxDecoderWrapper(model)
    torch.onnx.export(
        dec_model,
        dummy_latent,
        dec_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["latent_input"],
        output_names=["reconstructed_audio"],
        dynamic_axes={
            "latent_input": {0: "batch_size", 2: "num_frames"},
            "reconstructed_audio": {0: "batch_size", 2: "num_samples"},
        },
        dynamo=False,
    )

    # Export End-to-End
    print(f"Exporting End-to-End NeuroCodec to ONNX: {e2e_path}...")
    e2e_model = OnnxEndToEndWrapper(model)
    torch.onnx.export(
        e2e_model,
        dummy_audio,
        e2e_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["audio_input"],
        output_names=["reconstructed_audio", "importance_scores"],
        dynamic_axes={
            "audio_input": {0: "batch_size", 2: "num_samples"},
            "reconstructed_audio": {0: "batch_size", 2: "num_samples"},
            "importance_scores": {0: "batch_size", 1: "num_frames"},
        },
        dynamo=False,
    )

    # Validate ONNX models
    for p in [enc_path, dec_path, e2e_path]:
        proto = onnx.load(p)
        onnx.checker.check_model(proto)
        print(f"Verified {os.path.basename(p)} ({os.path.getsize(p) / (1024 * 1024):.2f} MB)")

    # 3. Benchmark ONNX Runtime on CPU
    print("\nBenchmarking ONNX Runtime Inference on CPU (1.0-sec audio @ 16 kHz)...")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = min(os.cpu_count() or 4, 8)

    # Benchmark Encoder
    sess_enc = ort.InferenceSession(enc_path, sess_opts, providers=["CPUExecutionProvider"])
    inp_name_enc = sess_enc.get_inputs()[0].name
    test_audio = np.random.randn(1, 1, 16000).astype(np.float32)
    for _ in range(5):
        sess_enc.run(None, {inp_name_enc: test_audio})
    enc_timings = []
    for _ in range(25):
        t0 = time.perf_counter()
        sess_enc.run(None, {inp_name_enc: test_audio})
        enc_timings.append((time.perf_counter() - t0) * 1000.0)
    avg_enc_lat = float(np.mean(enc_timings))

    # Benchmark Decoder
    sess_dec = ort.InferenceSession(dec_path, sess_opts, providers=["CPUExecutionProvider"])
    inp_name_dec = sess_dec.get_inputs()[0].name
    test_latent = np.random.randn(1, 512, 50).astype(np.float32)
    for _ in range(5):
        sess_dec.run(None, {inp_name_dec: test_latent})
    dec_timings = []
    for _ in range(25):
        t0 = time.perf_counter()
        sess_dec.run(None, {inp_name_dec: test_latent})
        dec_timings.append((time.perf_counter() - t0) * 1000.0)
    avg_dec_lat = float(np.mean(dec_timings))

    # Benchmark E2E
    sess_e2e = ort.InferenceSession(e2e_path, sess_opts, providers=["CPUExecutionProvider"])
    inp_name_e2e = sess_e2e.get_inputs()[0].name
    for _ in range(5):
        sess_e2e.run(None, {inp_name_e2e: test_audio})
    e2e_timings = []
    for _ in range(25):
        t0 = time.perf_counter()
        sess_e2e.run(None, {inp_name_e2e: test_audio})
        e2e_timings.append((time.perf_counter() - t0) * 1000.0)
    avg_e2e_lat = float(np.mean(e2e_timings))

    print(f"\n================ ONNX CPU BENCHMARK SUMMARY ================")
    print(f"Encoder + Importance Latency: {avg_enc_lat:.2f} ms  (Target: < 100 ms) -> {'PASSED' if avg_enc_lat < 100 else 'CHECK'}")
    print(f"Decoder Latency:              {avg_dec_lat:.2f} ms  (Target: < 100 ms) -> {'PASSED' if avg_dec_lat < 100 else 'CHECK'}")
    print(f"End-to-End Pipeline Latency:  {avg_e2e_lat:.2f} ms  (RTF: {avg_e2e_lat / 1000.0:.3f}x)")
    print(f"============================================================")

    return {
        "encoder_path": enc_path,
        "decoder_path": dec_path,
        "e2e_path": e2e_path,
        "encoder_latency_ms": avg_enc_lat,
        "decoder_latency_ms": avg_dec_lat,
        "e2e_latency_ms": avg_e2e_lat,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output_dir", type=str, default="onnx_models")
    args = parser.parse_args()

    export_and_benchmark(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
    )
