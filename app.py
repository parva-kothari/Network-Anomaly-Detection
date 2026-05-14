import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")


class MoEModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_p: float = 0.4):
        super().__init__()
        self.expert1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.expert2 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.expert3 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.gate = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 3))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.softmax(self.gate(x))
        stack = torch.stack([self.expert1(x), self.expert2(x), self.expert3(x)], dim=1)
        return (stack * gates.unsqueeze(-1)).sum(dim=1)


@dataclass
class MainMoEPredictor:
    model: MoEModel
    device: torch.device

    def predict(self, x_row: np.ndarray) -> Tuple[int, np.ndarray]:
        # x_row already arrives as shape (1, num_features) from DataFrame slicing.
        # Keep it 2D for Linear + BatchNorm1d layers.
        x = torch.from_numpy(x_row.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        return pred, probs


@dataclass
class BoostingWinnerPredictor:
    winner: str
    selected_features: List[str]
    feature_sets: Dict[str, List[str]]
    stacking_base_models: Dict
    stacking_meta_model: object
    stacking_schema: Dict
    moe_expert1: object
    moe_expert2: object
    moe_expert3: object
    moe_gate: object

    def _align_gate_probs(self, raw_probs: np.ndarray, gate_classes: np.ndarray, n_experts: int = 3) -> np.ndarray:
        out = np.zeros((raw_probs.shape[0], n_experts), dtype=np.float64)
        for i, cls in enumerate(gate_classes):
            out[:, int(cls)] = raw_probs[:, i]
        return out

    def _predict_stacking(self, row_df: pd.DataFrame) -> Tuple[int, np.ndarray]:
        x = row_df[self.selected_features].values
        blocks = [self.stacking_base_models[name].predict_proba(x) for name in self.stacking_schema["base_model_names"]]
        meta_x = np.hstack(blocks)
        probs = self.stacking_meta_model.predict_proba(meta_x)[0]
        pred = int(np.argmax(probs))
        return pred, probs

    def _predict_moe(self, row_df: pd.DataFrame) -> Tuple[int, np.ndarray]:
        x_full = row_df[self.selected_features].values
        x1 = row_df[self.feature_sets["expert1"]].values
        x2 = row_df[self.feature_sets["expert2"]].values
        x3 = row_df[self.feature_sets["expert3"]].values

        raw_gate = self.moe_gate.predict_proba(x_full)
        gate_probs = self._align_gate_probs(raw_gate, self.moe_gate.classes_, n_experts=3)

        p1 = self.moe_expert1.predict_proba(x1)
        p2 = self.moe_expert2.predict_proba(x2)
        p3 = self.moe_expert3.predict_proba(x3)

        probs = (gate_probs[:, [0]] * p1 + gate_probs[:, [1]] * p2 + gate_probs[:, [2]] * p3)[0]
        pred = int(np.argmax(probs))
        return pred, probs

    def predict(self, row_df: pd.DataFrame) -> Tuple[int, np.ndarray]:
        if self.winner == "stacking":
            return self._predict_stacking(row_df)
        return self._predict_moe(row_df)


@st.cache_data
def load_test_data() -> pd.DataFrame:
    path = os.path.join(SPLITS_DIR, "test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing test split: {path}")
    return pd.read_csv(path)


@st.cache_data
def load_label_data() -> Tuple[List[str], int]:
    le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    class_map_path = os.path.join(MODELS_DIR, "class_mapping.json")
    if os.path.exists(le_path):
        le = joblib.load(le_path)
        class_names = le.classes_.tolist()
    elif os.path.exists(class_map_path):
        with open(class_map_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        class_names = [mapping[str(i)] for i in sorted(int(k) for k in mapping.keys())]
    else:
        raise FileNotFoundError("Missing both label_encoder.pkl and class_mapping.json in models/.")
    return class_names, len(class_names)


@st.cache_resource
def load_main_moe_predictor(input_dim: int, num_classes: int) -> MainMoEPredictor:
    ckpt_path = os.path.join(MODELS_DIR, "moe_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoEModel(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return MainMoEPredictor(model=model, device=device)


@st.cache_resource
def load_boosting_predictor() -> BoostingWinnerPredictor:
    required = [
        "best_model_meta.pkl",
        "selected_features.pkl",
        "expert_feature_sets.pkl",
        "stacking_base_models.pkl",
        "stacking_meta_model.pkl",
        "stacking_schema.pkl",
        "moe_expert1_xgb.pkl",
        "moe_expert2_rf.pkl",
        "moe_expert3_et.pkl",
        "moe_gate_mlp.pkl",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(MODELS_DIR, name))]
    if missing:
        raise FileNotFoundError(f"Missing boosting artifacts: {', '.join(missing)}")

    best_model_meta = joblib.load(os.path.join(MODELS_DIR, "best_model_meta.pkl"))
    winner = best_model_meta.get("winner_model", "stacking")

    return BoostingWinnerPredictor(
        winner=winner,
        selected_features=joblib.load(os.path.join(MODELS_DIR, "selected_features.pkl")),
        feature_sets=joblib.load(os.path.join(MODELS_DIR, "expert_feature_sets.pkl")),
        stacking_base_models=joblib.load(os.path.join(MODELS_DIR, "stacking_base_models.pkl")),
        stacking_meta_model=joblib.load(os.path.join(MODELS_DIR, "stacking_meta_model.pkl")),
        stacking_schema=joblib.load(os.path.join(MODELS_DIR, "stacking_schema.pkl")),
        moe_expert1=joblib.load(os.path.join(MODELS_DIR, "moe_expert1_xgb.pkl")),
        moe_expert2=joblib.load(os.path.join(MODELS_DIR, "moe_expert2_rf.pkl")),
        moe_expert3=joblib.load(os.path.join(MODELS_DIR, "moe_expert3_et.pkl")),
        moe_gate=joblib.load(os.path.join(MODELS_DIR, "moe_gate_mlp.pkl")),
    )


def init_state() -> None:
    if "stream_idx" not in st.session_state:
        st.session_state.stream_idx = 0
    if "y_true_hist" not in st.session_state:
        st.session_state.y_true_hist = []
    if "y_pred_hist" not in st.session_state:
        st.session_state.y_pred_hist = []
    if "last_probs" not in st.session_state:
        st.session_state.last_probs = None
    if "last_true" not in st.session_state:
        st.session_state.last_true = None
    if "last_pred" not in st.session_state:
        st.session_state.last_pred = None
    if "last_signatures" not in st.session_state:
        st.session_state.last_signatures = []
    if "signature_events" not in st.session_state:
        st.session_state.signature_events = []
    if "mitigated_flows" not in st.session_state:
        st.session_state.mitigated_flows = 0


def reset_state() -> None:
    st.session_state.stream_idx = 0
    st.session_state.y_true_hist = []
    st.session_state.y_pred_hist = []
    st.session_state.last_probs = None
    st.session_state.last_true = None
    st.session_state.last_pred = None
    st.session_state.last_signatures = []
    st.session_state.signature_events = []
    st.session_state.mitigated_flows = 0


def compute_signature_thresholds(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    cols_for_thresholds = [
        "sload",
        "dload",
        "spkts",
        "dpkts",
        "sloss",
        "dloss",
        "ct_dst_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        "synack",
        "ackdat",
        "ct_ftp_cmd",
        "ct_srv_src",
        "ct_srv_dst",
        "dbytes",
        "sbytes",
    ]
    thresholds: Dict[str, Dict[str, float]] = {}
    for c in cols_for_thresholds:
        if c in df.columns:
            thresholds[c] = {
                "q50": float(df[c].quantile(0.50)),
                "q75": float(df[c].quantile(0.75)),
                "q90": float(df[c].quantile(0.90)),
                "q95": float(df[c].quantile(0.95)),
                "q99": float(df[c].quantile(0.99)),
            }
    return thresholds


def evaluate_signatures(row: pd.Series, thresholds: Dict[str, Dict[str, float]]) -> List[Dict[str, str]]:
    hits: List[Dict[str, str]] = []

    def has(col: str) -> bool:
        return col in row.index

    def val(col: str, default: float = 0.0) -> float:
        return float(row[col]) if has(col) else default

    def one_hot_on(col: str) -> bool:
        return has(col) and float(row[col]) > 0.0

    def th(col: str, q: str, fallback: float = 0.0) -> float:
        return thresholds.get(col, {}).get(q, fallback)

    # Signature 1: FTP brute-force / abuse
    if (
        (one_hot_on("service_ftp") or one_hot_on("service_ftp-data"))
        and (val("is_ftp_login") > 0.0 or val("ct_ftp_cmd") >= th("ct_ftp_cmd", "q90", 1.0))
        and val("ct_srv_src") >= th("ct_srv_src", "q90", 0.0)
    ):
        hits.append(
            {
                "signature": "FTP Abuse Pattern",
                "severity": "High",
                "reason": "FTP service with elevated login/command activity from source.",
                "mitigation": "Temporarily block source at firewall, enforce FTP login throttling, and disable weak FTP auth.",
            }
        )

    # Signature 2: HTTP flood / volumetric web DoS
    if (
        one_hot_on("service_http")
        and val("sload") >= th("sload", "q95", 0.0)
        and val("spkts") >= th("spkts", "q90", 0.0)
    ):
        hits.append(
            {
                "signature": "HTTP Flood Pattern",
                "severity": "High",
                "reason": "HTTP traffic with unusually high source load and packet rate.",
                "mitigation": "Apply rate limits, enable WAF challenge rules, and activate upstream DoS filtering profile.",
            }
        )

    # Signature 3: TCP handshake anomaly (possible SYN flood behavior)
    if (
        one_hot_on("proto_tcp")
        and val("synack") >= th("synack", "q95", 0.0)
        and val("ackdat") <= th("ackdat", "q50", 0.0)
        and val("ct_dst_src_ltm") >= th("ct_dst_src_ltm", "q90", 0.0)
    ):
        hits.append(
            {
                "signature": "SYN Flood-Like Pattern",
                "severity": "Medium",
                "reason": "TCP handshake timing imbalance with repeated destination-source interaction counts.",
                "mitigation": "Enable SYN cookies, tighten connection-rate limits, and blacklist offending source tuple for a short window.",
            }
        )

    # Signature 4: Horizontal scan / recon
    if (
        val("ct_dst_ltm") >= th("ct_dst_ltm", "q95", 0.0)
        and val("ct_dst_sport_ltm") >= th("ct_dst_sport_ltm", "q95", 0.0)
        and val("dbytes") <= th("dbytes", "q50", 0.0)
    ):
        hits.append(
            {
                "signature": "Recon Scan Pattern",
                "severity": "Medium",
                "reason": "Broad destination probing with low payload return traffic.",
                "mitigation": "Block scanning source, add IDS scan signature, and increase logging on targeted destination ranges.",
            }
        )

    # Signature 5: High-loss destabilization
    if val("sloss") >= th("sloss", "q95", 0.0) or val("dloss") >= th("dloss", "q95", 0.0):
        hits.append(
            {
                "signature": "High Packet-Loss Pattern",
                "severity": "Medium",
                "reason": "Packet loss is in the extreme tail compared with baseline traffic.",
                "mitigation": "Quarantine flow path, inspect network middleboxes, and trigger incident review for potential disruption attack.",
            }
        )

    return hits


def compute_metrics(y_true_hist: List[int], y_pred_hist: List[int], class_names: List[str]) -> Dict[str, float]:
    metrics = {}
    if len(y_true_hist) == 0:
        return metrics

    yt = np.array(y_true_hist, dtype=int)
    yp = np.array(y_pred_hist, dtype=int)

    metrics["accuracy"] = float(accuracy_score(yt, yp))
    metrics["macro_f1"] = float(f1_score(yt, yp, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(yt, yp, average="weighted", zero_division=0))

    normal_idx = None
    for idx, name in enumerate(class_names):
        if str(name).strip().lower() == "normal":
            normal_idx = idx
            break

    if normal_idx is not None:
        yt_bin = (yt != normal_idx).astype(int)
        yp_bin = (yp != normal_idx).astype(int)
        metrics["binary_precision"] = float(precision_score(yt_bin, yp_bin, zero_division=0))
        metrics["binary_recall"] = float(recall_score(yt_bin, yp_bin, zero_division=0))
        metrics["binary_f1"] = float(f1_score(yt_bin, yp_bin, zero_division=0))
    return metrics


def main() -> None:
    st.set_page_config(page_title="NIS Stream Inference", layout="wide")
    st.title("NIS Flow Stream Inference (3s per flow)")
    st.caption("Streams one flow from test data every 3 seconds and updates live metrics.")

    init_state()

    test_df = load_test_data()
    class_names, num_classes = load_label_data()
    all_feature_cols = [c for c in test_df.columns if c != "label"]
    signature_thresholds = compute_signature_thresholds(test_df[all_feature_cols])

    available_modes = ["Main MoE (PyTorch)"]
    boosting_available = os.path.exists(os.path.join(MODELS_DIR, "best_model_meta.pkl"))
    if boosting_available:
        available_modes.append("Boosting-MoE Winner")

    with st.sidebar:
        st.header("Controls")
        mode = st.selectbox("Pipeline", available_modes, index=0)
        max_flows = st.number_input(
            "Flows to stream",
            min_value=1,
            max_value=int(len(test_df)),
            value=min(200, len(test_df)),
            step=1,
        )
        auto_stream = st.checkbox("Auto stream (3 seconds)", value=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reset", use_container_width=True):
                reset_state()
        with c2:
            next_now = st.button("Send Next Now", use_container_width=True)

    # Reset stream state when switching mode to avoid mixed metrics.
    mode_key = "last_mode"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = mode
    elif st.session_state[mode_key] != mode:
        reset_state()
        st.session_state[mode_key] = mode

    if mode == "Main MoE (PyTorch)":
        predictor = load_main_moe_predictor(input_dim=len(all_feature_cols), num_classes=num_classes)
        infer_row = lambda row: predictor.predict(row[all_feature_cols].values)
        st.info("Using `models/moe_model.pt` over `data/splits/test.csv`.")
    else:
        predictor = load_boosting_predictor()
        infer_row = lambda row: predictor.predict(row)
        st.info(f"Using boosting artifacts. Winner model: `{predictor.winner}`.")

    should_step = bool(next_now or auto_stream)
    stream_done = st.session_state.stream_idx >= int(max_flows)

    if should_step and not stream_done:
        row = test_df.iloc[st.session_state.stream_idx : st.session_state.stream_idx + 1]
        row_series = row.iloc[0]
        true_label = int(row["label"].iloc[0])
        pred_label, probs = infer_row(row)
        sig_hits = evaluate_signatures(row_series, signature_thresholds)

        st.session_state.last_true = true_label
        st.session_state.last_pred = int(pred_label)
        st.session_state.last_probs = probs
        st.session_state.last_signatures = sig_hits
        st.session_state.y_true_hist.append(true_label)
        st.session_state.y_pred_hist.append(int(pred_label))
        if sig_hits:
            st.session_state.mitigated_flows += 1
            st.session_state.signature_events.extend([h["signature"] for h in sig_hits])
        st.session_state.stream_idx += 1

    left, mid, right = st.columns(3)
    with left:
        st.metric("Processed flows", st.session_state.stream_idx, int(max_flows))
    with mid:
        last_true_name = class_names[st.session_state.last_true] if st.session_state.last_true is not None else "-"
        st.metric("Last true class", last_true_name)
    with right:
        last_pred_name = class_names[st.session_state.last_pred] if st.session_state.last_pred is not None else "-"
        st.metric("Last predicted class", last_pred_name)

    metrics = compute_metrics(st.session_state.y_true_hist, st.session_state.y_pred_hist, class_names)
    m1, m2, m3 = st.columns(3)
    m1.metric("Running Accuracy", f"{metrics.get('accuracy', 0.0):.4f}")
    m2.metric("Running Macro F1", f"{metrics.get('macro_f1', 0.0):.4f}")
    m3.metric("Running Weighted F1", f"{metrics.get('weighted_f1', 0.0):.4f}")

    b1, b2, b3 = st.columns(3)
    b1.metric("Binary Precision", f"{metrics.get('binary_precision', 0.0):.4f}")
    b2.metric("Binary Recall", f"{metrics.get('binary_recall', 0.0):.4f}")
    b3.metric("Binary F1", f"{metrics.get('binary_f1', 0.0):.4f}")

    s1, s2, s3 = st.columns(3)
    s1.metric("Mitigated Flows", st.session_state.mitigated_flows)
    s2.metric("Signature Hits", len(st.session_state.signature_events))
    unique_sigs = len(set(st.session_state.signature_events))
    s3.metric("Unique Signatures", unique_sigs)

    st.subheader("Running Confusion Matrix")
    if len(st.session_state.y_true_hist) > 0:
        cm = confusion_matrix(
            np.array(st.session_state.y_true_hist, dtype=int),
            np.array(st.session_state.y_pred_hist, dtype=int),
            labels=list(range(num_classes)),
        )
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        st.dataframe(cm_df, use_container_width=True)
    else:
        st.write("No flows processed yet.")

    st.subheader("Last Flow Probabilities")
    if st.session_state.last_probs is not None:
        prob_df = pd.DataFrame({"class": class_names, "probability": st.session_state.last_probs})
        prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
        st.dataframe(prob_df, use_container_width=True)
    else:
        st.write("No prediction yet.")

    st.subheader("Signature-Based Mitigation (Last Flow)")
    if st.session_state.last_signatures:
        for hit in st.session_state.last_signatures:
            st.error(
                f"{hit['signature']} | Severity: {hit['severity']}\n\n"
                f"Reason: {hit['reason']}\n\n"
                f"Mitigation: {hit['mitigation']}"
            )
    else:
        st.success("No signature triggered for the last processed flow.")

    if auto_stream and not stream_done:
        st.caption("Waiting 3 seconds for next flow...")
        time.sleep(3)
        st.rerun()
    elif stream_done:
        st.success("Streaming complete for selected flow count.")


if __name__ == "__main__":
    main()
