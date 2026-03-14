import json
import os
import re
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===== Optional: OpenAI for advanced review =====
try:
    from openai import OpenAI
    USE_LLM = True
except ImportError:
    OpenAI = None
    USE_LLM = False


# =========================
# 基本設定
# =========================
st.set_page_config(
    page_title="銀行客服問題分析助手",
    page_icon="🏦",
    layout="centered",
)

MODEL_PATH = "models/intent_classifier"
ID2LABEL_PATH = os.path.join(MODEL_PATH, "id2label.json")

CONFIDENCE_THRESHOLD = 0.35
MARGIN_THRESHOLD = 0.10
TOP_K = 3
LLM_MODEL = "gpt-4o-mini"


# =========================
# Session State
# =========================
if "history" not in st.session_state:
    st.session_state.history = []


# =========================
# 顯示名稱 / 分流規則
# =========================
DISPLAY_LABELS: Dict[str, str] = {
    "check_balance": "帳戶餘額查詢",
    "pay_bill": "帳單繳費",
    "report_fraud": "卡片盜刷或交易異常",
    "book_flight": "旅遊訂票需求",
    "flight_status": "航班狀態查詢",
    "carry_on": "行李規範諮詢",
}

ROUTING_MAP: Dict[str, str] = {
    "check_balance": "存款帳務服務",
    "pay_bill": "信用卡帳務服務",
    "report_fraud": "風險控管與卡片安全服務",
    "book_flight": "旅遊服務專員",
    "flight_status": "旅遊資訊客服",
    "carry_on": "旅遊規範客服",
}


# =========================
# 載入模型與標籤
# =========================
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


@st.cache_data
def load_id2label():
    with open(ID2LABEL_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


tokenizer, model = load_model_and_tokenizer()
id2label = load_id2label()


# =========================
# OpenAI Client
# =========================
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


# =========================
# 工具函數
# =========================
def prettify_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label.replace("_", " ").title())


def get_routing_team(label: str) -> str:
    return ROUTING_MAP.get(label, "一般客服中心")


def clarity_label(score: float, margin: float) -> str:
    if score >= 0.85:
        return "很清楚"
    if score >= CONFIDENCE_THRESHOLD:
        return "大致清楚"
    if margin >= 0.20:
        return "大致清楚（前兩名差距大）"
    if margin >= MARGIN_THRESHOLD:
        return "普通"
    return "不太清楚"


def get_processing_advice(score: float, margin: float, n_subqueries: int) -> str:
    if n_subqueries > 1:
        return "建議把這個問題拆成多個小問題後，分別交給對應客服處理。"
    if score >= 0.85:
        return "結果相對明確，可直接轉交對應客服類別。"
    if score >= CONFIDENCE_THRESHOLD:
        return "結果大致明確，建議轉交對應客服類別，必要時由人工再次確認。"
    if margin >= MARGIN_THRESHOLD:
        return "雖然整體分數不高，但第一候選明顯領先，可優先依此結果處理。"
    return "這個結果不夠明確，建議啟用第二次判斷或轉交人工客服確認。"


def predict_intent(query: str):
    """用 BERT 對單一句子做 intent 預測"""
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    confidence = float(np.max(probs))

    top_indices = np.argsort(probs)[-TOP_K:][::-1]
    top_results = [
        {
            "intent_id": int(i),
            "intent": id2label[int(i)],
            "display_intent": prettify_label(id2label[int(i)]),
            "score": float(probs[i]),
        }
        for i in top_indices
    ]

    top1_score = float(probs[top_indices[0]])
    top2_score = float(probs[top_indices[1]]) if len(top_indices) > 1 else 0.0
    margin = top1_score - top2_score

    return {
        "pred_id": pred_id,
        "pred_intent": id2label[pred_id],
        "pred_display_intent": prettify_label(id2label[pred_id]),
        "confidence": confidence,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "margin": margin,
        "top_results": top_results,
    }


def split_multi_intent_rule_based(query: str) -> List[str]:
    """
    規則版拆句：
    適合快速 demo，但不會像第二次判斷那樣靈活。
    """
    parts = re.split(
        r"\band\b|\balso\b|\bthen\b|\bplus\b|,|;|/|\&",
        query,
        flags=re.IGNORECASE,
    )
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [query]


def llm_split_query(query: str) -> List[str]:
    """
    用第二次判斷模組拆解複合問題。
    若失敗則回退到 rule-based。
    """
    if not USE_LLM:
        return split_multi_intent_rule_based(query)

    client = get_openai_client()
    if client is None:
        return split_multi_intent_rule_based(query)

    prompt = f"""
你是一個客服問題拆解助理。
請將以下使用者問題拆成一個或多個獨立子問題。

規則：
1. 若只有一個需求，直接回傳一行。
2. 若包含多個需求，請逐行列出，每行一個子問題。
3. 不要解釋，不要加編號。

使用者問題：
{query}
""".strip()

    try:
        response = client.responses.create(
            model=LLM_MODEL,
            input=prompt,
        )
        text = response.output_text.strip()
        lines = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
        return lines if lines else split_multi_intent_rule_based(query)
    except Exception:
        return split_multi_intent_rule_based(query)


def llm_rerank_intent(query: str, candidates: List[str]) -> str:
    """
    當模型真的不明確時，讓第二次判斷模組從候選類別中選出較合適的一個。
    """
    if not USE_LLM:
        return ""

    client = get_openai_client()
    if client is None:
        return ""

    candidate_text = "\n".join([f"- {c}" for c in candidates])

    prompt = f"""
你是一個客服問題分類助理。
請根據使用者問題，從以下候選問題類型中選出最適合的一個。

使用者問題：
{query}

候選問題類型：
{candidate_text}

請只回傳一個問題類型名稱，不要加任何解釋。
""".strip()

    try:
        response = client.responses.create(
            model=LLM_MODEL,
            input=prompt,
        )
        return response.output_text.strip()
    except Exception:
        return ""


# =========================
# UI
# =========================
st.title("🏦 銀行客服問題分析助手")
st.caption("系統會自動判斷客戶想處理的問題；只有在模型真的不明確時，才進行第二次判斷。")

client = get_openai_client()

with st.sidebar:
    st.markdown("### 系統狀態")
    st.write("基本判斷功能：正常")
    st.write(f"第二次判斷功能：{'可用' if client is not None else '不可用'}")

    with st.expander("技術設定", expanded=False):
        st.write("主分類模型：BERT")
        st.write(f"API 連線：{'已連線' if client is not None else '未連線'}")
        st.write(f"進階模型：{LLM_MODEL}")
        st.write(f"Confidence 門檻：{CONFIDENCE_THRESHOLD}")
        st.write(f"Margin 門檻：{MARGIN_THRESHOLD}")

with st.expander("系統說明", expanded=False):
    st.write(
        """
        本系統會分析客戶提出的問題，判斷最可能的服務需求，
        並建議應該轉交給哪個客服類別處理。

        系統不只看第一名分數，也會看第一名和第二名的差距。
        如果第一名明顯領先，即使整體分數不高，也可能代表模型其實很確定。

        只有在結果真的不明確時，才會進行第二次分析。
        """
    )

st.markdown("### 範例案件")
col1, col2 = st.columns(2)
with col1:
    if st.button("帳戶餘額查詢"):
        st.session_state["demo_query"] = "I want to check my balance"
    if st.button("卡片盜刷異常"):
        st.session_state["demo_query"] = "Someone used my card and I want to report fraud"
with col2:
    if st.button("複合需求示範"):
        st.session_state["demo_query"] = "I want to check my balance and pay my bill"
    if st.button("旅遊服務示範"):
        st.session_state["demo_query"] = "I want to book a flight and reserve a hotel"

default_query = st.session_state.get("demo_query", "")
query = st.text_area("請輸入客戶問題：", value=default_query, height=120)

st.markdown("### 處理選項")
use_llm_split = st.checkbox("把一句話拆成多個問題來分析", value=False)
use_llm_fallback = st.checkbox("當結果真的不明確時，啟用第二次判斷", value=True)

if st.button("開始分析"):
    if not query.strip():
        st.warning("請先輸入問題。")
        st.stop()

    # ===== 拆解需求 =====
    if use_llm_split:
        sub_queries = llm_split_query(query)
    else:
        sub_queries = split_multi_intent_rule_based(query)

    st.markdown("## 1) 問題拆解結果")
    for i, sq in enumerate(sub_queries, start=1):
        st.write(f"{i}. {sq}")

    st.markdown("## 2) 分析結果與建議")

    for idx, sub_q in enumerate(sub_queries, start=1):
        result = predict_intent(sub_q)
        pred_intent = result["pred_intent"]
        pred_display_intent = result["pred_display_intent"]
        confidence = result["confidence"]
        top1_score = result["top1_score"]
        top2_score = result["top2_score"]
        margin = result["margin"]
        top_results = result["top_results"]

        need_second_review = (
            confidence < CONFIDENCE_THRESHOLD and margin < MARGIN_THRESHOLD
        )

        final_label = pred_intent
        final_display_label = pred_display_intent
        review_used = False

        with st.container(border=True):
            st.markdown(f"### 問題 {idx}")
            st.write(f"**客戶問題：** {sub_q}")
            st.write(f"**系統判斷這是什麼問題：** `{pred_display_intent}`")
            st.write(f"**判斷信心：** {confidence:.3f}")
            st.write(f"**第一候選分數：** {top1_score:.3f}")
            st.write(f"**第二候選分數：** {top2_score:.3f}")
            st.write(f"**前兩名差距：** {margin:.3f}")
            st.write(f"**結果清楚程度：** {clarity_label(confidence, margin)}")
            st.write(f"**建議客服類別：** {get_routing_team(pred_intent)}")

            top_df = pd.DataFrame({
                "候選問題類型": [x["display_intent"] for x in top_results],
                "分數": [round(x["score"], 3) for x in top_results],
            })
            st.write("**其他可能結果：**")
            st.table(top_df)

            if need_second_review:
                st.warning("模型對這個結果仍不夠確定，建議啟用第二次判斷或轉交人工客服確認。")

                if use_llm_fallback:
                    candidates = [x["intent"] for x in top_results]
                    reviewed_label = llm_rerank_intent(sub_q, candidates)

                    if reviewed_label:
                        final_label = reviewed_label.strip()
                        final_display_label = prettify_label(final_label)
                        review_used = True

                        st.write(f"**第二次判斷結果：** `{final_display_label}`")
                        st.write(f"**更新後建議客服類別：** {get_routing_team(final_label)}")
                    else:
                        st.write("**第二次判斷結果：** 目前無法取得，建議改由人工客服確認。")
            else:
                if confidence < CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
                    st.success("雖然整體分數不高，但第一候選明顯領先，系統判定此結果可接受。")
                else:
                    st.success("這個結果已經相對明確，可直接進一步處理。")

            st.markdown("#### 建議怎麼處理")
            st.write(get_processing_advice(confidence, margin, len(sub_queries)))

            if review_used:
                st.info("因為模型對第一次結果不夠確定，系統已自動進行第二次分析。")

            # ===== 記錄歷史資料（供 Dashboard 使用）=====
            st.session_state.history.append({
                "query": sub_q,
                "intent": final_display_label,
                "confidence": confidence,
                "top1_score": top1_score,
                "top2_score": top2_score,
                "margin": margin,
                "used_second_review": review_used,
                "routing_team": get_routing_team(final_label),
            })

    st.markdown("## 3) 整體建議")
    st.write(
        """
        - **結果很清楚的問題**：可直接轉交對應客服類別  
        - **分數不高但前兩名差距很大的問題**：可優先採用第一候選  
        - **分數低且前兩名很接近的問題**：建議再做一次判斷或由人工客服確認  
        - **同一句話包含多個需求**：建議拆成多個問題後分別處理
        """
    )

# =========================
# Dashboard
# =========================
st.markdown("---")
st.header("📊 案件統計 Dashboard")

history = st.session_state.history

if len(history) == 0:
    st.info("目前還沒有案件資料。請先進行幾次分析。")
else:
    df = pd.DataFrame(history)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("已分析問題數", len(df))

    with col2:
        avg_conf = df["confidence"].mean()
        st.metric("平均判斷信心", f"{avg_conf:.2f}")

    with col3:
        ambiguous_cases = (
            (df["confidence"] < CONFIDENCE_THRESHOLD) &
            (df["margin"] < MARGIN_THRESHOLD)
        ).sum()
        st.metric("真正不明確案例", int(ambiguous_cases))

    with col4:
        second_review_cases = int(df["used_second_review"].sum())
        st.metric("已啟用第二次判斷", second_review_cases)

    st.subheader("問題類型分布")
    intent_counts = df["intent"].value_counts()
    st.bar_chart(intent_counts)

    st.subheader("客服問題熱門排行榜（Top 5）")
    top5_issues = df["intent"].value_counts().reset_index()
    top5_issues.columns = ["問題類型", "次數"]
    top5_issues = top5_issues.head(5)
    top5_issues.insert(0, "排名", range(1, len(top5_issues) + 1))
    st.table(top5_issues)

    st.subheader("建議客服類別分布")
    routing_counts = df["routing_team"].value_counts()
    st.bar_chart(routing_counts)

    with st.expander("查看歷史分析紀錄", expanded=False):
        history_view = df.copy()
        history_view.columns = [
            "客戶問題", "問題類型", "判斷信心", "第一候選分數",
            "第二候選分數", "前兩名差距", "是否啟用第二次判斷", "建議客服類別"
        ]
        st.dataframe(history_view, use_container_width=True)