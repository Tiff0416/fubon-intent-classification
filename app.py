import json
import os
import re
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
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

# Hugging Face model repo
MODEL_PATH = "CarolineKuo/fubon-intent-classifier"

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
    # banking / finance
    "account_blocked": "帳戶遭封鎖",
    "application_status": "申請進度查詢",
    "apr": "年利率查詢",
    "balance": "帳戶餘額查詢",
    "bill_balance": "帳單餘額查詢",
    "bill_due": "帳單到期日查詢",
    "card_declined": "刷卡失敗處理",
    "credit_limit": "信用額度查詢",
    "credit_limit_change": "信用額度調整",
    "credit_score": "信用分數查詢",
    "direct_deposit": "直接存款設定",
    "exchange_rate": "匯率查詢",
    "expiration_date": "卡片到期日查詢",
    "freeze_account": "凍結帳戶",
    "improve_credit_score": "提升信用分數建議",
    "income": "收入相關問題",
    "insurance": "保險相關問題",
    "insurance_change": "保險方案變更",
    "interest_rate": "利率查詢",
    "international_fees": "國際交易手續費查詢",
    "international_visa": "國際簽證資訊",
    "min_payment": "最低應繳金額查詢",
    "new_card": "新卡申請",
    "order_checks": "支票簿申請",
    "pay_bill": "帳單繳費",
    "payday": "發薪日查詢",
    "pin_change": "PIN 碼變更",
    "redeem_rewards": "紅利兌換",
    "replacement_card_duration": "補發卡片時間查詢",
    "report_fraud": "卡片盜刷或交易異常",
    "report_lost_card": "掛失卡片",
    "rewards_balance": "紅利餘額查詢",
    "rollover_401k": "401k 轉移",
    "routing": "銀行 routing number 查詢",
    "spending_history": "消費紀錄查詢",
    "taxes": "稅務相關問題",
    "transactions": "交易紀錄查詢",
    "transfer": "轉帳服務",
    "w2": "W2 文件查詢",

    # travel / transport
    "book_flight": "航班預訂",
    "book_hotel": "飯店預訂",
    "car_rental": "租車服務",
    "carry_on": "隨身行李規範",
    "current_location": "目前位置查詢",
    "directions": "路線導航",
    "distance": "距離查詢",
    "flight_status": "航班狀態查詢",
    "gas": "加油站查詢",
    "gas_type": "油種查詢",
    "jump_start": "電瓶啟動協助",
    "last_maintenance": "上次保養紀錄",
    "lost_luggage": "行李遺失處理",
    "mpg": "油耗查詢",
    "oil_change_how": "如何換機油",
    "oil_change_when": "何時換機油",
    "plug_type": "插頭規格查詢",
    "schedule_maintenance": "安排保養",
    "share_location": "分享位置",
    "tire_change": "換胎協助",
    "tire_pressure": "胎壓查詢",
    "traffic": "交通路況查詢",
    "travel_alert": "旅遊警示",
    "travel_notification": "旅遊通知設定",
    "travel_suggestion": "旅遊建議",
    "uber": "叫車服務",

    # calendar / productivity
    "alarm": "鬧鐘設定",
    "calendar": "行事曆查詢",
    "calendar_update": "行事曆更新",
    "date": "日期查詢",
    "find_phone": "尋找手機",
    "make_call": "撥打電話",
    "meeting_schedule": "會議時間查詢",
    "next_holiday": "下一個假日查詢",
    "pto_balance": "休假餘額查詢",
    "pto_request": "請假申請",
    "pto_request_status": "請假申請狀態",
    "pto_used": "已使用休假查詢",
    "reminder": "提醒設定",
    "reminder_update": "提醒更新",
    "schedule_meeting": "安排會議",
    "sync_device": "裝置同步",
    "text": "發送簡訊",
    "time": "時間查詢",
    "timer": "計時器設定",
    "timezone": "時區查詢",
    "todo_list": "待辦清單新增",
    "todo_list_update": "待辦清單更新",

    # shopping / order / reservation / restaurant
    "accept_reservations": "是否接受訂位查詢",
    "cancel": "取消操作",
    "cancel_reservation": "取消預約",
    "confirm_reservation": "確認預約",
    "how_busy": "人潮繁忙程度查詢",
    "meal_suggestion": "餐點建議",
    "nutrition_info": "營養資訊查詢",
    "order": "下單服務",
    "order_status": "訂單狀態查詢",
    "restaurant_reservation": "餐廳訂位",
    "restaurant_reviews": "餐廳評論查詢",
    "restaurant_suggestion": "餐廳推薦",
    "shopping_list": "購物清單新增",
    "shopping_list_update": "購物清單更新",

    # cooking / food
    "calories": "熱量查詢",
    "cook_time": "烹飪時間查詢",
    "food_last": "食物保存期限查詢",
    "ingredient_substitution": "食材替代建議",
    "ingredients_list": "食材清單查詢",
    "recipe": "食譜查詢",

    # device / assistant settings
    "are_you_a_bot": "是否為機器人",
    "change_accent": "語音口音設定",
    "change_ai_name": "AI 名稱變更",
    "change_language": "語言設定變更",
    "change_speed": "語速調整",
    "change_user_name": "使用者名稱變更",
    "change_volume": "音量調整",
    "do_you_have_pets": "是否有寵物",
    "goodbye": "結束對話",
    "greeting": "打招呼",
    "how_old_are_you": "年齡詢問",
    "meaning_of_life": "人生意義詢問",
    "maybe": "不確定回應",
    "next_song": "下一首歌曲",
    "no": "否定回應",
    "play_music": "播放音樂",
    "repeat": "重複內容",
    "reset_settings": "重設設定",
    "smart_home": "智慧家庭控制",
    "spelling": "拼字查詢",
    "tell_joke": "講笑話",
    "thank_you": "感謝回應",
    "translate": "翻譯服務",
    "update_playlist": "更新播放清單",
    "user_name": "使用者名稱查詢",
    "what_are_your_hobbies": "興趣詢問",
    "what_can_i_ask_you": "功能詢問",
    "what_is_your_name": "名稱詢問",
    "what_song": "目前歌曲查詢",
    "where_are_you_from": "來歷詢問",
    "whisper_mode": "低語模式設定",
    "who_do_you_work_for": "服務對象詢問",
    "who_made_you": "開發者詢問",
    "yes": "肯定回應",

    # utility / knowledge
    "calculator": "計算機",
    "definition": "詞義查詢",
    "flip_coin": "擲硬幣",
    "fun_fact": "冷知識",
    "measurement_conversion": "單位換算",
    "news": "新聞查詢",
    "weather": "天氣查詢",
    "vaccines": "疫苗資訊查詢",
}

ROUTING_MAP: Dict[str, str] = {
    # banking / finance
    "account_blocked": "帳戶安全與解鎖服務",
    "application_status": "申請進度服務",
    "apr": "信用卡與貸款利率服務",
    "balance": "存款帳務服務",
    "bill_balance": "信用卡帳務服務",
    "bill_due": "信用卡帳務服務",
    "card_declined": "卡片交易異常服務",
    "credit_limit": "信用卡額度服務",
    "credit_limit_change": "信用卡額度服務",
    "credit_score": "信用評分服務",
    "direct_deposit": "入帳與薪資設定服務",
    "exchange_rate": "外匯服務",
    "expiration_date": "卡片資訊服務",
    "freeze_account": "帳戶安全服務",
    "improve_credit_score": "信用評分服務",
    "income": "帳戶與財務資料服務",
    "insurance": "保險服務",
    "insurance_change": "保險服務",
    "interest_rate": "信用卡與貸款利率服務",
    "international_fees": "國際交易服務",
    "international_visa": "旅遊與簽證資訊服務",
    "min_payment": "信用卡帳務服務",
    "new_card": "開卡與新卡服務",
    "order_checks": "支票與帳戶服務",
    "pay_bill": "信用卡帳務服務",
    "payday": "入帳與薪資設定服務",
    "pin_change": "卡片與安全設定服務",
    "redeem_rewards": "紅利與回饋服務",
    "replacement_card_duration": "卡片補發服務",
    "report_fraud": "風險控管與卡片安全服務",
    "report_lost_card": "卡片掛失與補發服務",
    "rewards_balance": "紅利與回饋服務",
    "rollover_401k": "退休金與投資服務",
    "routing": "帳戶資訊服務",
    "spending_history": "交易與消費紀錄服務",
    "taxes": "稅務文件服務",
    "transactions": "交易紀錄服務",
    "transfer": "轉帳服務",
    "w2": "稅務文件服務",

    # travel / transport
    "book_flight": "旅遊預訂服務",
    "book_hotel": "旅遊預訂服務",
    "car_rental": "旅遊預訂服務",
    "carry_on": "旅遊規範服務",
    "current_location": "一般資訊服務",
    "directions": "導航服務",
    "distance": "導航服務",
    "flight_status": "旅遊資訊服務",
    "gas": "交通資訊服務",
    "gas_type": "交通資訊服務",
    "jump_start": "道路救援服務",
    "last_maintenance": "車輛保養服務",
    "lost_luggage": "旅遊支援服務",
    "mpg": "車輛資訊服務",
    "oil_change_how": "車輛保養服務",
    "oil_change_when": "車輛保養服務",
    "plug_type": "旅遊資訊服務",
    "schedule_maintenance": "車輛保養服務",
    "share_location": "導航服務",
    "tire_change": "道路救援服務",
    "tire_pressure": "車輛資訊服務",
    "traffic": "導航服務",
    "travel_alert": "旅遊資訊服務",
    "travel_notification": "旅遊資訊服務",
    "travel_suggestion": "旅遊建議服務",
    "uber": "交通服務",

    # calendar / productivity
    "alarm": "提醒與鬧鐘服務",
    "calendar": "行程管理服務",
    "calendar_update": "行程管理服務",
    "date": "一般資訊服務",
    "find_phone": "裝置協助服務",
    "make_call": "通訊服務",
    "meeting_schedule": "行程管理服務",
    "next_holiday": "一般資訊服務",
    "pto_balance": "人事與休假服務",
    "pto_request": "人事與休假服務",
    "pto_request_status": "人事與休假服務",
    "pto_used": "人事與休假服務",
    "reminder": "提醒與鬧鐘服務",
    "reminder_update": "提醒與鬧鐘服務",
    "schedule_meeting": "行程管理服務",
    "sync_device": "裝置協助服務",
    "text": "通訊服務",
    "time": "一般資訊服務",
    "timer": "提醒與鬧鐘服務",
    "timezone": "一般資訊服務",
    "todo_list": "待辦事項服務",
    "todo_list_update": "待辦事項服務",

    # shopping / order / restaurant
    "accept_reservations": "餐飲預訂服務",
    "cancel": "一般客服中心",
    "cancel_reservation": "預約管理服務",
    "confirm_reservation": "預約管理服務",
    "how_busy": "餐飲資訊服務",
    "meal_suggestion": "餐飲建議服務",
    "nutrition_info": "食品與營養資訊服務",
    "order": "訂單服務",
    "order_status": "訂單服務",
    "restaurant_reservation": "餐飲預訂服務",
    "restaurant_reviews": "餐飲資訊服務",
    "restaurant_suggestion": "餐飲建議服務",
    "shopping_list": "購物清單服務",
    "shopping_list_update": "購物清單服務",

    # cooking / food
    "calories": "食品與營養資訊服務",
    "cook_time": "料理資訊服務",
    "food_last": "食品保存資訊服務",
    "ingredient_substitution": "料理資訊服務",
    "ingredients_list": "料理資訊服務",
    "recipe": "料理資訊服務",

    # device / assistant settings
    "are_you_a_bot": "一般資訊服務",
    "change_accent": "裝置設定服務",
    "change_ai_name": "裝置設定服務",
    "change_language": "裝置設定服務",
    "change_speed": "裝置設定服務",
    "change_user_name": "帳戶與設定服務",
    "change_volume": "裝置設定服務",
    "do_you_have_pets": "一般資訊服務",
    "goodbye": "一般客服中心",
    "greeting": "一般客服中心",
    "how_old_are_you": "一般資訊服務",
    "meaning_of_life": "一般資訊服務",
    "maybe": "一般客服中心",
    "next_song": "音樂控制服務",
    "no": "一般客服中心",
    "play_music": "音樂控制服務",
    "repeat": "一般客服中心",
    "reset_settings": "裝置設定服務",
    "smart_home": "智慧家庭服務",
    "spelling": "一般資訊服務",
    "tell_joke": "一般資訊服務",
    "thank_you": "一般客服中心",
    "translate": "翻譯服務",
    "update_playlist": "音樂控制服務",
    "user_name": "帳戶與設定服務",
    "what_are_your_hobbies": "一般資訊服務",
    "what_can_i_ask_you": "一般資訊服務",
    "what_is_your_name": "一般資訊服務",
    "what_song": "音樂控制服務",
    "where_are_you_from": "一般資訊服務",
    "whisper_mode": "裝置設定服務",
    "who_do_you_work_for": "一般資訊服務",
    "who_made_you": "一般資訊服務",
    "yes": "一般客服中心",

    # utility / knowledge
    "calculator": "一般資訊服務",
    "definition": "一般資訊服務",
    "flip_coin": "一般資訊服務",
    "fun_fact": "一般資訊服務",
    "measurement_conversion": "一般資訊服務",
    "news": "一般資訊服務",
    "weather": "一般資訊服務",
    "vaccines": "健康資訊服務",
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
    label_path = hf_hub_download(
        repo_id=MODEL_PATH,
        filename="id2label.json"
    )
    with open(label_path, "r", encoding="utf-8") as f:
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
    if label in DISPLAY_LABELS:
        return DISPLAY_LABELS[label]
    return label.replace("_", " ").title()


def get_routing_team(label: str) -> str:
    if label in ROUTING_MAP:
        return ROUTING_MAP[label]
    return "一般客服中心"


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
        st.write(f"模型來源：{MODEL_PATH}")

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