# ==========================================
# حل مشكلة event loop - أبسط حل
# ==========================================
import asyncio
import nest_asyncio

# تطبيق nest_asyncio
nest_asyncio.apply()

# ==========================================
# استيراد المكتبات
# ==========================================
import streamlit as st
from ib_insync import *
import pandas as pd
import ta
from openai import OpenAI
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')

# 🤖 محرك الذكاء الاصطناعي المحلي
from ai_models import LocalAITradingEngine

# ==========================================
# إعدادات Streamlit
# ==========================================
st.set_page_config(
    page_title="AI Trading Bot (IBKR)",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# الثوابت
# ==========================================
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7497
DEFAULT_SYMBOL = "AAPL"
DEFAULT_QUANTITY = 10
DEFAULT_INTERVAL = 300

# ==========================================
# دوال IBKR
# ==========================================

def get_market_data(symbol, host, port):
    """جلب البيانات من IBKR"""
    ib = IB()
    try:
        ib.connect(host, int(port), clientId=99)
        contract = Stock(symbol, 'SMART', 'USD')
        
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True
        )
        
        df = util.df(bars)
        
        if df.empty:
            return None, "لا توجد بيانات"
        
        # المؤشرات الفنية
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=10)
        df['date'] = df.index
        
        ib.disconnect()
        return df, None
        
    except Exception as e:
        try:
            ib.disconnect()
        except:
            pass
        return None, str(e)

def execute_ib_order(action, symbol, qty, host, port):
    """تنفيذ أمر تداول"""
    ib = IB()
    try:
        ib.connect(host, int(port), clientId=100)
        contract = Stock(symbol, 'SMART', 'USD')
        order = MarketOrder(action, qty)
        trade = ib.placeOrder(contract, order)
        ib.sleep(2)
        
        status = trade.orderStatus.status
        ib.disconnect()
        
        if status in ['Filled', 'Submitted']:
            return f"✅ تم تنفيذ أمر {action} بنجاح!"
        else:
            return f"⚠️ الحالة: {status}"
            
    except Exception as e:
        try:
            ib.disconnect()
        except:
            pass
        return f"❌ خطأ: {e}"

# ==========================================
# دوال التحليل
# ==========================================

def analyze_with_local_ai(df):
    """تحليل باستخدام المحرك المحلي"""
    engine = st.session_state['ai_engine']
    
    if not engine.is_trained:
        with st.spinner("تدريب النموذج..."):
            engine.train_quick_model(df)
    
    action, confidence, reason = engine.predict_opportunity(df)
    
    result = f"[RECOMMENDATION: {action}]\n"
    result += f"الثقة: {confidence}%\n"
    result += f"السبب: {reason}\n"
    
    return result, action, confidence

def analyze_with_openai(df_summary, api_key, symbol_name):
    """تحليل باستخدام OpenAI"""
    if not api_key:
        return "⚠️ مطلوب مفتاح OpenAI", "HOLD", 0
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    حلل البيانات الفنية لسهم {symbol_name}:
    {df_summary}
    
    أجب بتنسيق:
    [RECOMMENDATION: BUY] أو [RECOMMENDATION: SELL] أو [RECOMMENDATION: HOLD]
    ثم اشرح السبب.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        result = response.choices[0].message.content
        
        if "[RECOMMENDATION: BUY]" in result:
            action = "BUY"
        elif "[RECOMMENDATION: SELL]" in result:
            action = "SELL"
        else:
            action = "HOLD"
        
        return result, action, 70
        
    except Exception as e:
        return f"❌ خطأ: {e}", "HOLD", 0

def analyze_hybrid(df, api_key, symbol_name):
    """تحليل هجين"""
    local_result, local_action, local_conf = analyze_with_local_ai(df)
    
    if api_key:
        openai_result, openai_action, openai_conf = analyze_with_openai(
            df.tail(10).to_string(), api_key, symbol_name
        )
        
        # دمج النتائج
        if local_action == openai_action and local_action != "HOLD":
            final_action = local_action
            hybrid_result = f"✅ توافق: {local_action}\n{local_result}\n\n{openai_result}"
        else:
            final_action = "HOLD"
            hybrid_result = f"⚠️ تباين - انتظار\n{local_result}\n\n{openai_result}"
    else:
        final_action = local_action
        hybrid_result = local_result
    
    return hybrid_result, final_action, local_conf

# ==========================================
# دوال الرسم البياني
# ==========================================

def plot_chart(df, symbol_name):
    """رسم بياني تفاعلي"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'📈 {symbol_name}', 'RSI', 'Volume')
    )
    
    # الشموع
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # المتوسطات
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='cyan')),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    colors = ['green' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'red' for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], name='Volume', marker_color=colors, opacity=0.6),
        row=3, col=1
    )
    
    fig.update_layout(
        height=600,
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig

# ==========================================
# الدالة الرئيسية
# ==========================================

def main():
    """التطبيق الرئيسي"""
    
    st.title("🤖 بوت التداول بالذكاء الاصطناعي (IBKR)")
    
    # تهيئة المحرك
    if 'ai_engine' not in st.session_state:
        st.session_state['ai_engine'] = LocalAITradingEngine()
    
    # ===== الشريط الجانبي =====
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        api_key = st.text_input("OpenAI API Key (اختياري)", type="password")
        ib_host = st.text_input("IB Host", DEFAULT_HOST)
        ib_port = st.number_input("IB Port", value=DEFAULT_PORT)
        symbol = st.text_input("رمز السهم", DEFAULT_SYMBOL)
        quantity = st.number_input("الكمية", value=DEFAULT_QUANTITY, step=1)
        
        analysis_mode = st.radio(
            "وضع التحليل",
            ["المحرك المحلي", "OpenAI", "هجين"]
        )
        
        st.divider()
        st.subheader("📊 حالة المحرك")
        engine = st.session_state['ai_engine']
        if engine.is_trained:
            st.success("✅ النموذج جاهز")
        else:
            st.warning("⚠️ غير مدرب")
    
    # ===== الأعمدة الرئيسية =====
    col1, col2 = st.columns([1.5, 1])
    
    # العمود الأول - البيانات
    with col1:
        st.subheader("📊 البيانات الفنية")
        
        if st.button("🔄 جلب البيانات", use_container_width=True):
            with st.spinner("جاري الاتصال بـ IBKR..."):
                df, error = get_market_data(symbol, ib_host, ib_port)
                
                if error:
                    st.error(f"❌ {error}")
                    st.info("تأكد من تشغيل TWS/IB Gateway مع تمكين API")
                else:
                    st.session_state['df'] = df
                    
                    with st.spinner("تدريب النموذج..."):
                        if engine.train_quick_model(df):
                            st.success("✅ تم التدريب بنجاح!")
                    
                    st.success(f"✅ {len(df)} شمعة جاهزة")
                    
                    last_price = df['close'].iloc[-1]
                    change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if len(df) > 1 else 0
                    st.metric("السعر الحالي", f"${last_price:.2f}", f"{change:.2f}%")
        
        if 'df' in st.session_state:
            df = st.session_state['df']
            
            # الرسم البياني
            fig = plot_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # الجدول
            with st.expander("📋 البيانات"):
                cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'SMA_20', 'SMA_50']
                st.dataframe(df[cols].tail(10), use_container_width=True)
    
    # العمود الثاني - التحليل والتنفيذ
    with col2:
        st.subheader("🤖 التحليل والتنفيذ")
        
        if st.button("🧠 تحليل", use_container_width=True, type="primary"):
            if 'df' not in st.session_state:
                st.warning("⚠️ جلب البيانات أولاً")
            else:
                df = st.session_state['df']
                
                with st.spinner("جاري التحليل..."):
                    if analysis_mode == "المحرك المحلي":
                        result, action, conf = analyze_with_local_ai(df)
                    elif analysis_mode == "OpenAI":
                        result, action, conf = analyze_with_openai(
                            df.tail(10).to_string(), api_key, symbol
                        )
                    else:  # هجين
                        result, action, conf = analyze_hybrid(df, api_key, symbol)
                    
                    st.session_state['result'] = result
                    st.session_state['action'] = action
                    st.session_state['confidence'] = conf
                    
                    st.success("✅ تم التحليل")
        
        # عرض النتيجة
        if 'result' in st.session_state:
            st.divider()
            
            action = st.session_state['action']
            conf = st.session_state.get('confidence', 0)
            
            if action == "BUY":
                st.success(f"🟢 **شراء** (ثقة: {conf}%)")
            elif action == "SELL":
                st.error(f"🔴 **بيع** (ثقة: {conf}%)")
            else:
                st.warning(f"⏸️ **انتظار** (ثقة: {conf}%)")
            
            st.text_area("التفاصيل:", st.session_state['result'], height=150)
            
            # أزرار التنفيذ
            st.divider()
            st.subheader("💼 التنفيذ")
            
            if action == "BUY":
                if st.button(f"🚀 شراء {quantity} سهم", use_container_width=True, type="primary"):
                    msg = execute_ib_order("BUY", symbol, quantity, ib_host, ib_port)
                    st.success(msg)
                    
            elif action == "SELL":
                if st.button(f"🔻 بيع {quantity} سهم", use_container_width=True, type="primary"):
                    msg = execute_ib_order("SELL", symbol, quantity, ib_host, ib_port)
                    st.success(msg)
            
            if st.button("🗑️ مسح", use_container_width=True):
                for key in ['result', 'action', 'confidence']:
                    st.session_state.pop(key, None)
                st.rerun()

# ==========================================
# التشغيل
# ==========================================

if __name__ == "__main__":
    main()
