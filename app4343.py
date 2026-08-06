import streamlit as st
from ib_insync import *
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 🤖 استدعاء محرك الذكاء الاصطناعي المحلي
from ai_models import LocalAITradingEngine

# تهيئة المحرك
ai_engine = LocalAITradingEngine()

st.set_page_config(page_title="Local AI Trading Bot (IBKR)", layout="wide")
st.title("🤖 بوت التداول بمحرك الذكاء الاصطناعي المحلي (IBKR)")

# --- الشريط الجانبي ---
st.sidebar.header("⚙️ إعدادات الاتصال")
ib_host = st.sidebar.text_input("IB Host", "127.0.0.1")
ib_port = st.sidebar.number_input("IB Port (Demo: 7497/4002)", value=7497)
symbol = st.sidebar.text_input("رمز السهم", "AAPL")
quantity = st.sidebar.number_input("الكمية المحددة", value=10, step=1)

# --- دالة جلب البيانات من IBKR ---
def get_market_data(symbol):
    ib = IB()
    try:
        ib.connect(ib_host, int(ib_port), clientId=10)
        contract = Stock(symbol, 'SMART', 'USD')
        bars = ib.reqHistoricalData(
            contract, endDateTime='', durationStr='5 D',
            barSizeSetting='5 mins', whatToShow='TRADES', useRTH=True
        )
        df = util.df(bars)
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        ib.disconnect()
        return df, None
    except Exception as e:
        return None, str(e)

# --- دالة تنفيذ الأوامر ---
def execute_order(action, symbol, qty):
    ib = IB()
    try:
        ib.connect(ib_host, int(ib_port), clientId=11)
        contract = Stock(symbol, 'SMART', 'USD')
        order = MarketOrder(action, qty)
        trade = ib.placeOrder(contract, order)
        ib.sleep(2)
        ib.disconnect()
        return f"✅ تم تنفيذ أمر {action} لـ {qty} سهم في {symbol} بنجاح!"
    except Exception as e:
        return f"❌ خطأ أثناء تنفيذ الأمر: {e}"

# --- الواجهة الرئيسية ---
col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.subheader("📊 فحص وتحديث البيانات")
    if st.button("جلب البيانات من IBKR"):
        with st.spinner("جاري جلب البيانات..."):
            df, error = get_market_data(symbol)
            if error:
                st.error(f"خطأ في الاتصال: {error}")
            else:
                st.session_state['df'] = df
                st.success("تم التحديث!")

    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # رسم الشارت التفاعلي
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_20'], mode='lines', name='SMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_50'], mode='lines', name='SMA 50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
        fig.update_layout(height=500, template='plotly_dark', xaxis_rangeslider_visible=False)
        
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🤖 فحص النموذج المحلي (Local Machine Learning)")
    
    if st.button("تحليل الفرصة بالذكاء الاصطناعي المحلي"):
        if 'df' not in st.session_state:
            st.warning("يرجى جلب البيانات أولاً.")
        else:
            df = st.session_state['df']
            action, conf, reason = ai_engine.predict_opportunity(df)
            st.session_state['ai_decision'] = (action, conf, reason)

    if 'ai_decision' in st.session_state:
        action, conf, reason = st.session_state['ai_decision']
        
        st.write("---")
        st.markdown(f"**القرار:** `{action}` | **درجة الثقة:** `{conf}%`")
        st.info(f"💡 **التحليل:** {reason}")
        
        if action == 'BUY':
            if st.button(f"🚀 تنفيذ أمر شراء ({quantity} أسهم)", type="primary"):
                res = execute_order("BUY", symbol, quantity)
                st.success(res)
        elif action == 'SELL':
            if st.button(f"🔻 تنفيذ أمر بيع ({quantity} أسهم)", type="primary"):
                res = execute_order("SELL", symbol, quantity)
                st.success(res)
        else:
            st.warning("النموذج يوصي بالانتظار (HOLD) لعدم تبلور فرصة واضحة.")
