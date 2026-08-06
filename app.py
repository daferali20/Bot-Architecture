import streamlit as st
from ib_insync import *
import pandas as pd
import pandas_ta as ta
from openai import OpenAI
import os

# --- 1. إعداد واجهة Streamlit ---
st.set_page_config(page_title="AI Trading Bot (IBKR Demo)", layout="wide")
st.title("🤖 بوت التداول بالذكاء الاصطناعي (Interactive Brokers)")

# الشريط الجانبي للإعدادات
st.sidebar.header("⚙️ إعدادات الاتصال والحساب")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
ib_host = st.sidebar.text_input("IB Host", "127.0.0.1")
ib_port = st.sidebar.number_input("IB Port (Demo: 7497/4002)", value=7497)
symbol = st.sidebar.text_input("رمز السهم (Symbol)", "AAPL")
quantity = st.sidebar.number_input("الكمية المحددة", value=10, step=1)

# --- 2. دوال الاتصال وجلب البيانات ---
def get_market_data(symbol):
    """الاتصال بـ IBKR وجلب البيانات الفنية"""
    ib = IB()
    try:
        ib.connect(ib_host, int(ib_port), clientId=99)
        contract = Stock(symbol, 'SMART', 'USD')
        bars = ib.reqHistoricalData(
            contract, endDateTime='', durationStr='2 D',
            barSizeSetting='5 mins', whatToShow='TRADES', useRTH=True
        )
        df = util.df(bars)
        
        # حساب المؤشرات
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        
        ib.disconnect()
        return df, None
    except Exception as e:
        return None, str(e)

def analyze_with_ai(df_summary, api_key):
    """إرسال البيانات إلى OpenAI للتحليل والقرارات"""
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    أنت خبير تداول وسيگنال بالذكاء الاصطناعي. إليك أحدث البيانات الفنية لسهم {symbol}:
    {df_summary}

    قم بتحليل المؤشرات التالية:
    1. اتجاه السعر بالنسبة للمتوسطات SMA 20 و SMA 50.
    2. حالة مؤشر RSI.

    أعطني إجابتك بتنسيق واضح ويحتوي مجبرًا على إحدى الكلمات التالية في السطر الأول فقط:
    [RECOMMENDATION: BUY] أو [RECOMMENDATION: SELL] أو [RECOMMENDATION: HOLD]
    ثم اشرح السبب باختصار في أسطر لاحقة.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content
    #--------------------------------
def plot_interactive_chart(df, symbol):
    """إنشاء رسم بياني تفاعلي باستخدام Plotly للشموع والمؤشرات الفنية"""
    
    # إنشاء لوحة رسم مقسمة إلى صفين (الصف الأول للأسعار، الصف الثاني لـ RSI)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(f'سعر الشموع لـ {symbol} والمتوسطات المتحركة', 'مؤشر القوة النسبية (RSI)'),
        row_heights=[0.7, 0.3]
    )

    # 1. رسم الشموع اليابانية (Candlestick)
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='الأسعار'
        ),
        row=1, col=1
    )

    # 2. إضافة المتوسط المتحرك 20
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=1.5)
        ),
        row=1, col=1
    )

    # 3. إضافة المتوسط المتحرك 50
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=1.5)
        ),
        row=1, col=1
    )

    # 4. رسم خط مؤشر RSI
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=1.5)
        ),
        row=2, col=1
    )

    # 5. إضافة خطوط مناطق التشبع الشرائي والبيعي (70 و 30)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # تعديل التنسيق والعرض
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,  # لإخفاء شريط التكبير السفلي لتنظيف الشاشة
        template='plotly_dark',           # الثيم الداكن (Dark Mode)
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig
    #----------------------------------------
def execute_ib_order(action, symbol, qty):
    """تنفيذ أمر التداول على الحساب التجريبي"""
    ib = IB()
    try:
        ib.connect(ib_host, int(ib_port), clientId=100)
        contract = Stock(symbol, 'SMART', 'USD')
        order = MarketOrder(action, qty)
        trade = ib.placeOrder(contract, order)
        ib.sleep(2) # انتظار لتأكيد الإرسال
        ib.disconnect()
        return f"تم إرسال أمر {action} لـ {qty} سهم في {symbol} بنجاح!"
    except Exception as e:
        return f"خطأ أثناء تنفيذ الأمر: {e}"

# --- 3. تشغيل الواجهة والتفاعل ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 فحص البيانات الفنية")
    if st.button("جلب البيانات الفنية الآن"):
        with st.spinner("جاري الاتصال بـ IBKR وجلب الشموع..."):
            df, error = get_market_data(symbol)
            if error:
                st.error(f"فشل الاتصال بـ IBKR: {error}")
            else:
                st.session_state['df'] = df
                st.success("تم جلب البيانات بنجاح!")

    if 'df' in st.session_state:
        df = st.session_state['df']
        st.dataframe(df[['date', 'close', 'RSI', 'SMA_20', 'SMA_50']].tail(5))

with col2:
    st.subheader("🤖 تحليل الذكاء الاصطناعي وتنفيذ الأوامر")
    if st.button("تحليل الفرصة بواسطة الذكاء الاصطناعي"):
        if not api_key:
            st.warning("يرجى إدخال مفتاح OpenAI API Key أولاً في الشريط الجانبي.")
        elif 'df' not in st.session_state:
            st.warning("يرجى جلب البيانات الفنية أولاً.")
        else:
            with st.spinner("جاري تحليل البيانات بواسطة GPT-4o..."):
                latest_data = st.session_state['df'].tail(5).to_string()
                ai_result = analyze_with_ai(latest_data, api_key)
                st.session_state['ai_result'] = ai_result

    if 'ai_result' in st.session_state:
        ai_res = st.session_state['ai_result']
        st.write("---")
        st.markdown("**نتيجة تحليل الذكاء الاصطناعي:**")
        st.info(ai_res)
        
        # استخراج القرار لتسهيل التنفيذ
        if "[RECOMMENDATION: BUY]" in ai_res:
            if st.button(f"🚀 تنفيذ أمر شراء ({quantity} أسهم)", type="primary"):
                msg = execute_ib_order("BUY", symbol, quantity)
                st.success(msg)
        elif "[RECOMMENDATION: SELL]" in ai_res:
            if st.button(f"🔻 تنفيذ أمر بيع ({quantity} أسهم)", type="primary"):
                msg = execute_ib_order("SELL", symbol, quantity)
                st.success(msg)
        else:
            st.warning("الذكاء الاصطناعي ينصح بالانتظار (HOLD)، لا توجد صفقة متاحة حالياً.")
