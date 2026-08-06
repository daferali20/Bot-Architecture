import streamlit as st
from ib_insync import *
import pandas as pd
import pandas_ta as ta
from openai import OpenAI
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 🤖 استدعاء محرك الذكاء الاصطناعي المحلي
from ai_models import LocalAITradingEngine

# تهيئة المحرك المحلي (مرة واحدة)
if 'ai_engine' not in st.session_state:
    st.session_state['ai_engine'] = LocalAITradingEngine()

# ==========================================
# دوال التحليل المزدوج (محلي + OpenAI)
# ==========================================

def analyze_with_local_ai(df):
    """
    استخدام المحرك المحلي للتحليل
    """
    engine = st.session_state['ai_engine']
    
    # تدريب النموذج على البيانات الحالية
    if not engine.is_trained:
        engine.train_quick_model(df)
    
    # الحصول على التوصية
    action, confidence, reason = engine.predict_opportunity(df)
    
    # تنسيق النتيجة بنفس نمط OpenAI
    result = f"[RECOMMENDATION: {action}]\n"
    result += f"الثقة: {confidence}%\n"
    result += f"السبب: {reason}\n"
    
    # إضافة تفاصيل إضافية للمستخدم
    if action == 'BUY':
        result += "📈 إشارة شراء: النموذج يتوقع ارتفاعاً"
    elif action == 'SELL':
        result += "📉 إشارة بيع: النموذج يتوقع هبوطاً"
    else:
        result += "⏸️ انتظار: السوق في منطقة حياد"
    
    return result

def analyze_with_openai(df_summary, api_key, symbol):
    """
    استخدام OpenAI API للتحليل (الطريقة الحالية)
    """
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

def analyze_with_hybrid(df, api_key=None, symbol="AAPL"):
    """
    تحليل هجين: يستخدم المحرك المحلي أولاً، ثم يعزز بـ OpenAI إذا توفر المفتاح
    """
    # 1. التحليل المحلي أولاً
    local_result = analyze_with_local_ai(df)
    
    # استخراج التوصية المحلية
    if "[RECOMMENDATION: BUY]" in local_result:
        local_action = "BUY"
    elif "[RECOMMENDATION: SELL]" in local_result:
        local_action = "SELL"
    else:
        local_action = "HOLD"
    
    # 2. إذا كان مفتاح OpenAI متاحاً، نحصل على تحليل إضافي للمقارنة
    if api_key:
        latest_data = df.tail(10).to_string()
        openai_result = analyze_with_openai(latest_data, api_key, symbol)
        
        # استخراج توصية OpenAI
        if "[RECOMMENDATION: BUY]" in openai_result:
            openai_action = "BUY"
        elif "[RECOMMENDATION: SELL]" in openai_result:
            openai_action = "SELL"
        else:
            openai_action = "HOLD"
        
        # دمج النتائج
        hybrid_result = f"🤖 **التحليل الهجين (المحلي + OpenAI)**\n\n"
        hybrid_result += f"**المحرك المحلي:** {local_action}\n"
        hybrid_result += f"**OpenAI:** {openai_action}\n\n"
        
        # حالة التوافق
        if local_action == openai_action:
            hybrid_result += f"✅ **توافق**: كلا النموذجين يوصيان بـ {local_action}\n"
        else:
            hybrid_result += f"⚠️ **تباين**: النماذج مختلفة - يُفضل التريث\n"
            # في حالة التباين، نعطي الأولية للمحرك المحلي (أسرع وأكثر استقراراً)
            local_action = "HOLD"
        
        hybrid_result += f"\n---\n**تحليل المحرك المحلي:**\n{local_result}\n"
        hybrid_result += f"\n**تحليل OpenAI:**\n{openai_result}"
        
        return hybrid_result, local_action
    else:
        # العودة للتحليل المحلي فقط
        return local_result, local_action
def plot_interactive_chart(df, symbol):
    """إنشاء الشارت التفاعلي"""
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(f'سعر الشموع لـ {symbol}', 'مؤشر RSI', 'حجم التداول'),
        row_heights=[0.6, 0.2, 0.2]
    )

    # الشموع
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
    
    # المتوسطات
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange')), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='green')), 
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')), 
        row=2, col=1
    )
    # خطوط RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # حجم التداول
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], name='Volume', marker_color='lightblue'),
        row=3, col=1
    )
    
    fig.update_layout(
        height=600, 
        xaxis_rangeslider_visible=False, 
        template='plotly_dark',
        showlegend=True
    )
    
    return fig
# ==========================================
# الواجهة الرئيسية
# ==========================================

st.set_page_config(page_title="Local AI Trading Bot (IBKR)", layout="wide")
st.title("🤖 بوت التداول بمحرك الذكاء الاصطناعي المحلي (IBKR)")

# الشريط الجانبي
st.sidebar.header("⚙️ إعدادات الاتصال والحساب")
api_key = st.sidebar.text_input("OpenAI API Key (اختياري)", type="password")
ib_host = st.sidebar.text_input("IB Host", "127.0.0.1")
ib_port = st.sidebar.number_input("IB Port (Demo: 7497/4002)", value=7497)
symbol = st.sidebar.text_input("رمز السهم (Symbol)", "AAPL")
quantity = st.sidebar.number_input("الكمية المحددة", value=10, step=1)

# اختيار نوع التحليل
analysis_mode = st.sidebar.radio(
    "اختر نوع التحليل:",
    ["المحرك المحلي فقط", "OpenAI فقط", "هجين (مدمج)"],
    help="الهجين يستخدم المحرك المحلي أولاً ثم يعزز بـ OpenAI"
)

# عرض حالة المحرك المحلي
st.sidebar.markdown("---")
st.sidebar.subheader("📊 حالة المحرك المحلي")
engine = st.session_state['ai_engine']
if engine.is_trained:
    st.sidebar.success("✅ النموذج المدرب: جاهز")
    # عرض أهمية الميزات
    if engine.feature_importance is not None:
        with st.sidebar.expander("أهم الميزات"):
            st.dataframe(engine.feature_importance.head(5))
else:
    st.sidebar.warning("⚠️ النموذج غير مدرب (سيتم تدريبه تلقائياً)")

# ==========================================
# الأعمدة الرئيسية
# ==========================================

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 فحص البيانات الفنية")
    
    if st.button("🔄 جلب البيانات الفنية الآن"):
        with st.spinner("جاري الاتصال بـ IBKR وجلب الشموع..."):
            df, error = get_market_data(symbol)
            if error:
                st.error(f"فشل الاتصال بـ IBKR: {error}")
            else:
                st.session_state['df'] = df
                # تدريب المحرك المحلي تلقائياً
                with st.spinner("جاري تدريب النموذج المحلي..."):
                    engine.train_quick_model(df)
                st.success(f"✅ تم جلب البيانات وتدريب النموذج بنجاح!")
    
    # عرض البيانات
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # عرض الشارت
        st.subheader("📈 الرسم البياني التفاعلي")
        fig = plot_interactive_chart(df, symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        # عرض الجدول
        with st.expander("📋 عرض تفاصيل الجدول"):
            st.dataframe(
                df[['date', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'SMA_20', 'SMA_50']].tail(10),
                use_container_width=True
            )

with col2:
    st.subheader("🤖 تحليل الذكاء الاصطناعي وتنفيذ الأوامر")
    
    if st.button("🧠 تحليل الفرصة"):
        if 'df' not in st.session_state:
            st.warning("⚠️ يرجى جلب البيانات الفنية أولاً.")
        else:
            with st.spinner("جاري تحليل البيانات..."):
                df = st.session_state['df']
                
                # اختيار نوع التحليل
                if analysis_mode == "المحرك المحلي فقط":
                    ai_result, action = analyze_with_hybrid(df, api_key=None, symbol=symbol)
                    st.session_state['ai_result'] = ai_result
                    st.session_state['recommended_action'] = action
                
                elif analysis_mode == "OpenAI فقط":
                    if not api_key:
                        st.warning("⚠️ يرجى إدخال مفتاح OpenAI API Key")
                    else:
                        latest_data = df.tail(10).to_string()
                        ai_result = analyze_with_openai(latest_data, api_key, symbol)
                        st.session_state['ai_result'] = ai_result
                        # استخراج الإجراء
                        if "[RECOMMENDATION: BUY]" in ai_result:
                            st.session_state['recommended_action'] = "BUY"
                        elif "[RECOMMENDATION: SELL]" in ai_result:
                            st.session_state['recommended_action'] = "SELL"
                        else:
                            st.session_state['recommended_action'] = "HOLD"
                
                else:  # هجين
                    ai_result, action = analyze_with_hybrid(df, api_key, symbol)
                    st.session_state['ai_result'] = ai_result
                    st.session_state['recommended_action'] = action
    
    # عرض النتيجة
    if 'ai_result' in st.session_state:
        ai_res = st.session_state['ai_result']
        st.markdown("---")
        st.markdown("**📊 نتيجة التحليل:**")
        
        # عرض النتيجة في صندوق
        with st.container():
            st.info(ai_res)
        
        # أزرار التنفيذ
        action = st.session_state.get('recommended_action', 'HOLD')
        
        col_exec1, col_exec2 = st.columns(2)
        
        if action == "BUY":
            with col_exec1:
                if st.button(f"🚀 تنفيذ شراء ({quantity} أسهم)", type="primary", use_container_width=True):
                    msg = execute_ib_order("BUY", symbol, quantity)
                    if "نجاح" in msg:
                        st.success(msg)
                    else:
                        st.error(msg)
            with col_exec2:
                if st.button("📊 محاكاة الشراء", use_container_width=True):
                    st.info(f"✅ محاكاة: شراء {quantity} سهم من {symbol} بسعر {df['close'].iloc[-1]:.2f}")
        
        elif action == "SELL":
            with col_exec1:
                if st.button(f"🔻 تنفيذ بيع ({quantity} أسهم)", type="primary", use_container_width=True):
                    msg = execute_ib_order("SELL", symbol, quantity)
                    if "نجاح" in msg:
                        st.success(msg)
                    else:
                        st.error(msg)
            with col_exec2:
                if st.button("📊 محاكاة البيع", use_container_width=True):
                    st.info(f"✅ محاكاة: بيع {quantity} سهم من {symbol} بسعر {df['close'].iloc[-1]:.2f}")
        
        else:  # HOLD
            st.warning("⏸️ الذكاء الاصطناعي ينصح بالانتظار (HOLD)، لا توجد صفقة متاحة حالياً.")
            
            # عرض إحصائيات إضافية
            with st.expander("📈 تفاصيل إضافية"):
                latest = df.iloc[-1]
                st.metric("آخر سعر", f"${latest['close']:.2f}")
                st.metric("RSI", f"{latest['RSI']:.1f}")
                st.metric("SMA 20", f"${latest['SMA_20']:.2f}")
                st.metric("SMA 50", f"${latest['SMA_50']:.2f}")
                
                # حساب القوة النسبية
                price_vs_sma20 = ((latest['close'] - latest['SMA_20']) / latest['SMA_20']) * 100
                st.metric("البعد عن SMA20", f"{price_vs_sma20:.1f}%")

# ==========================================
# دوال تنفيذ الأوامر
# ==========================================

def execute_ib_order(action, symbol, qty):
    """تنفيذ أمر التداول على الحساب التجريبي"""
    ib = IB()
    try:
        ib.connect(ib_host, int(ib_port), clientId=100)
        contract = Stock(symbol, 'SMART', 'USD')
        order = MarketOrder(action, qty)
        trade = ib.placeOrder(contract, order)
        ib.sleep(2)
        ib.disconnect()
        return f"✅ تم إرسال أمر {action} لـ {qty} سهم في {symbol} بنجاح!"
    except Exception as e:
        return f"❌ خطأ أثناء تنفيذ الأمر: {e}"

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
