import streamlit as st
from ib_insync import *
import pandas as pd
import pandas_ta as ta
from openai import OpenAI
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 🤖 استدعاء محرك الذكاء الاصطناعي المحلي
from ai_models import LocalAITradingEngine

# ==========================================
# تهيئة الصفحة والإعدادات
# ==========================================
st.set_page_config(page_title="Local AI Trading Bot (IBKR)", layout="wide")
st.title("🤖 بوت التداول بمحرك الذكاء الاصطناعي المحلي (IBKR)")

# تهيئة المحرك في session_state
if 'ai_engine' not in st.session_state:
    st.session_state['ai_engine'] = LocalAITradingEngine()

# ==========================================
# الشريط الجانبي للإعدادات
# ==========================================
st.sidebar.header("⚙️ إعدادات الاتصال والحساب")

# إعدادات OpenAI (اختياري)
api_key = st.sidebar.text_input("OpenAI API Key (اختياري)", type="password", 
                                help="إذا كنت تريد تحليل متقدم باستخدام GPT-4o")

# إعدادات IBKR
ib_host = st.sidebar.text_input("IB Host", "127.0.0.1")
ib_port = st.sidebar.number_input("IB Port (Demo: 7497/4002)", value=7497)
symbol = st.sidebar.text_input("رمز السهم (Symbol)", "AAPL")
quantity = st.sidebar.number_input("الكمية المحددة", value=10, step=1)

# وضع التحليل
analysis_mode = st.sidebar.radio(
    "🧠 وضع التحليل:",
    ["المحرك المحلي فقط", "OpenAI فقط", "🌐 هجين (مدمج)"],
    help="المحرك المحلي: سريع ومجاني. OpenAI: تحليل متقدم. الهجين: يجمع بينهما"
)

# عرض حالة المحرك المحلي
st.sidebar.markdown("---")
st.sidebar.subheader("📊 حالة المحرك المحلي")
engine = st.session_state['ai_engine']

if engine.is_trained:
    st.sidebar.success("✅ النموذج المدرب: جاهز")
    
    # عرض أهم الميزات
    if hasattr(engine, 'feature_importance') and engine.feature_importance is not None:
        with st.sidebar.expander("📊 أهم الميزات المؤثرة"):
            st.dataframe(engine.feature_importance.head(5), use_container_width=True)
else:
    st.sidebar.warning("⚠️ النموذج غير مدرب (سيتم تدريبه عند جلب البيانات)")

# ==========================================
# دوال الاتصال وجلب البيانات
# ==========================================

def get_market_data(symbol, host, port):
    """الاتصال بـ IBKR وجلب البيانات الفنية"""
    ib = IB()
    try:
        # الاتصال
        ib.connect(host, int(port), clientId=99)
        
        # تعريف العقد
        contract = Stock(symbol, 'SMART', 'USD')
        
        # جلب البيانات التاريخية
        bars = ib.reqHistoricalData(
            contract, 
            endDateTime='', 
            durationStr='2 D',
            barSizeSetting='5 mins', 
            whatToShow='TRADES', 
            useRTH=True
        )
        
        # تحويل إلى DataFrame
        df = util.df(bars)
        
        # حساب المؤشرات الفنية
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['volume_ma'] = ta.sma(df['volume'], length=10)
        
        # إضافة عمود الوقت للرسم البياني
        df['date'] = df.index
        
        ib.disconnect()
        return df, None
        
    except Exception as e:
        return None, str(e)

def execute_ib_order(action, symbol, qty, host, port):
    """تنفيذ أمر التداول على الحساب التجريبي"""
    ib = IB()
    try:
        ib.connect(host, int(port), clientId=100)
        contract = Stock(symbol, 'SMART', 'USD')
        order = MarketOrder(action, qty)
        trade = ib.placeOrder(contract, order)
        ib.sleep(2)  # انتظار لتأكيد الإرسال
        
        # الحصول على تفاصيل الصفقة
        status = trade.orderStatus.status
        
        ib.disconnect()
        
        if status in ['Filled', 'Submitted']:
            return f"✅ تم إرسال أمر {action} لـ {qty} سهم في {symbol} بنجاح!"
        else:
            return f"⚠️ الأمر قيد التنفيذ... الحالة: {status}"
            
    except Exception as e:
        return f"❌ خطأ أثناء تنفيذ الأمر: {e}"

# ==========================================
# دوال التحليل
# ==========================================

def analyze_with_local_ai(df):
    """استخدام المحرك المحلي للتحليل"""
    engine = st.session_state['ai_engine']
    
    # تدريب النموذج إذا لم يكن مدرباً
    if not engine.is_trained:
        with st.spinner("جاري تدريب النموذج المحلي..."):
            engine.train_quick_model(df)
    
    # الحصول على التوصية
    action, confidence, reason = engine.predict_opportunity(df)
    
    # تنسيق النتيجة
    result = f"[RECOMMENDATION: {action}]\n"
    result += f"الثقة: {confidence}%\n"
    result += f"السبب: {reason}\n"
    
    # إضافة تفاصيل إضافية
    if action == 'BUY':
        result += "\n📈 إشارة شراء: النموذج يتوقع ارتفاعاً في السعر"
    elif action == 'SELL':
        result += "\n📉 إشارة بيع: النموذج يتوقع هبوطاً في السعر"
    else:
        result += "\n⏸️ انتظار: السوق في منطقة حياد"
    
    return result, action, confidence

def analyze_with_openai(df_summary, api_key, symbol_name):
    """استخدام OpenAI API للتحليل"""
    if not api_key:
        return "⚠️ مفتاح OpenAI API غير موجود", "HOLD", 0
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    أنت خبير تداول وسيگنال بالذكاء الاصطناعي. إليك أحدث البيانات الفنية لسهم {symbol_name}:
    {df_summary}

    قم بتحليل المؤشرات التالية:
    1. اتجاه السعر بالنسبة للمتوسطات SMA 20 و SMA 50.
    2. حالة مؤشر RSI.
    3. حجم التداول.

    أعطني إجابتك بتنسيق واضح ويحتوي مجبراً على إحدى الكلمات التالية في السطر الأول فقط:
    [RECOMMENDATION: BUY] أو [RECOMMENDATION: SELL] أو [RECOMMENDATION: HOLD]
    ثم اشرح السبب باختصار في أسطر لاحقة.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        ai_result = response.choices[0].message.content
        
        # استخراج الإجراء
        if "[RECOMMENDATION: BUY]" in ai_result:
            action = "BUY"
        elif "[RECOMMENDATION: SELL]" in ai_result:
            action = "SELL"
        else:
            action = "HOLD"
        
        # تقدير الثقة
        confidence = 70  # ثقة افتراضية لـ OpenAI
        
        return ai_result, action, confidence
        
    except Exception as e:
        return f"❌ خطأ في OpenAI: {str(e)}", "HOLD", 0

def analyze_hybrid(df, api_key, symbol_name):
    """تحليل هجين: المحرك المحلي + OpenAI"""
    # 1. التحليل المحلي
    local_result, local_action, local_confidence = analyze_with_local_ai(df)
    
    # 2. تحليل OpenAI (إذا كان المفتاح موجوداً)
    if api_key:
        latest_data = df.tail(10).to_string()
        openai_result, openai_action, openai_confidence = analyze_with_openai(
            latest_data, api_key, symbol_name
        )
    else:
        openai_result = "⚠️ لا يوجد مفتاح OpenAI"
        openai_action = "HOLD"
        openai_confidence = 0
    
    # 3. دمج النتائج
    hybrid_result = "🤖 **التحليل الهجين (المحلي + OpenAI)**\n\n"
    hybrid_result += f"**المحرك المحلي:** {local_action} (ثقة: {local_confidence}%)\n"
    
    if api_key:
        hybrid_result += f"**OpenAI:** {openai_action} (ثقة: {openai_confidence}%)\n\n"
    
    # اتخاذ القرار النهائي
    if local_action == openai_action and local_action != "HOLD":
        # توافق كامل
        final_action = local_action
        hybrid_result += f"✅ **توافق كامل**: كلا النموذجين يوصيان بـ {local_action}\n"
        hybrid_result += f"   الثقة النهائية: {(local_confidence + openai_confidence) / 2:.1f}%\n"
    elif local_action == "HOLD" and openai_action != "HOLD":
        # المحلي ينتظر ولكن OpenAI يوصي بشيء
        final_action = "HOLD"  # نعطي الأولوية للحذر
        hybrid_result += f"⚠️ **تباين**: المحلي يوصي بالانتظار، OpenAI يوصي بـ {openai_action}\n"
        hybrid_result += f"   القرار النهائي: انتظار (حذر)\n"
    elif openai_action == "HOLD" and local_action != "HOLD":
        final_action = "HOLD"  # نعطي الأولوية للحذر
        hybrid_result += f"⚠️ **تباين**: OpenAI يوصي بالانتظار، المحلي يوصي بـ {local_action}\n"
        hybrid_result += f"   القرار النهائي: انتظار (حذر)\n"
    else:
        final_action = "HOLD"
        hybrid_result += f"⏸️ **حياد**: النماذج غير حاسمة، انتظار أفضل\n"
    
    hybrid_result += f"\n---\n**تفاصيل المحرك المحلي:**\n{local_result}\n"
    
    if api_key:
        hybrid_result += f"\n**تفاصيل OpenAI:**\n{openai_result}"
    
    return hybrid_result, final_action, local_confidence

# ==========================================
# دوال الرسم البياني
# ==========================================

def plot_interactive_chart(df, symbol_name):
    """إنشاء الرسم البياني التفاعلي المتقدم"""
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        subplot_titles=(f'📈 سعر الشموع لـ {symbol_name}', '📊 مؤشر RSI', '📊 حجم التداول'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # 1. الشموع والمتوسطات
    fig.add_trace(
        go.Candlestick(
            x=df['date'], 
            open=df['open'], 
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            name='الأسعار',
            increasing_line_color='#00FF00',
            decreasing_line_color='#FF0000'
        ), 
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['SMA_20'], 
            mode='lines', 
            name='SMA 20', 
            line=dict(color='orange', width=2)
        ), 
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['SMA_50'], 
            mode='lines', 
            name='SMA 50', 
            line=dict(color='cyan', width=2)
        ), 
        row=1, col=1
    )
    
    # 2. RSI
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['RSI'], 
            mode='lines', 
            name='RSI',
            line=dict(color='purple', width=2)
        ), 
        row=2, col=1
    )
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # منطقة RSI
    fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="gray", opacity=0.1, row=2, col=1)
    
    # 3. حجم التداول
    colors = ['#00FF00' if df.iloc[i]['close'] >= df.iloc[i]['open'] else '#FF0000' 
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df['date'], 
            y=df['volume'], 
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['volume_ma'],
            mode='lines',
            name='Volume MA 10',
            line=dict(color='yellow', width=1)
        ),
        row=3, col=1
    )
    
    # تحديث التخطيط
    fig.update_layout(
        height=700, 
        xaxis_rangeslider_visible=False, 
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified'
    )
    
    # تحديث محاور الرسم
    fig.update_xaxes(title_text="الوقت", row=3, col=1)
    fig.update_yaxes(title_text="السعر ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="الحجم", row=3, col=1)
    
    return fig

def show_trading_signals(df):
    """عرض إشارات التداول على الرسم البياني"""
    engine = st.session_state['ai_engine']
    
    # الحصول على آخر إشارة
    if engine.is_trained:
        action, confidence, reason = engine.predict_opportunity(df)
        
        # إضافة علامات على الرسم
        last_price = df['close'].iloc[-1]
        last_date = df['date'].iloc[-1]
        
        if action == 'BUY':
            return f"🟢 شراء عند {last_price:.2f}$ (ثقة: {confidence}%)"
        elif action == 'SELL':
            return f"🔴 بيع عند {last_price:.2f}$ (ثقة: {confidence}%)"
        else:
            return f"⏸️ انتظار - السعر الحالي: {last_price:.2f}$"
    
    return "⚠️ النموذج غير مدرب"

# ==========================================
# الواجهة الرئيسية
# ==========================================

# إنشاء أعمدة رئيسية
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📊 فحص البيانات الفنية")
    
    # زر جلب البيانات
    if st.button("🔄 جلب البيانات الفنية الآن", use_container_width=True):
        with st.spinner("جاري الاتصال بـ IBKR وجلب الشموع..."):
            df, error = get_market_data(symbol, ib_host, ib_port)
            
            if error:
                st.error(f"❌ فشل الاتصال بـ IBKR: {error}")
                st.info("💡 تأكد من تشغيل TWS أو IB Gateway مع تمكين API")
            else:
                # تخزين البيانات
                st.session_state['df'] = df
                
                # تدريب المحرك المحلي تلقائياً
                with st.spinner("🧠 جاري تدريب النموذج المحلي..."):
                    engine = st.session_state['ai_engine']
                    if engine.train_quick_model(df):
                        st.success("✅ تم تدريب النموذج المحلي بنجاح!")
                    else:
                        st.warning("⚠️ بيانات غير كافية لتدريب النموذج (يحتاج 50 شمعة على الأقل)")
                
                st.success(f"✅ تم جلب {len(df)} شمعة لـ {symbol} بنجاح!")
                
                # عرض آخر سعر
                last_price = df['close'].iloc[-1]
                last_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                st.metric(
                    label=f"آخر سعر لـ {symbol}",
                    value=f"${last_price:.2f}",
                    delta=f"{last_change:.2f}%"
                )
    
    # عرض البيانات والرسم البياني
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # عرض الرسم البياني
        st.subheader("📈 الرسم البياني التفاعلي")
        fig = plot_interactive_chart(df, symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        # عرض الإشارات
        signal = show_trading_signals(df)
        st.info(f"🚦 **آخر إشارة:** {signal}")
        
        # عرض الجدول
        with st.expander("📋 عرض تفاصيل الجدول"):
            display_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'SMA_20', 'SMA_50']
            st.dataframe(
                df[display_cols].tail(10).style.format({
                    'close': '${:.2f}',
                    'open': '${:.2f}',
                    'high': '${:.2f}',
                    'low': '${:.2f}',
                    'volume': '{:,.0f}',
                    'RSI': '{:.1f}',
                    'SMA_20': '${:.2f}',
                    'SMA_50': '${:.2f}'
                }),
                use_container_width=True
            )

with col2:
    st.subheader("🤖 تحليل الذكاء الاصطناعي وتنفيذ الأوامر")
    
    # زر التحليل
    if st.button("🧠 تحليل الفرصة الآن", use_container_width=True, type="primary"):
        if 'df' not in st.session_state:
            st.warning("⚠️ يرجى جلب البيانات الفنية أولاً")
        else:
            df = st.session_state['df']
            
            with st.spinner(f"جاري تحليل البيانات باستخدام {analysis_mode}..."):
                
                if analysis_mode == "المحرك المحلي فقط":
                    result, action, confidence = analyze_with_local_ai(df)
                    st.session_state['ai_result'] = result
                    st.session_state['recommended_action'] = action
                    st.session_state['confidence'] = confidence
                    
                elif analysis_mode == "OpenAI فقط":
                    if not api_key:
                        st.warning("⚠️ يرجى إدخال مفتاح OpenAI API Key")
                    else:
                        latest_data = df.tail(10).to_string()
                        result, action, confidence = analyze_with_openai(latest_data, api_key, symbol)
                        st.session_state['ai_result'] = result
                        st.session_state['recommended_action'] = action
                        st.session_state['confidence'] = confidence
                        
                else:  # هجين
                    result, action, confidence = analyze_hybrid(df, api_key, symbol)
                    st.session_state['ai_result'] = result
                    st.session_state['recommended_action'] = action
                    st.session_state['confidence'] = confidence
                
                st.success("✅ تم التحليل بنجاح!")
    
    # عرض النتيجة
    if 'ai_result' in st.session_state:
        ai_res = st.session_state['ai_result']
        action = st.session_state.get('recommended_action', 'HOLD')
        confidence = st.session_state.get('confidence', 0)
        
        st.markdown("---")
        st.markdown("**📊 نتيجة التحليل:**")
        
        # عرض النتيجة في صندوق منسق
        with st.container():
            # إضافة لون حسب الإشارة
            if action == "BUY":
                st.success(f"🟢 **توصية: شراء** (ثقة: {confidence}%)")
            elif action == "SELL":
                st.error(f"🔴 **توصية: بيع** (ثقة: {confidence}%)")
            else:
                st.warning(f"⏸️ **توصية: انتظار** (ثقة: {confidence}%)")
            
            st.text_area("التفاصيل:", ai_res, height=200, key="result_display")
        
        # أزرار التنفيذ
        st.markdown("---")
        st.subheader("💼 تنفيذ الصفقة")
        
        col_exec1, col_exec2 = st.columns(2)
        
        if action == "BUY":
            with col_exec1:
                if st.button(f"🚀 تنفيذ شراء ({quantity} سهم)", use_container_width=True, type="primary"):
                    msg = execute_ib_order("BUY", symbol, quantity, ib_host, ib_port)
                    if "✅" in msg:
                        st.success(msg)
                    else:
                        st.error(msg)
            
            with col_exec2:
                if st.button("📊 محاكاة الشراء", use_container_width=True):
                    df = st.session_state['df']
                    last_price = df['close'].iloc[-1]
                    total = last_price * quantity
                    st.info(f"✅ **محاكاة:** شراء {quantity} سهم من {symbol} بسعر ${last_price:.2f} = ${total:,.2f}")
        
        elif action == "SELL":
            with col_exec1:
                if st.button(f"🔻 تنفيذ بيع ({quantity} سهم)", use_container_width=True, type="primary"):
                    msg = execute_ib_order("SELL", symbol, quantity, ib_host, ib_port)
                    if "✅" in msg:
                        st.success(msg)
                    else:
                        st.error(msg)
            
            with col_exec2:
                if st.button("📊 محاكاة البيع", use_container_width=True):
                    df = st.session_state['df']
                    last_price = df['close'].iloc[-1]
                    total = last_price * quantity
                    st.info(f"✅ **محاكاة:** بيع {quantity} سهم من {symbol} بسعر ${last_price:.2f} = ${total:,.2f}")
        
        else:  # HOLD
            st.info("⏸️ **لا توجد صفقة**: النموذج يوصي بالانتظار")
            
            # عرض تفاصيل إضافية
            with st.expander("📈 تفاصيل السوق الحالية"):
                if 'df' in st.session_state:
                    df = st.session_state['df']
                    latest = df.iloc[-1]
                    
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    with col_metrics1:
                        st.metric("آخر سعر", f"${latest['close']:.2f}")
                        st.metric("RSI", f"{latest['RSI']:.1f}")
                    
                    with col_metrics2:
                        st.metric("SMA 20", f"${latest['SMA_20']:.2f}")
                        st.metric("SMA 50", f"${latest['SMA_50']:.2f}")
                    
                    with col_metrics3:
                        diff_sma = ((latest['close'] - latest['SMA_20']) / latest['SMA_20']) * 100
                        st.metric("البعد عن SMA20", f"{diff_sma:.1f}%")
                        st.metric("حجم التداول", f"{latest['volume']:,.0f}")
        
        # زر مسح النتائج
        if st.button("🗑️ مسح النتائج", use_container_width=True):
            st.session_state.pop('ai_result', None)
            st.session_state.pop('recommended_action', None)
            st.session_state.pop('confidence', None)
            st.rerun()

# ==========================================
# معلومات إضافية في الأسفل
# ==========================================
with st.expander("ℹ️ معلومات عن البوت"):
    st.markdown("""
    ### 🤖 كيف يعمل هذا البوت؟
    
    1. **جلب البيانات**: يتصل بـ Interactive Brokers (TWS/IB Gateway) لجلب بيانات الشموع
    2. **المؤشرات الفنية**: يحسب RSI، SMA 20، SMA 50، ومتوسط حجم التداول
    3. **التحليل**: يستخدم ثلاثة أوضاع:
       - **المحرك المحلي**: نموذج Random Forest مدرب محلياً (سريع ومجاني)
       - **OpenAI**: تحليل متقدم باستخدام GPT-4o (يتطلب مفتاح API)
       - **هجين**: يجمع بين الاثنين للحصول على أفضل النتائج
    4. **التنفيذ**: ينفذ الأوامر مباشرة على حساب IBKR التجريبي
    
    ### ⚠️ تنبيهات مهمة
    - هذا البوت للأغراض التعليمية فقط
    - تأكد من اختباره على الحساب التجريبي أولاً
    - لا تخاطر بأكثر مما يمكنك تحمل خسارته
    - راقب أداء النموذج باستمرار
    
    ### 📚 متطلبات التشغيل
    - TWS أو IB Gateway مفتوح مع تمكين API
    - Python 3.8+
    - المكتبات: streamlit, ib-insync, pandas, pandas_ta, openai, plotly
    """)

# عرض إحصائيات الجلسة
st.sidebar.markdown("---")
st.sidebar.subheader("📊 إحصائيات الجلسة")

if 'df' in st.session_state:
    df = st.session_state['df']
    st.sidebar.metric("عدد الشموع", len(df))
    st.sidebar.metric("آخر سعر", f"${df['close'].iloc[-1]:.2f}")
    
    # حساب التغير
    if len(df) > 1:
        change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
        st.sidebar.metric("التغير الأخير", f"{change:.2f}%")

if 'recommended_action' in st.session_state:
    action = st.session_state['recommended_action']
    confidence = st.session_state.get('confidence', 0)
    
    if action == "BUY":
        st.sidebar.success(f"🟢 آخر توصية: شراء ({confidence}%)")
    elif action == "SELL":
        st.sidebar.error(f"🔴 آخر توصية: بيع ({confidence}%)")
    else:
        st.sidebar.warning(f"⏸️ آخر توصية: انتظار")

# إضافة زر لإعادة تعيين النموذج
if st.sidebar.button("🔄 إعادة تعيين النموذج المحلي"):
    st.session_state['ai_engine'] = LocalAITradingEngine()
    st.sidebar.success("✅ تم إعادة تعيين النموذج")
    st.rerun()
