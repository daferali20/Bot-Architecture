import time
from ib_insync import *
import pandas as pd
import pandas_ta as ta  # مكتبة حساب المؤشرات الفنية

# 1. الاتصال بمنصة Interactive Brokers
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# 2. تحديد السهم والكمية المحددة
SYMBOL = 'AAPL'
FIXED_QUANTITY = 10  # الكمية المحددة لكل صفقة
contract = Stock(SYMBOL, 'SMART', 'USD')

def get_historical_data(contract):
    """جلب بيانات الأسعار التاريخية واستخراج المؤشرات"""
    # طلب الشموع (شمعة كل 5 دقائق على مدار يومين)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='2 D',
        barSizeSetting='5 mins',
        whatToShow='TRADES',
        useRTH=True
    )
    
    # تحويل البيانات إلى DF
    df = util.df(bars)
    
    # حساب المتوسط المتحرك السريع والبطيء
    df['SMA_20'] = ta.sma(df['close'], length=20)
    df['SMA_50'] = ta.sma(df['close'], length=50)
    
    # حساب مؤشر RSI (فترة 14)
    df['RSI'] = ta.rsi(df['close'], length=14)
    
    return df

def check_signal(df):
    """فحص توفر فرصة الشراء"""
    latest = df.iloc[-1]       # الشمعة الحالية
    previous = df.iloc[-2]     # الشمعة السابقة
    
    # شروط دخول صفقة شراء (Bullish Signal):
    # 1. تقاطع صاعد للمتوسط السريع (20) فوق المتوسط البطيء (50)
    # 2. مؤشر RSI أقل من 70 (لتجنب المناطق المشتراة بشكل مفرط)
    ma_crossover = (previous['SMA_20'] <= previous['SMA_50']) and (latest['SMA_20'] > latest['SMA_50'])
    rsi_condition = latest['RSI'] < 70
    
    if ma_crossover and rsi_condition:
        return 'BUY'
    
    return None

def run_bot():
    """حلقة التشغيل المستمرة لتفحص السوق"""
    print(f"بدء مراقبة السهم {SYMBOL}...")
    
    while True:
        try:
            df = get_historical_data(contract)
            signal = check_signal(df)
            
            latest_price = df['close'].iloc[-1]
            latest_rsi = round(df['RSI'].iloc[-1], 2)
            
            print(f"السعر الحالي: {latest_price} | RSI: {latest_rsi}")
            
            if signal == 'BUY':
                # التأكد من عدم وجود أوامر معلقة بنفس السهم
                open_orders = ib.openTrades()
                has_pending_order = any(t.contract.symbol == SYMBOL for t in open_orders)
                
                if not has_pending_order:
                    order = MarketOrder('BUY', FIXED_QUANTITY)
                    trade = ib.placeOrder(contract, order)
                    print(f"🚀 تم اكتشاف فرصة! تم إرسال أمر شراء {FIXED_QUANTITY} سهم في {SYMBOL}")
                else:
                    print("توجد صفقة أو أمر معلق بالفعل لنفس السهم.")
            
            # الانتظار 5 دقائق قبل الفحص التالي
            time.sleep(300)
            
        except Exception as e:
            print(f"حدث خطأ أثناء فحص البيانات: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
