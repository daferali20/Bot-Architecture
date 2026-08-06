import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier

class LocalAITradingEngine:
    """
    محرك الذكاء الاصطناعي المحلي المطور مع دعم MACD و Bollinger Bands
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def extract_features(self, df):
        """استخراج الميزات الفنية والمؤشرات المركبة"""
        data = df.copy()
        
        # 1. المؤشرات الأساسية السابقة
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=10).std()
        data['ma_diff'] = data['SMA_20'] - data['SMA_50']
        data['dist_sma20'] = (data['close'] - data['SMA_20']) / data['SMA_20']
        data['rsi_momentum'] = data['RSI'].diff()
        
        # 2. إضافة مؤشر MACD
        macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            data['macd_hist'] = macd['MACDh_12_26_9']  # قيم الهستغرام
        else:
            data['macd_hist'] = 0

        # 3. إضافة نطاقات بولينجر (Bollinger Bands)
        bb = ta.bbands(data['close'], length=20, std=2)
        if bb is not None and not bb.empty:
            data['bb_upper'] = bb['BBU_20_2.0']
            data['bb_lower'] = bb['BBL_20_2.0']
            # %B: نسبة موقع السعر بين النطاقين (0 = السفلي، 1 = العلوي)
            data['bb_percent'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-9)
        else:
            data['bb_percent'] = 0.5

        return data

    def train_quick_model(self, df):
        """تدريب النموذج بناءً على قائمة الميزات الجديدة"""
        data = self.extract_features(df).dropna()
        if len(data) < 35:
            return False
        
        # Target: 1 للحركة الصاعدة في الشمعة التالية
        data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        
        # قائمة كافة الميزات بما فيها MACD و Bollinger Bands
        features = [
            'RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility', 
            'ma_diff', 'dist_sma20', 'rsi_momentum', 
            'macd_hist', 'bb_percent'
        ]
        
        X = data[features][:-1]
        y = data['target'][:-1]
        
        if len(X) > 10:
            self.model.fit(X, y)
            self.is_trained = True
            return True
        return False

    def predict_opportunity(self, df):
        """تحليل الفرصة بناءً على مخرجات النموذج وقواعد المؤشرات المركبة"""
        data = self.extract_features(df).dropna()
        if data.empty:
            return 'HOLD', 0.0, "لا تتوفر بيانات كافية للتحليل."

        latest = data.iloc[-1]
        
        if not self.is_trained:
            self.train_quick_model(df)
            
        features = [
            'RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility', 
            'ma_diff', 'dist_sma20', 'rsi_momentum', 
            'macd_hist', 'bb_percent'
        ]
        
        latest_features = np.array(latest[features]).reshape(1, -1)
        
        if self.is_trained:
            prob_up = self.model.predict_proba(latest_features)[0][1]
        else:
            prob_up = 0.5

        rsi_val = round(latest['RSI'], 2)
        macd_h = round(latest['macd_hist'], 4)
        bb_pct = round(latest['bb_percent'], 2)
        
        # شروط التأكيد المركبة:
        macd_bullish = macd_h > 0
        bb_oversold = bb_pct < 0.2    # السعر متواجد قرب الحد السفلي
        bb_overbought = bb_pct > 0.8  # السعر متواجد قرب الحد العلوي

        # اتخاذ القرار
        if prob_up > 0.55 and macd_bullish and rsi_val < 65:
            confidence = round(prob_up * 100, 1)
            reason = f"فرصة شراء: احتمالية ML للصعود {confidence}% | هستغرام MACD إيجابي ({macd_h}) | موقع بولينجر (%B={bb_pct})."
            return 'BUY', confidence, reason
            
        elif prob_up < 0.40 or bb_overbought or rsi_val > 70:
            confidence = round((1 - prob_up) * 100, 1)
            reason = f"إشارة بيع/تراجع: السعر عند النطاق العلوي لبولينجر (%B={bb_pct}) أو RSI مرتفع ({rsi_val})."
            return 'SELL', confidence, reason
            
        else:
            confidence = round(abs(prob_up - 0.5) * 200, 1)
            reason = f"انتظار: مؤشر MACD عند ({macd_h}) وموقع بولينجر (%B={bb_pct}) لا يظهران اتجاهاً حاسماً."
            return 'HOLD', confidence, reason
