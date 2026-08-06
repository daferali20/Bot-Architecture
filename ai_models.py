import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class LocalAITradingEngine:
    """
    محرك الذكاء الاصطناعي والتعلم الآلي المحلي لاكتشاف الفرص
    """
    def __init__(self):
        # نموذج تعلم آلي محلي (Random Forest)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def extract_features(self, df):
        """استخراج وتجهيز الميزات الفنية (Features) للنموذج"""
        data = df.copy()
        
        # 1. التغير النسبي في السعر
        data['returns'] = data['close'].pct_change()
        
        # 2. مؤشر الفولاذية (Volatility)
        data['volatility'] = data['returns'].rolling(window=10).std()
        
        # 3. اتجاه المتوسطات (MA Crossover Signal)
        data['ma_diff'] = data['SMA_20'] - data['SMA_50']
        
        # 4. موقع السعر بالنسبة للمتوسط
        data['dist_sma20'] = (data['close'] - data['SMA_20']) / data['SMA_20']
        
        # 5. زخم مؤشر RSI
        data['rsi_momentum'] = data['RSI'].diff()
        
        return data

    def train_quick_model(self, df):
        """تدريب سريع للنموذج المحلي على بيانات السهم الحالية"""
        data = self.extract_features(df).dropna()
        if len(data) < 30:
            return False
        
        # الهدف (Target): 1 إذا ارتفع السعر في الشمعة التالية، 0 إذا انخفض
        data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        
        features = ['RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility', 'ma_diff', 'dist_sma20', 'rsi_momentum']
        X = data[features][:-1]
        y = data['target'][:-1]
        
        if len(X) > 10:
            self.model.fit(X, y)
            self.is_trained = True
            return True
        return False

    def predict_opportunity(self, df):
        """
        تحليل الشمعة الأخيرة واكتشاف الفرصة بواسطة النموذج المحلي
        Returns:
            action (str): 'BUY', 'SELL', 'HOLD'
            confidence (float): نسبة ثقة الذكاء الاصطناعي (0% - 100%)
            reason (str): شرح السبب الفني
        """
        data = self.extract_features(df).dropna()
        if data.empty:
            return 'HOLD', 0.0, "لا تتوفر بيانات كافية للتحليل."

        latest = data.iloc[-1]
        
        # 1. تدريب النموذج محلياً إذا لم يكن مدرباً
        if not self.is_trained:
            self.train_quick_model(df)
            
        features = ['RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility', 'ma_diff', 'dist_sma20', 'rsi_momentum']
        latest_features = np.array(latest[features]).reshape(1, -1)
        
        # التنبؤ باحتمالية الصعود
        if self.is_trained:
            prob_up = self.model.predict_proba(latest_features)[0][1]
        else:
            prob_up = 0.5

        rsi_val = round(latest['RSI'], 2)
        sma20 = latest['SMA_20']
        sma50 = latest['SMA_50']
        close_price = latest['close']
        
        # قواعد اتخاذ القرار المركبة (ML + Technical Logic)
        if prob_up > 0.60 and rsi_val < 65 and close_price > sma20:
            confidence = round(prob_up * 100, 1)
            reason = f"نموذج ML يتوقع صعوداً بثقة {confidence}%. المؤشر RSI عند {rsi_val} والسعر فوق SMA20."
            return 'BUY', confidence, reason
            
        elif prob_up < 0.35 or rsi_val > 75:
            confidence = round((1 - prob_up) * 100, 1)
            reason = f"إشارة هبوط/تشبع شرائي! RSI عند {rsi_val}، احتمالية الصعود منخفضة ({round(prob_up*100, 1)}%)."
            return 'SELL', confidence, reason
            
        else:
            confidence = round(abs(prob_up - 0.5) * 200, 1)
            reason = f"السوق في حالة تذبذب/حياد. RSI عند {rsi_val} واحتمالية الاتجاه غير حاسمة."
            return 'HOLD', confidence, reason
