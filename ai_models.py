import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LocalAITradingEngine:
    """
    محرك تداول بالذكاء الاصطناعي - نسخة محسّنة
    """
    def __init__(self, min_samples=50, max_position=0.30):
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.is_trained = False
        self.min_samples = min_samples
        self.max_position = max_position
        self.feature_importance = None
        
    def extract_features(self, df):
        """استخراج ميزات محسّنة"""
        data = df.copy()
        
        # الميزات الأساسية
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(10).std()
        data['ma_diff'] = data['SMA_20'] - data['SMA_50']
        data['dist_sma20'] = (data['close'] - data['SMA_20']) / data['SMA_20']
        data['dist_sma50'] = (data['close'] - data['SMA_50']) / data['SMA_50']
        data['rsi_momentum'] = data['RSI'].diff()
        
        # ميزات متقدمة
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        data['price_range'] = (data['high'] - data['low']) / data['close']
        data['body_ratio'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
        
        # ميزات التأخر (للتنبؤ التسلسلي)
        for lag in [1, 2]:
            data[f'return_lag_{lag}'] = data['returns'].shift(lag)
            data[f'rsi_lag_{lag}'] = data['RSI'].shift(lag)
        
        return data
    
    def train_quick_model(self, df):
        """تدريب النموذج مع التحقق من الجودة"""
        data = self.extract_features(df).dropna()
        
        if len(data) < self.min_samples:
            return False
        
        # تجهيز الهدف
        data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        
        features = ['RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility', 
                   'ma_diff', 'dist_sma20', 'dist_sma50', 'rsi_momentum',
                   'volume_ratio', 'price_range', 'body_ratio',
                   'return_lag_1', 'return_lag_2', 'rsi_lag_1', 'rsi_lag_2']
        
        X = data[features][:-1]
        y = data['target'][:-1]
        
        if len(X) < 20:
            return False
        
        # تقسيم البيانات للتحقق
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # تدريب النموذج
        self.model.fit(X_train, y_train)
        
        # تقييم الأداء
        accuracy = self.model.score(X_test, y_test)
        
        if accuracy > 0.55:  # أفضل من العشوائية
            self.is_trained = True
            self.feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"✅ تم تدريب النموذج بدقة: {accuracy*100:.1f}%")
            return True
        else:
            self.is_trained = False
            print(f"⚠️ النموذج ضعيف (دقة {accuracy*100:.1f}%) - لم يتم اعتماده")
            return False
    
    def predict_opportunity(self, df):
        """التنبؤ بالفرصة مع حساب الثقة"""
        data = self.extract_features(df).dropna()
        if data.empty:
            return 'HOLD', 0.0, "بيانات غير كافية"
        
        latest = data.iloc[-1]
        
        # تدريب النموذج إذا لزم الأمر
        if not self.is_trained:
            self.train_quick_model(df)
        
        features = ['RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility', 
                   'ma_diff', 'dist_sma20', 'dist_sma50', 'rsi_momentum',
                   'volume_ratio', 'price_range', 'body_ratio',
                   'return_lag_1', 'return_lag_2', 'rsi_lag_1', 'rsi_lag_2']
        
        latest_features = np.array(latest[features]).reshape(1, -1)
        
        # حساب الثقة مع معايرة
        if self.is_trained:
            prob_up = self.model.predict_proba(latest_features)[0][1]
            
            # حساب عدم اليقين
            probas = [tree.predict_proba(latest_features)[0][1] 
                     for tree in self.model.estimators_]
            std_prob = np.std(probas)
            
            # معايرة الثقة
            uncertainty = std_prob * 0.4
            calibrated_prob = prob_up * (1 - uncertainty)
            calibrated_prob = np.clip(calibrated_prob, 0.25, 0.75)
        else:
            calibrated_prob = 0.5
        
        # استخراج المؤشرات الفنية
        rsi_val = round(latest['RSI'], 2)
        sma20 = latest['SMA_20']
        sma50 = latest['SMA_50']
        close_price = latest['close']
        volume_ratio = latest['volume_ratio']
        
        # منطق القرار المتقدم
        # شراء
        if (calibrated_prob > 0.58 and 
            rsi_val < 60 and 
            close_price > sma20 and 
            volume_ratio > 0.8):
            
            confidence = round(calibrated_prob * 100, 1)
            reason = (f"🚀 إشارة شراء قوية!\n"
                     f"• ثقة النموذج: {confidence}%\n"
                     f"• RSI: {rsi_val} (منطقة آمنة)\n"
                     f"• السعر فوق SMA20: {close_price:.2f} > {sma20:.2f}\n"
                     f"• حجم التداول: {volume_ratio:.2f}x المتوسط")
            return 'BUY', confidence, reason
        
        # بيع
        elif (calibrated_prob < 0.42 or 
              rsi_val > 72 or 
              (rsi_val > 65 and calibrated_prob < 0.50)):
            
            confidence = round((1 - calibrated_prob) * 100, 1)
            reason = (f"🔻 إشارة بيع!\n"
                     f"• ثقة النموذج: {confidence}%\n"
                     f"• RSI: {rsi_val}" + 
                     (" (تشبع شرائي!)" if rsi_val > 72 else "") + "\n" +
                     f"• السعر نسبة لـ SMA20: {((close_price/sma20)-1)*100:.1f}%")
            return 'SELL', confidence, reason
        
        # انتظار
        else:
            confidence = round(abs(calibrated_prob - 0.5) * 200, 1)
            reason = (f"⏸️ منطقة انتظار\n"
                     f"• RSI: {rsi_val}\n"
                     f"• احتمالية الصعود: {round(calibrated_prob*100,1)}%\n"
                     f"• السوق في حالة تذبذب")
            return 'HOLD', confidence, reason
    
    def get_feature_importance(self):
        """عرض أهمية الميزات"""
        if self.feature_importance is not None:
            print("\n📊 أهم الميزات في النموذج:")
            print(self.feature_importance.head(10))
            return self.feature_importance
        else:
            print("⚠️ النموذج غير مدرب بعد")
            return None
