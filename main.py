from ib_insync import *
import util

# 1. الاتصال بالمنصة
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# 2. تحديد الأصل والكمية
symbol = 'AAPL'
fixed_quantity = 10  # الكمية المحددة لكل صفقة
contract = Stock(symbol, 'SMART', 'USD')

def check_opportunity():
    # ضَع شرط الدخول هنا (مثال: تقاطع متوسطات، إشارة استراتيجية...)
    return True 

def execute_trade():
    if check_opportunity():
        # إنشاء أمر شراء بسعر السوق
        order = MarketOrder('BUY', fixed_quantity)
        trade = ib.placeOrder(contract, order)
        print(f"تم إرسال أمر شراء {fixed_quantity} سهم في {symbol}")

execute_trade()
