# -*- coding: utf-8 -*-
import ccxt
import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import RobustScaler  # 改用鲁棒缩放器[1](@ref)
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, concatenate, Conv1D  # 新增卷积层[2](@ref)
from keras.optimizers import Adam
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # 新增回调函数[1,4](@ref)
from concurrent.futures import ThreadPoolExecutor  # 并行处理[6](@ref)

# ====================
# 增强型数据获取模块
# ====================
class EnhancedCryptoLoader:
    def __init__(self, retries=3):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.retries = retries
        
    def fetch_ohlcv_with_retry(self, symbol='BTC/USDT', timeframe='1h', limit=1000):
        """带重试机制的增强数据获取"""
        for _ in range(self.retries):
            try:
                data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # 扩展技术指标（新增波动率指标）[1,5](@ref)
                df['RSI'] = talib.RSI(df['close'], timeperiod=14)
                df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
                df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'])
                df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
                df['price_diff_pct'] = (df['close'] - df['open']) / df['open'] * 100  # 价格差异百分比[5](@ref)
                return df.set_index('timestamp').dropna()
            except Exception as e:
                print(f"API请求失败，剩余重试次数{self.retries-1}: {str(e)}")
        raise ConnectionError("数据获取失败")

# ====================
# 改进特征工程模块
# ==================== 
class AdvancedFeatureEngineer:
    def __init__(self, window_size=60, n_steps=5):
        self.scaler = RobustScaler()  # 改用鲁棒缩放
        self.window = window_size
        self.n_steps = n_steps  # 多步预测[1](@ref)
        
    def create_enhanced_sequences(self, data):
        """创建带噪声注入的增强序列"""
        scaled_data = self.scaler.fit_transform(data)
        
        # 数据增强：添加高斯噪声[1](@ref)
        noisy_data = scaled_data * (1 + 0.001 * np.random.randn(*scaled_data.shape))
        
        X, y = [], []
        for i in range(self.window, len(noisy_data)-self.n_steps):
            X.append(noisy_data[i-self.window:i])
            y.append(noisy_data[i:i+self.n_steps, 3])  # 预测未来n步close价格
        return np.array(X), np.array(y)

# ====================
# 混合模型架构改进
# ====================
def create_enhanced_model(input_shape):
    # 输入层
    inputs = Input(shape=input_shape)
    
    # CNN特征提取[2](@ref)
    conv = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    
    # 堆叠LSTM层[2](@ref)
    lstm = LSTM(128, return_sequences=True)(conv)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(64, return_sequences=True)(lstm)
    
    # Transformer分支改进
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(lstm, lstm)
    norm = LayerNormalization(epsilon=1e-6)(attention + lstm)
    
    # 融合层增加残差连接
    combined = concatenate([lstm, norm])
    dense = Dense(64, activation='relu')(combined)
    outputs = Dense(5)(dense)  # 多步预测输出[1](@ref)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0),  # 梯度裁剪[1](@ref)
                 loss='huber_loss',
                 metrics=['mae'])
    return model

# ====================
# 动态风险控制模块
# ====================
class DynamicRiskManager:
    def __init__(self, base_threshold=0.03):
        self.base_threshold = base_threshold
        
    def calculate_dynamic_threshold(self, volatility):
        """基于波动率的动态阈值[5](@ref)"""
        return min(0.1, max(0.02, self.base_threshold * (1 + volatility*10)))
    
    def check_max_drawdown(self, portfolio_value, max_drawdown=0.15):
        """最大回撤熔断机制[3](@ref)"""
        peak = np.max(portfolio_value)
        trough = np.min(portfolio_value)
        drawdown = (peak - trough) / peak
        return drawdown > max_drawdown

# ====================
# 并行预测引擎
# ====================
class ParallelPredictor:
    def __init__(self, model, workers=4):
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=workers)
        
    async def async_predict(self, data):
        """异步批量预测[6](@ref)"""
        futures = []
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            futures.append(self.executor.submit(self.model.predict, batch))
        return [f.result() for f in futures]

# ====================
# 主执行流程优化
# ====================
if __name__ == "__main__":
    # 数据加载与增强
    loader = EnhancedCryptoLoader()
    df = loader.fetch_ohlcv_with_retry()
    features = df[['close', 'volume', 'RSI', 'MACD', 'BB_upper', 'ATR', 'price_diff_pct']]
    
    # 特征工程优化
    engineer = AdvancedFeatureEngineer(window_size=60, n_steps=5)
    X, y = engineer.create_enhanced_sequences(features.values)
    
    # 模型训练改进
    model = create_enhanced_model(input_shape=(X.shape[1], X.shape[2]))
    callbacks = [
        EarlyStopping(monitor='val_mae', patience=10),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    ]
    model.fit(X, y, batch_size=128, epochs=100, 
             validation_split=0.2, callbacks=callbacks)
    
    # 并行预测
    latest_data = features[-60:].values
    scaled_data = engineer.scaler.transform(latest_data)
    predictor = ParallelPredictor(model)
    predictions = predictor.async_predict(np.array([scaled_data]))
    
    # 动态风险控制
    risk_manager = DynamicRiskManager()
    volatility = df['ATR'].iloc[-1] / df['close'].iloc[-1]
    threshold = risk_manager.calculate_dynamic_threshold(volatility)
    
    # 交易决策优化
    current_price = df['close'].iloc[-1]
    predicted_prices = engineer.scaler.inverse_transform(
        np.concatenate([np.zeros((5,6)), predictions[0]], axis=1))[:,3]
    print(f"当前价格: {current_price:.2f} 预测序列: {predicted_prices}")
    
    # 生成交易信号
    max_pred = np.max(predicted_prices)
    signal = 'BUY' if (max_pred - current_price)/current_price > threshold else 'HOLD'
    print(f"动态阈值: {threshold:.2%} 交易信号: {signal}")