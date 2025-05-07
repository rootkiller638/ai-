


# 加密货币量化预测与动态交易决策系统  

本项目是一个集成加密货币数据获取、增强特征工程、混合模型预测及动态风险控制的量化交易系统，采用Python实现，结合深度学习与传统量化方法，旨在对加密货币价格走势进行多步预测并生成交易信号。  

---

## 系统架构模块说明  

### 1. 增强型数据获取模块 (`EnhancedCryptoLoader`)  
- **功能**：通过Binance API获取OHLCV数据，包含重试机制确保稳定性，并扩展计算RSI、MACD、布林带（BBANDS）、平均真实波幅（ATR）、价格差异百分比等技术指标。  
- **关键特性**：  
  - 自动重试机制应对API请求失败。  
  - 实时转换时间戳并生成丰富技术分析指标。  

### 2. 改进特征工程模块 (`AdvancedFeatureEngineer`)  
- **功能**：对数据进行鲁棒缩放（Robust Scaling），创建带噪声注入的数据增强序列，支持多步预测（`n_steps`）。  
- **关键特性**：  
  - 采用鲁棒缩放器减少异常值影响。  
  - 注入高斯噪声增强模型泛化能力。  

### 3. 混合模型架构 (`create_enhanced_model`)  
- **结构**：融合CNN（特征提取）、LSTM（时序建模）与Transformer（注意力机制），支持多步输出。  
- **优化**：  
  - 梯度裁剪（`clipnorm`）防止梯度爆炸。  
  - 使用Huber损失函数提升对异常值的鲁棒性。  
  - 集成早停（`EarlyStopping`）与学习率调整（`ReduceLROnPlateau`）回调。  

### 4. 动态风险控制模块 (`DynamicRiskManager`)  
- **功能**：根据波动率（ATR）动态调整交易阈值，实现最大回撤熔断机制。  
- **关键特性**：  
  - 动态阈值公式：`min(0.1, max(0.02, base_threshold * (1 + volatility*10)))`。  
  - 实时监控投资组合最大回撤。  

### 5. 并行预测引擎 (`ParallelPredictor`)  
- **功能**：利用线程池并行处理批量预测，提升推理效率。  

---

## 安装依赖  
```bash  
pip install ccxt numpy pandas talib scikit-learn keras tensorflow  
```  

---

## 使用方法  
1. **数据获取与预处理**：  
   ```python  
   loader = EnhancedCryptoLoader()  
   df = loader.fetch_ohlcv_with_retry(symbol='BTC/USDT', timeframe='1h')  
   features = df[['close', 'volume', 'RSI', 'MACD', 'BB_upper', 'ATR', 'price_diff_pct']]  
   ```  

2. **特征工程与序列创建**：  
   ```python  
   engineer = AdvancedFeatureEngineer(window_size=60, n_steps=5)  
   X, y = engineer.create_enhanced_sequences(features.values)  
   ```  

3. **模型训练**：  
   ```python  
   model = create_enhanced_model(input_shape=(X.shape[1], X.shape[2]))  
   model.fit(X, y, batch_size=128, epochs=100, validation_split=0.2, callbacks=callbacks)  
   ```  

4. **并行预测与交易决策**：  
   ```python  
   latest_data = features[-60:].values  
   scaled_data = engineer.scaler.transform(latest_data)  
   predictor = ParallelPredictor(model)  
   predictions = predictor.async_predict(np.array([scaled_data]))  
   # 动态风险控制与信号生成  
   risk_manager = DynamicRiskManager()  
   volatility = df['ATR'].iloc[-1] / df['close'].iloc[-1]  
   threshold = risk_manager.calculate_dynamic_threshold(volatility)  
   # 生成交易信号  
   current_price = df['close'].iloc[-1]  
   predicted_prices = engineer.scaler.inverse_transform(np.concatenate([np.zeros((5,6)), predictions[0]], axis=1))[:,3]  
   signal = 'BUY' if (np.max(predicted_prices) - current_price)/current_price > threshold else 'HOLD'  
   ```  

---

## 注意事项  
- 确保Binance API网络连接稳定，可根据需求修改`fetch_ohlcv_with_retry`中的交易对（`symbol`）及时帧（`timeframe`）。  
- 调整`AdvancedFeatureEngineer`中的`window_size`和`n_steps`以适配不同时序模式。  
- 动态风险控制模块中的`base_threshold`可根据策略风险偏好调整。  

通过本系统，用户可实现从数据获取、模型训练到实时预测与风险控制的全流程量化交易支持。 
