import time
import pandas as pd
from iqoptionapi.stable_api import IQ_Option
from utils import log, load_config
from fundamental import FundamentalAnalyzer
from technical import TechnicalAnalyzer
from risk import RiskManager
from ml_model import MLModel

def get_candles_df(IQ, asset, timeframe, num_candles):
    candles = IQ.get_candles(asset, timeframe, num_candles, time.time())
    return pd.DataFrame(candles)

def main():
    config = load_config('config.yaml')
    IQ = IQ_Option(config['email'], config['password']); IQ.connect(); IQ.change_balance(config['account_type'].lower())

    fundamental = FundamentalAnalyzer(buffer_minutes=config['news_buffer_minutes'])
    technical = TechnicalAnalyzer(
        ma_fast=config['trend_ma_fast'],
        ma_slow=config['trend_ma_slow'],
        volume_period=config['volume_period'],
    )
    risk = RiskManager(
        stop_loss_amount=config['stop_loss_amount'],
        stop_loss_consecutive=config['stop_loss_consecutive'],
        stop_win_amount=config['stop_win_amount'],
        stop_win_victories=config['stop_win_victories'],
        strategy=config['strategy'],
        martingale_factor=config['martingale_factor'],
        soros_level=config['soros_level'],
        use_martingale_if_high_chance=config['use_martingale_if_high_chance'],
        use_soros_if_low_payout=config['use_soros_if_low_payout'],
        min_payout_for_soros=config['min_payout_for_soros'],
        assets=config['assets'],
    )
    ml = MLModel()

    daily_wins = 0
    last_trade_date = None

    while True:
        log("Loop principal...")
        ml.check_and_train_daily()

        if fundamental.check_high_impact_news():
            log("Esperando — notícia importante próxima...")
            time.sleep(60)
            continue

        # Reseta win diário
        if last_trade_date is None or last_trade_date.date() < pd.Timestamp.now().date():
            daily_wins = 0
        last_trade_date = pd.Timestamp.now()

        if daily_wins >= config['stop_win_victories']:
            log("Stop win diário atingido — esperando até amanhã...")
            time.sleep(60*60)
            continue

        for asset in config['assets']:
            payout = IQ.get_profitability(asset)
            if payout < config['min_payout'] or payout > config['max_payout']:
                continue

            df = get_candles_df(IQ, asset, config['timeframe_main'], num_candles=100)
            df = technical.calculate_moving_averages(df)
            breakout = technical.detect_breakout(df)
            trend = technical.detect_trend(df)
            patterns = technical.detect_candlestick_patterns(df)
            pattern_name = patterns[0][0] if patterns else None
            last_candle = df.iloc[-1]

            # Calcular features
            avg_volume = df['volume'].rolling(config['volume_period']).mean().iloc[-1]
            volume_ratio = last_candle.volume / avg_volume
            high_chance = all([breakout, pattern_name, volume_ratio > 1.0, trend != "flat"])

            features = {
                "pattern_name": pattern_name or "unknown",
                "breakout": breakout or "none",
                "trend": trend,
                "volume_ratio": volume_ratio,
                "payout": payout,
                # incluir aqui outras features que você queira treinar
            }

            if breakout and risk.can_trade(asset):
                direction = "call" if breakout == "breakout_up" and trend == "up" else "put" if breakout == "breakout_down" and trend == "down" else None
                if direction:
                    if ml.predict_high_chance(features):
                        amount = risk.next_amount(asset, high_chance=True, payout=payout)
                        log(f"[{asset}] Entrando {direction} com {amount} — alta_chance={True}")
                        status, order_id = IQ.buy(amount, asset, direction, expiry=1)
                        result, _ = IQ.check_win(order_id)
                        risk.register_trade(asset, result)
                        ml.log_trade(features, result)

                        if result:
                            daily_wins += 1

        log("Esperando próximo ciclo...")
        time.sleep(config['timeframe_main'])

if __name__ == "__main__":
    main()
