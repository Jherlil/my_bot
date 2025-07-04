from utils import log

class RiskManager:
    def __init__(self, stop_loss_amount, stop_loss_consecutive, stop_win_amount, stop_win_victories,
                 strategy, martingale_factor, soros_level, use_martingale_if_high_chance,
                 use_soros_if_low_payout, min_payout_for_soros, assets):
        self.stop_loss_amount = stop_loss_amount
        self.stop_loss_consecutive = stop_loss_consecutive
        self.stop_win_amount = stop_win_amount
        self.stop_win_victories = stop_win_victories
        self.strategy = strategy
        self.martingale_factor = martingale_factor
        self.soros_level = soros_level
        self.use_martingale_if_high_chance = use_martingale_if_high_chance
        self.use_soros_if_low_payout = use_soros_if_low_payout
        self.min_payout_for_soros = min_payout_for_soros
        self.assets = {asset: {"current_amount": 1, "losses_amount": 0, "wins_amount": 0,
                               "consecutive_losses": 0, "consecutive_wins": 0} for asset in assets}

    def can_trade(self, asset):
        a = self.assets[asset]
        if a["losses_amount"] >= self.stop_loss_amount:
            log(f"[{asset}] Stop loss global atingido — perdas: {a['losses_amount']}")
            return False
        if a["consecutive_losses"] >= self.stop_loss_consecutive:
            log(f"[{asset}] Stop loss consecutivo atingido — {a['consecutive_losses']} perdas seguidas")
            return False
        if a["wins_amount"] >= self.stop_win_amount:
            log(f"[{asset}] Stop win global atingido — ganhos: {a['wins_amount']}")
            return False
        if a["consecutive_wins"] >= self.stop_win_victories:
            log(f"[{asset}] Stop win consecutivo atingido — {a['consecutive_wins']} vitórias seguidas")
            return False
        return True

    def next_amount(self, asset, high_chance=False, payout=1.0):
        a = self.assets[asset]
        amount = a["current_amount"]

        # Martingale apenas se high_chance
        if self.strategy == "martingale" and high_chance:
            amount = a["current_amount"]
        elif self.strategy == "soros" and self.use_soros_if_low_payout and payout < self.min_payout_for_soros and high_chance:
            amount = a["current_amount"]
        return amount

    def register_trade(self, asset, result):
        a = self.assets[asset]
        if result:
            a["wins_amount"] += a["current_amount"]
            a["consecutive_wins"] += 1
            a["consecutive_losses"] = 0
            a["current_amount"] = 1
        else:
            a["losses_amount"] += a["current_amount"]
            a["consecutive_losses"] += 1
            a["consecutive_wins"] = 0
            if self.strategy == "martingale":
                a["current_amount"] *= self.martingale_factor
            elif self.strategy == "soros":
                a["current_amount"] *= self.soros_level
            else:
                a["current_amount"] = 1
        log(f"[{asset}] Novo valor: {a['current_amount']} | perdas: {a['losses_amount']} | vitórias seguidas: {a['consecutive_wins']} | perdas seguidas: {a['consecutive_losses']} | ganhos totais: {a['wins_amount']}")
