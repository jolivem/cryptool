import pandas as pd

def simulate(df, usdc_per_order, buy_drop_pct, buy_pullback_pct, sell_gain_pct, sell_pullback_pct, do_log):

    fee_pct = 0.00075
    positions = []
    entry_price = None
    profit = 0
    log = []
    
    # start_time = pd.Timestamp("2025-07-06 11:30:00+0200")
    # df = df[df['timestamp'] >= start_time]

    lowest_price = None
    highest_price = None
    top_price = None 

    # Boucle principale
    for i in range(1, len(df)):
        #volume = df['volume'].iloc[i]
        # if volume == 0:
        #     continue

        price = df['close'].iloc[i]
        time = df['timestamp'].iloc[i]
        # print(f"[{time}]\n")

        # === Logique d'achat (grille) ===
        should_buy = False
        max_cov = 0

        if not positions:
            # Pas de position : premier achat
            should_buy = True
            if do_log:
                print(f"[{time}] New session -*-*-*-*-*-*-")
        else:
            # if top_price is None or price > top_price:
            #     top_price = price
            #     lowest_price = None  # reset car nouveau sommet

            if lowest_price is None or price < lowest_price:
                lowest_price = price

            # drop_from_top = 1 - price / top_price

            # if drop_from_top >= buy_drop_pct:
            #     pull_back_price = lowest_price * (1 + buy_pullback_pct)
            #     if price >= lowest_price * (1 + buy_pullback_pct):
            #         should_buy = True
            # On identifie le prix d'entrée le plus bas parmi les positions ouvertes
            lowest_entry_price = min(p['entry'] for p in positions)
            drop_from_lowest = 1.0 - price / lowest_entry_price
            # if df['close'].iloc[i] == 0.00013121:
            #     print(f"[{time}] ✅ drop_from_lowest {drop_from_lowest:.6f} lowest_entry_price {lowest_entry_price} lowest_price {lowest_price} buy_drop_pct {buy_drop_pct}")
            
            if drop_from_lowest >= buy_drop_pct:
                
                print(f"AA [{time}] lowest_entry_price:{lowest_entry_price} drop_from_lowest{drop_from_lowest:.4f}")
                # Optionnel : détecter un petit rebond après le point bas
                previous_price = df['close'].iloc[i - 1]
                pullback_price = lowest_price * (1 + buy_pullback_pct)
                print(f"BB [{time}] price {price} lowest_price {lowest_price} pullback_price {pullback_price:.6f}")
                if price < previous_price and price >= pullback_price:

                    should_buy = True

        if should_buy:
            sol_qty = usdc_per_order / price
            fee = sol_qty * price * fee_pct
            positions.append({
                'qty': sol_qty,
                'entry': price,
                'fee': fee,
                'highest': price  # suivi du plus haut après achat
            })
            log.append({'time': time, 'type': 'BUY', 'price': price, 'fee': fee})
            cov = len(positions)
            if max_cov < cov:
                max_cov = cov
            if do_log:
                print(f"[{time}] ✅ Achat à {price:.8f} USDC (qty: {sol_qty:.4f}), pos: {cov}")
            #print(f"drop_from_top:{drop_from_top:.4f} buy_drop_pct:{buy_drop_pct:.4f} price:{price:.4f} lowest_price:{lowest_price:.4f} pull_back_price:{pull_back_price:.4f}\n")
            # Reset cycle de détection
            top_price = None
            lowest_price = None
            last_pos_price = price

        # === Logique de vente (indépendante par position) ===
        to_close = []

        for pos in positions:
            if price > pos['highest']:
                pos['highest'] = price

            gain_pct = price / pos['entry'] - 1
            # if df['close'].iloc[i] == 0.00013121:
            #     print(f"[{time}] ✅ gain_pct {gain_pct:.6f} pos['entry'] {pos['entry']} sell_gain_pct {sell_gain_pct}")
            
            if gain_pct >= sell_gain_pct:
                
                if price <= pos['highest'] * (1 - sell_pullback_pct):
                    usdc_out = pos['qty'] * price
                    fee = usdc_out * fee_pct
                    net_gain = usdc_out - fee - (pos['entry'] * pos['qty']) - pos['fee']
                    profit += net_gain
                    log.append({'time': time, 'type': 'SELL', 'price': price, 'fee': fee})
                    cov = len(positions)
                    if max_cov < cov:
                        max_cov = cov

                    if do_log:
                        print(f"[{time}] 💰 Vente à {price:.8f} (gain: {net_gain:.2f} USDC, total: {profit:.2f}), pos: {cov}")
                    to_close.append(pos)

        for pos in to_close:
            positions.remove(pos)

    if do_log:
        print(f"\n🔚 Profit total simulé : {profit:.2f} USDC")
    #return profit
    return profit, max_cov

