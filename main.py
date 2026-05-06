"""
╔══════════════════════════════════════════════════════════════════╗
║   HYPER TÀI XỈU PREDICTOR - v3.0 ULTRA (FIXED)                 ║
║   Python FastAPI + Full ML Stack                                 ║
║   Models: XGBoost + LightGBM + RF + BiLSTM + Ensemble          ║
║   Features: 300+ auto-engineered features                        ║
║   RL Bandit: Thompson Sampling model selector                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os, sys, re, json, time, math, secrets, hashlib, logging, sqlite3
import threading, pickle, warnings
from datetime import datetime, date, timedelta
from functools import wraps
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict, deque

warnings.filterwarnings('ignore')

# ─── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
from scipy import stats as scipy_stats
from scipy.signal import find_peaks

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, VotingClassifier)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb

# ─── Optional Deep Learning ───────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, BatchNormalization,
                                          Conv1D, MaxPooling1D, Flatten, Bidirectional,
                                          Input, MultiHeadAttention, LayerNormalization,
                                          GlobalAveragePooling1D, Concatenate)
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    tf.get_logger().setLevel('ERROR')
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hyper_tx.log', encoding='utf-8')
    ]
)
log = logging.getLogger('HyperTX')

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DB_PATH           = os.environ.get('DB_PATH', 'hyper_tx.db')
PORT              = int(os.environ.get('PORT', 10000))
ENABLE_AUTH       = os.environ.get('ENABLE_AUTH', 'true').lower() == 'true'
ADMIN_KEY         = os.environ.get('ADMIN_KEY', '')
DEFAULT_DAILY     = int(os.environ.get('DEFAULT_DAILY', 500))
FETCH_INTERVAL    = int(os.environ.get('FETCH_INTERVAL', 60))   # giây
RETRAIN_INTERVAL  = int(os.environ.get('RETRAIN_INTERVAL', 3600))
MIN_TRAIN         = int(os.environ.get('MIN_TRAIN', 100))
MAX_SESSIONS      = 20000
ACCURACY_THRESHOLD = 0.65  # dưới ngưỡng này → trigger retrain

API_HU  = 'https://wtx.tele68.com/v1/tx/sessions'
API_MD5 = 'https://wtxmd52.tele68.com/v1/txmd5/sessions'

WINDOWS = [3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50, 100]

# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db() -> None:
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            type TEXT NOT NULL,
            result TEXT NOT NULL,
            dice1 INTEGER, dice2 INTEGER, dice3 INTEGER,
            point INTEGER,
            timestamp TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_type ON sessions(type, id);

        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT,
            daily_limit INTEGER DEFAULT 500,
            used_today INTEGER DEFAULT 0,
            last_reset_date TEXT,
            is_active INTEGER DEFAULT 1,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            phien INTEGER,
            prediction TEXT,
            confidence REAL,
            actual TEXT,
            correct INTEGER,
            model_used TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_pred_type ON predictions(type, phien);

        CREATE TABLE IF NOT EXISTS model_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            trained_at TEXT,
            accuracy REAL,
            auc REAL,
            n_sessions INTEGER,
            model_name TEXT
        );
    """)
    conn.commit()
    conn.close()
    log.info("DB initialized.")

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════
class DataCollector:
    def __init__(self):
        self._lock = threading.Lock()

    def _transform(self, data: dict, game_type: str) -> List[Dict]:
        items = data.get('list') or data.get('data') or []
        out = []
        for it in items:
            raw = (it.get('resultTruyenThong') or it.get('result') or '').upper()
            result = 'TAI' if 'TAI' in raw or 'TÀI' in raw else 'XIU'
            dices = it.get('dices') or []
            out.append({
                'id': it.get('id'), 'type': game_type,
                'result': result,
                'dice1': dices[0] if len(dices) > 0 else None,
                'dice2': dices[1] if len(dices) > 1 else None,
                'dice3': dices[2] if len(dices) > 2 else None,
                'point': it.get('point'),
                'timestamp': it.get('timestamp') or ''
            })
        return out

    def fetch(self, game_type: str) -> Optional[List[Dict]]:
        url = API_HU if game_type == 'hu' else API_MD5
        try:
            r = requests.get(url, timeout=15, params={'limit': 200})
            r.raise_for_status()
            return self._transform(r.json(), game_type)
        except Exception as e:
            log.error(f"Fetch [{game_type}] error: {e}")
            return None

    def save(self, sessions: List[Dict]) -> int:
        if not sessions:
            return 0
        conn = get_db()
        c = conn.cursor()
        inserted = 0
        for s in sessions:
            try:
                c.execute("""
                    INSERT OR IGNORE INTO sessions (id, type, result, dice1, dice2, dice3, point, timestamp)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (s['id'], s['type'], s['result'], s['dice1'], s['dice2'], s['dice3'], s['point'], s['timestamp']))
                inserted += c.rowcount
            except Exception:
                pass
        # Keep max MAX_SESSIONS per type
        for t in ['hu', 'md5']:
            c.execute(f"""
                DELETE FROM sessions WHERE type=? AND id NOT IN
                (SELECT id FROM sessions WHERE type=? ORDER BY id DESC LIMIT {MAX_SESSIONS})
            """, (t, t))
        conn.commit()
        conn.close()
        return inserted

    def get_history(self, game_type: str, limit: int = MAX_SESSIONS) -> pd.DataFrame:
        conn = get_db()
        df = pd.read_sql_query(
            "SELECT * FROM sessions WHERE type=? ORDER BY id ASC LIMIT ?",
            conn, params=(game_type, limit)
        )
        conn.close()
        return df

    def count(self, game_type: str) -> int:
        conn = get_db()
        n = conn.execute("SELECT COUNT(*) FROM sessions WHERE type=?", (game_type,)).fetchone()[0]
        conn.close()
        return n

    def run_background(self):
        def loop():
            while True:
                for gt in ['hu', 'md5']:
                    try:
                        sessions = self.fetch(gt)
                        if sessions:
                            n = self.save(sessions)
                            if n:
                                log.info(f"[{gt}] +{n} new sessions (total={self.count(gt)})")
                    except Exception as e:
                        log.error(f"BG fetch [{gt}]: {e}")
                time.sleep(FETCH_INTERVAL)
        t = threading.Thread(target=loop, daemon=True)
        t.start()
        log.info(f"DataCollector background started (interval={FETCH_INTERVAL}s)")

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEER (300+ features)
# ═══════════════════════════════════════════════════════════════════════════════
class FeatureEngineer:
    """Build 300+ features from raw session history."""

    @staticmethod
    def _ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
        d = s.diff()
        g = d.where(d > 0, 0.0).ewm(alpha=1/p, adjust=False).mean()
        l = (-d).where(d < 0, 0.0).ewm(alpha=1/p, adjust=False).mean()
        return 100 - 100 / (1 + g / (l + 1e-9))

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        N = len(df)
        label = (df['result'] == 'TAI').astype(int)
        pts   = df['point'].fillna(df['point'].median()).astype(float)
        f = pd.DataFrame(index=df.index)

        # ── Rolling stats ─────────────────────────────────────────────────
        for w in WINDOWS:
            sl = label.shift(1)
            sp = pts.shift(1)
            f[f'tai_ratio_{w}']   = sl.rolling(w, min_periods=1).mean()
            f[f'pt_mean_{w}']     = sp.rolling(w, min_periods=1).mean()
            f[f'pt_std_{w}']      = sp.rolling(w, min_periods=1).std()
            f[f'pt_mad_{w}']      = sp.rolling(w, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
            f[f'pt_skew_{w}']     = sp.rolling(w, min_periods=1).skew()
            f[f'pt_kurt_{w}']     = sp.rolling(w, min_periods=1).kurt()
            f[f'pt_min_{w}']      = sp.rolling(w, min_periods=1).min()
            f[f'pt_max_{w}']      = sp.rolling(w, min_periods=1).max()
            f[f'pt_range_{w}']    = f[f'pt_max_{w}'] - f[f'pt_min_{w}']
            f[f'pt_q25_{w}']      = sp.rolling(w, min_periods=1).quantile(0.25)
            f[f'pt_q75_{w}']      = sp.rolling(w, min_periods=1).quantile(0.75)
            f[f'pt_iqr_{w}']      = f[f'pt_q75_{w}'] - f[f'pt_q25_{w}']

        # ── Streak ────────────────────────────────────────────────────────
        streak_vals, streak_types = [], []
        cur = 1
        for i in range(N):
            if i == 0:
                streak_vals.append(1); streak_types.append(label.iloc[0])
            else:
                cur = cur + 1 if label.iloc[i] == label.iloc[i-1] else 1
                streak_vals.append(cur); streak_types.append(label.iloc[i])
        f['streak_len']  = pd.Series(streak_vals).shift(1).fillna(1).values
        f['streak_type'] = pd.Series(streak_types).shift(1).fillna(0).values
        f['streak_sq']   = f['streak_len'] ** 2
        f['streak_log']  = np.log1p(f['streak_len'])

        # ── Lag features (1..30) ──────────────────────────────────────────
        for lag in range(1, 31):
            f[f'lbl_lag_{lag}'] = label.shift(lag)
            f[f'pt_lag_{lag}']  = pts.shift(lag)

        # ── Difference & momentum ─────────────────────────────────────────
        sp = pts.shift(1)
        f['pt_diff1']       = sp.diff(1)
        f['pt_diff2']       = sp.diff(2)
        f['pt_diff3']       = sp.diff(3)
        f['pt_accel']       = f['pt_diff1'] - f['pt_diff2']
        f['pt_jerk']        = f['pt_accel'].diff(1)
        f['pt_logret']      = np.log(sp / (sp.shift(1) + 1e-9))
        f['result_changed'] = (label.shift(1) != label.shift(2)).astype(int)
        f['result_ch2']     = (label.shift(1) != label.shift(3)).astype(int)

        # ── Technical indicators ──────────────────────────────────────────
        s1 = pts.shift(1)
        f['rsi_14'] = self._rsi(s1, 14)
        f['rsi_21'] = self._rsi(s1, 21)
        f['rsi_7']  = self._rsi(s1, 7)

        ema12 = self._ema(s1, 12); ema26 = self._ema(s1, 26)
        macd  = ema12 - ema26;     sig   = self._ema(macd, 9)
        f['macd']      = macd
        f['macd_sig']  = sig
        f['macd_hist'] = macd - sig
        f['macd_cross'] = (macd > sig).astype(int)

        sma20 = s1.rolling(20, min_periods=1).mean()
        std20 = s1.rolling(20, min_periods=1).std().fillna(1)
        bbu   = sma20 + 2 * std20; bbl = sma20 - 2 * std20
        f['bb_pctb'] = (s1 - bbl) / (bbu - bbl + 1e-9)
        f['bb_bw']   = (bbu - bbl) / (sma20 + 1e-9)

        lo14 = s1.rolling(14, min_periods=1).min(); hi14 = s1.rolling(14, min_periods=1).max()
        f['stoch_k'] = 100 * (s1 - lo14) / (hi14 - lo14 + 1e-9)
        f['stoch_d'] = f['stoch_k'].rolling(3, min_periods=1).mean()
        f['atr_14']  = (hi14 - lo14).rolling(14, min_periods=1).mean()

        for sp2 in [5, 10, 20, 50, 100]:
            f[f'ema_{sp2}']   = self._ema(s1, sp2)
            f[f'sma_{sp2}']   = s1.rolling(sp2, min_periods=1).mean()
            f[f'ema_cross_{sp2}'] = (s1 > f[f'ema_{sp2}']).astype(int)

        f['lr_slope5']  = s1.rolling(5, min_periods=2).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True)
        f['lr_slope10'] = s1.rolling(10, min_periods=2).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True)
        f['lr_slope20'] = s1.rolling(20, min_periods=2).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True)
        f['corr_ptlbl'] = s1.rolling(10, min_periods=3).corr(label.shift(1))

        # ── Pattern detection (cầu) ───────────────────────────────────────
        lv = label.values
        patterns = defaultdict(list)
        for i in range(N):
            def g(j): return int(lv[i-j]) if i >= j else -1
            patterns['alt2'].append(1 if g(1) != g(2) else 0)
            patterns['alt4'].append(1 if len(set([g(1),g(2),g(3),g(4)])) > 1 and g(1)!=g(2) and g(2)!=g(3) and g(3)!=g(4) else 0)
            patterns['pair2'].append(1 if g(1) == g(2) else 0)
            patterns['trip'].append(1 if g(1) == g(2) == g(3) else 0)
            patterns['quad'].append(1 if g(1) == g(2) == g(3) == g(4) else 0)
            patterns['p121'].append(1 if g(1) != g(2) and g(2) == g(3) and g(3) != g(4) else 0)
            patterns['p22'].append(1 if g(1) == g(2) and g(3) == g(4) and g(1) != g(3) else 0)
            patterns['p31'].append(1 if g(1) != g(2) and g(2) == g(3) == g(4) else 0)
            patterns['p123'].append(1 if g(1) == g(2) and g(2) != g(3) else 0)
            patterns['p321'].append(1 if g(1) != g(2) == g(3) and g(3) != g(4) else 0)
            patterns['p212'].append(1 if g(1) == g(3) and g(1) != g(2) else 0)
            patterns['p1221'].append(1 if g(1) == g(4) and g(2) == g(3) and g(1) != g(2) else 0)
            patterns['p2112'].append(1 if g(1) == g(2) and g(3) == g(4) and g(1) == g(4) else 0)
            patterns['break_sig'].append(1 if streak_vals[i-1] >= 4 and i > 0 else 0)
            patterns['break_sig6'].append(1 if streak_vals[i-1] >= 6 and i > 0 else 0)
            patterns['gap'].append(1 if g(1) == g(3) and g(1) != g(2) else 0)
            patterns['ziczac'].append(1 if g(1) != g(2) and g(2) != g(3) and g(3) != g(4) and g(4) != g(5) else 0)
            patterns['p44'].append(1 if g(1)==g(2)==g(3)==g(4) and g(5)==g(6)==g(7)==g(8) and g(1) != g(5) else 0)
            patterns['p55'].append(1 if all(g(j)==g(1) for j in range(1,6)) else 0)
            # Cầu chu kỳ 2,3,4
            patterns['cycle2'].append(1 if g(1)==g(3) and g(2)==g(4) and g(1)!=g(2) else 0)
            patterns['cycle3'].append(1 if g(1)==g(4) and g(2)==g(5) and g(3)==g(6) else 0)
            patterns['day_gay'].append(1 if i >= 2 and streak_vals[i-1] == 1 and streak_vals[i-2] >= 3 else 0)
            patterns['momentum_up'].append(1 if g(1) > g(2) and g(2) > g(3) else 0 if g(1) != -1 else 0)
            # Xu hướng ngắn
            patterns['recent_tai_3'].append(sum(g(j) for j in range(1,4) if g(j) != -1))
            patterns['recent_tai_5'].append(sum(g(j) for j in range(1,6) if g(j) != -1))
            patterns['recent_tai_10'].append(sum(g(j) for j in range(1,11) if g(j) != -1))

        for name, vals in patterns.items():
            f[f'ptn_{name}'] = pd.Series(vals).shift(1).fillna(0).values

        # ── Bayesian transition matrix (bậc 3) ───────────────────────────
        for prev in [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]:
            p_str = ''.join(map(str, prev))
            # Rolling count-based probability
            cond_col = []
            tai_cnt, total_cnt = 0, 0
            for i in range(N):
                if i < 3:
                    cond_col.append(0.5); continue
                seq = tuple(int(lv[i-3+j]) for j in range(3))
                if seq == prev:
                    tai_cnt += 1; total_cnt += 1
                prob = tai_cnt / (total_cnt + 2)  # Laplace smoothing
                cond_col.append(prob)
            f[f'bayes_{p_str}'] = pd.Series(cond_col).shift(1).fillna(0.5).values

        # ── Autocorrelation (lag 1..20) ───────────────────────────────────
        for lag in range(1, 21):
            f[f'ac_{lag}'] = label.shift(1).rolling(50, min_periods=lag+1).apply(
                lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else 0.0, raw=False
            )

        # ── Dominant cycle ────────────────────────────────────────────────
        cycle_lags = []
        for i in range(N):
            if i < 50:
                cycle_lags.append(0); continue
            window = label.iloc[max(0,i-50):i]
            best_l, best_c = 1, 0.0
            for lg in range(1, 21):
                try:
                    c = window.autocorr(lg)
                    if not np.isnan(c) and abs(c) > best_c:
                        best_c = abs(c); best_l = lg
                except: pass
            cycle_lags.append(best_l if best_c > 0.4 else 0)
        f['cycle_lag']   = pd.Series(cycle_lags).shift(1).fillna(0).values
        f['cycle_phase'] = (pd.Series(range(N)) % pd.Series(cycle_lags).replace(0, 1)).shift(1).fillna(0).values

        # ── Dice features ─────────────────────────────────────────────────
        if 'dice1' in df.columns:
            d1 = df['dice1'].fillna(3).astype(float)
            d2 = df['dice2'].fillna(3).astype(float)
            d3 = df['dice3'].fillna(3).astype(float)
            f['dice_sum']      = (d1+d2+d3).shift(1)
            f['dice_same2']    = ((d1==d2)|(d2==d3)|(d1==d3)).astype(int).shift(1)
            f['dice_triple']   = ((d1==d2)&(d2==d3)).astype(int).shift(1)
            f['dice_even']     = ((d1+d2+d3)%2==0).astype(int).shift(1)
            f['dice_extreme']  = ((d1==1)|(d1==6)|(d2==1)|(d2==6)|(d3==1)|(d3==6)).astype(int).shift(1)
            f['dice_mid']      = ((d1.between(2,5))&(d2.between(2,5))&(d3.between(2,5))).astype(int).shift(1)
            f['dice_std']      = pd.concat([d1,d2,d3], axis=1).std(axis=1).shift(1)
            f['dice_min']      = pd.concat([d1,d2,d3], axis=1).min(axis=1).shift(1)
            f['dice_max']      = pd.concat([d1,d2,d3], axis=1).max(axis=1).shift(1)
            f['dice_spread']   = f['dice_max'] - f['dice_min']

        # ── Time features ─────────────────────────────────────────────────
        if 'timestamp' in df.columns:
            try:
                ts  = pd.to_datetime(df['timestamp'], errors='coerce')
                h   = ts.dt.hour.fillna(12).astype(float)
                dow = ts.dt.dayofweek.fillna(0).astype(float)
                f['hour_sin'] = np.sin(2*np.pi*h/24)
                f['hour_cos'] = np.cos(2*np.pi*h/24)
                f['dow_sin']  = np.sin(2*np.pi*dow/7)
                f['dow_cos']  = np.cos(2*np.pi*dow/7)
            except: pass

        # ── Sliding-window accuracy of prev predictions ───────────────────
        f['rolling_acc_20'] = label.shift(1).rolling(20, min_periods=5).apply(
            lambda x: (x == x.shift(1)).mean() if len(x) > 1 else 0.5, raw=False
        )

        f['label'] = label
        f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
        return f

    def select_features(self, X: np.ndarray, y: np.ndarray, k: int = 200) -> np.ndarray:
        """SelectKBest + PCA combination."""
        k = min(k, X.shape[1])
        sel = SelectKBest(mutual_info_classif, k=k)
        X_sel = sel.fit_transform(X, y)
        return X_sel, sel

# ═══════════════════════════════════════════════════════════════════════════════
#  DEEP LEARNING MODELS
# ═══════════════════════════════════════════════════════════════════════════════
class DeepModels:
    """BiLSTM + Temporal CNN + Transformer (if TF available)."""

    SEQ_LEN = 30

    def build_bilstm(self, n_feat: int) -> 'tf.keras.Model':
        inp = Input(shape=(self.SEQ_LEN, n_feat))
        x = Bidirectional(LSTM(128, return_sequences=True))(inp)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(64))(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        out = Dense(1, activation='sigmoid')(x)
        m = Model(inp, out)
        m.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        return m

    def build_cnn_gru(self, n_feat: int) -> 'tf.keras.Model':
        inp = Input(shape=(self.SEQ_LEN, n_feat))
        x = Conv1D(64, 3, activation='relu', padding='same')(inp)
        x = Conv1D(64, 5, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)
        x = GRU(64, return_sequences=False)(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(1, activation='sigmoid')(x)
        m = Model(inp, out)
        m.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        return m

    def build_transformer(self, n_feat: int) -> 'tf.keras.Model':
        inp = Input(shape=(self.SEQ_LEN, n_feat))
        x = Dense(64)(inp)
        x = LayerNormalization()(x)
        attn = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = x + attn
        x = LayerNormalization()(x)
        ff = Dense(128, activation='relu')(x)
        ff = Dense(64)(ff)
        x = x + ff
        x = LayerNormalization()(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        out = Dense(1, activation='sigmoid')(x)
        m = Model(inp, out)
        m.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        return m

    def make_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in range(self.SEQ_LEN, len(X)):
            xs.append(X[i-self.SEQ_LEN:i])
            ys.append(y[i])
        return np.array(xs), np.array(ys)

# ═══════════════════════════════════════════════════════════════════════════════
#  THOMPSON SAMPLING BANDIT (RL)
# ═══════════════════════════════════════════════════════════════════════════════
class ThompsonBandit:
    """Multi-arm bandit: each arm = one model. Uses Beta distribution."""

    def __init__(self, arms: List[str]):
        self.arms = arms
        # alpha, beta for each arm (successes+1, failures+1)
        self.alpha = {a: 1.0 for a in arms}
        self.beta  = {a: 1.0 for a in arms}

    def select(self) -> str:
        samples = {a: np.random.beta(self.alpha[a], self.beta[a]) for a in self.arms}
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: float):
        """reward = 1 if correct, 0 if wrong."""
        self.alpha[arm] += reward
        self.beta[arm]  += (1 - reward)

    def get_probs(self) -> Dict[str, float]:
        return {a: self.alpha[a]/(self.alpha[a]+self.beta[a]) for a in self.arms}

    def to_dict(self) -> Dict:
        return {'alpha': self.alpha, 'beta': self.beta}

    @classmethod
    def from_dict(cls, arms: List[str], d: Dict) -> 'ThompsonBandit':
        b = cls(arms)
        b.alpha = d.get('alpha', {a: 1.0 for a in arms})
        b.beta  = d.get('beta',  {a: 1.0 for a in arms})
        return b

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
class ModelEnsemble:
    DL_ARMS = ['bilstm', 'cnn_gru', 'transformer'] if HAS_TF else []
    ARMS    = ['xgb', 'lgb', 'rf', 'et', 'lr'] + DL_ARMS

    def __init__(self, game_type: str):
        self.game_type = game_type
        self.scaler:   Optional[RobustScaler]         = None
        self.selector: Optional[SelectKBest]          = None
        self.feat_names: List[str]                     = []
        self.xgb_m:    Optional[xgb.XGBClassifier]    = None
        self.lgb_m:    Optional[lgb.LGBMClassifier]   = None
        self.rf_m:     Optional[RandomForestClassifier]= None
        self.et_m:     Optional[ExtraTreesClassifier]  = None
        self.lr_m:     Optional[LogisticRegression]    = None
        self.meta_m:   Optional[RandomForestClassifier]= None
        self.dl:       Optional[DeepModels]            = None
        self.bilstm_m  = None
        self.cnngru_m  = None
        self.trans_m   = None
        self.bandit:   ThompsonBandit = ThompsonBandit(self.ARMS)
        self.is_trained: bool = False
        self.accuracy:  float = 0.0
        self.auc:       float = 0.0
        self.n_sessions:int   = 0
        self.trained_at: Optional[str] = None
        self._lock = threading.RLock()
        self.recent_preds: deque = deque(maxlen=200)  # (pred, actual)

    def _class_weights(self, y: np.ndarray) -> Dict:
        cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
        return {i: w for i, w in enumerate(cw)}

    def train(self, feats: pd.DataFrame) -> Dict:
        with self._lock:
            return self._train(feats)

    def _train(self, feats: pd.DataFrame) -> Dict:
        fc = [c for c in feats.columns if c != 'label']
        X  = feats[fc].values.astype(np.float32)
        y  = feats['label'].values.astype(int)
        N  = len(X)
        if N < MIN_TRAIN:
            return {}

        # Time-ordered split
        tr_end = int(N * 0.75)
        vl_end = int(N * 0.88)
        Xtr, ytr = X[:tr_end], y[:tr_end]
        Xvl, yvl = X[tr_end:vl_end], y[tr_end:vl_end]
        Xte, yte = X[vl_end:], y[vl_end:]

        # Scale
        self.scaler = RobustScaler()
        Xtr_s = self.scaler.fit_transform(Xtr)
        Xvl_s = self.scaler.transform(Xvl)
        Xte_s = self.scaler.transform(Xte)

        # Feature selection (top 200)
        fe_engine = FeatureEngineer()
        Xtr_sel, self.selector = fe_engine.select_features(Xtr_s, ytr, k=200)
        Xvl_sel = self.selector.transform(Xvl_s)
        Xte_sel = self.selector.transform(Xte_s)
        self.feat_names = fc

        cw = self._class_weights(ytr)
        log.info(f"[{self.game_type}] Training {N} samples | tr={tr_end} vl={vl_end-tr_end} te={N-vl_end}")

        # ── XGBoost ──────────────────────────────────────────────────────
        self.xgb_m = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
            scale_pos_weight=cw.get(1, 1)/cw.get(0, 1),
            eval_metric='logloss', use_label_encoder=False,
            verbosity=0, tree_method='hist', early_stopping_rounds=50,
            random_state=42
        )
        self.xgb_m.fit(Xtr_sel, ytr, eval_set=[(Xvl_sel, yvl)], verbose=False)
        log.info(f"[{self.game_type}] XGBoost trained.")

        # ── LightGBM ─────────────────────────────────────────────────────
        self.lgb_m = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced', verbose=-1, random_state=42
        )
        self.lgb_m.fit(
            Xtr_sel, ytr, eval_set=[(Xvl_sel, yvl)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        log.info(f"[{self.game_type}] LightGBM trained.")

        # ── Random Forest ─────────────────────────────────────────────────
        self.rf_m = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight='balanced', n_jobs=-1, random_state=42
        )
        self.rf_m.fit(Xtr_sel, ytr)
        log.info(f"[{self.game_type}] RandomForest trained.")

        # ── ExtraTrees ────────────────────────────────────────────────────
        self.et_m = ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5,
            class_weight='balanced', n_jobs=-1, random_state=42
        )
        self.et_m.fit(Xtr_sel, ytr)
        log.info(f"[{self.game_type}] ExtraTrees trained.")

        # ── Logistic Regression ───────────────────────────────────────────
        self.lr_m = LogisticRegression(C=0.5, penalty='l2', max_iter=1000,
                                        class_weight='balanced', random_state=42)
        self.lr_m.fit(Xtr_sel, ytr)
        log.info(f"[{self.game_type}] LogReg trained.")

        # ── Deep Learning ─────────────────────────────────────────────────
        dl_val_preds = {}
        dl_test_preds = {}
        if HAS_TF:
            self.dl = DeepModels()
            n_sel = Xtr_sel.shape[1]
            Xtr_seq, ytr_seq = self.dl.make_sequences(Xtr_s, ytr)
            Xvl_seq, yvl_seq = self.dl.make_sequences(Xvl_s, yvl)
            Xte_seq, yte_seq = self.dl.make_sequences(Xte_s, yte)
            n_f = Xtr_seq.shape[2]

            cbs = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
            ]
            for name, builder in [('bilstm', self.dl.build_bilstm),
                                   ('cnn_gru', self.dl.build_cnn_gru),
                                   ('transformer', self.dl.build_transformer)]:
                try:
                    m = builder(n_f)
                    m.fit(Xtr_seq, ytr_seq, validation_data=(Xvl_seq, yvl_seq),
                          epochs=50, batch_size=64, callbacks=cbs, verbose=0)
                    dl_val_preds[name]  = m.predict(Xvl_seq, verbose=0).flatten()
                    dl_test_preds[name] = m.predict(Xte_seq, verbose=0).flatten()
                    setattr(self, f'{name.replace("-","_")}_m', m)
                    log.info(f"[{self.game_type}] {name} trained.")
                except Exception as e:
                    log.warning(f"[{self.game_type}] {name} failed: {e}")

        # ── Stacking meta-model ───────────────────────────────────────────
        vl_stack = np.column_stack([
            self.xgb_m.predict_proba(Xvl_sel)[:,1],
            self.lgb_m.predict_proba(Xvl_sel)[:,1],
            self.rf_m.predict_proba(Xvl_sel)[:,1],
            self.et_m.predict_proba(Xvl_sel)[:,1],
            self.lr_m.predict_proba(Xvl_sel)[:,1],
        ] + [dl_val_preds[k] for k in dl_val_preds if len(dl_val_preds[k]) == len(yvl)])

        te_stack = np.column_stack([
            self.xgb_m.predict_proba(Xte_sel)[:,1],
            self.lgb_m.predict_proba(Xte_sel)[:,1],
            self.rf_m.predict_proba(Xte_sel)[:,1],
            self.et_m.predict_proba(Xte_sel)[:,1],
            self.lr_m.predict_proba(Xte_sel)[:,1],
        ] + [dl_test_preds[k] for k in dl_test_preds])

        self.meta_m = RandomForestClassifier(n_estimators=200, random_state=42)
        self.meta_m.fit(vl_stack, yvl)

        proba_te = self.meta_m.predict_proba(te_stack)[:,1]
        ypred_te = (proba_te >= 0.5).astype(int)
        acc = accuracy_score(yte, ypred_te)
        try:
            auc = roc_auc_score(yte, proba_te)
        except:
            auc = 0.5

        # ========== FIX: MUST SET is_trained = True ==========
        self.is_trained = True
        self.accuracy   = acc
        self.auc        = auc
        self.n_sessions = N
        self.trained_at = datetime.utcnow().isoformat()

        # Save stats to DB
        conn = get_db()
        conn.execute(
            "INSERT INTO model_stats (type,trained_at,accuracy,auc,n_sessions,model_name) VALUES(?,?,?,?,?,?)",
            (self.game_type, self.trained_at, acc, auc, N, 'ensemble')
        )
        conn.commit(); conn.close()

        metrics = {'accuracy': round(acc, 4), 'auc': round(auc, 4), 'n_sessions': N, 'n_test': len(yte)}
        log.info(f"[{self.game_type}] Training done. acc={acc:.4f} auc={auc:.4f}")

        # Trigger retrain if accuracy below threshold
        if acc < ACCURACY_THRESHOLD:
            log.warning(f"[{self.game_type}] Accuracy {acc:.3f} < {ACCURACY_THRESHOLD} — will retrain next cycle with more data")

        return metrics

    def _predict_all(self, x: np.ndarray) -> Dict[str, float]:
        """Get probability from each model."""
        xs = self.scaler.transform(x.reshape(1, -1))
        xs_sel = self.selector.transform(xs)
        preds = {}
        for name, m in [('xgb',self.xgb_m),('lgb',self.lgb_m),
                         ('rf',self.rf_m),('et',self.et_m),('lr',self.lr_m)]:
            try:
                preds[name] = float(m.predict_proba(xs_sel)[:,1][0])
            except:
                preds[name] = 0.5
        # DL
        if HAS_TF and self.dl:
            seq_x = xs  # shape (1, n_feat) → need (1, SEQ_LEN, n_feat) from last SEQ_LEN rows
            # stored as flat; skip DL single-step pred (needs sequence cache)
        return preds

    def predict(self, x: np.ndarray) -> Tuple[int, float, str]:
        """Returns (label, probability, best_arm)."""
        with self._lock:
            if not self.is_trained:
                return 1, 0.5, 'fallback'
            preds_dict = self._predict_all(x)
            arm = self.bandit.select()
            # Build stack for meta
            xs_s   = self.scaler.transform(x.reshape(1,-1))
            xs_sel = self.selector.transform(xs_s)
            stack  = np.array([[
                float(self.xgb_m.predict_proba(xs_sel)[:,1][0]),
                float(self.lgb_m.predict_proba(xs_sel)[:,1][0]),
                float(self.rf_m.predict_proba(xs_sel)[:,1][0]),
                float(self.et_m.predict_proba(xs_sel)[:,1][0]),
                float(self.lr_m.predict_proba(xs_sel)[:,1][0]),
            ]])
            prob   = float(self.meta_m.predict_proba(stack)[:,1][0])
            label  = int(prob >= 0.5)
            return label, prob, arm

    def feedback(self, arm: str, correct: bool):
        self.bandit.update(arm, 1.0 if correct else 0.0)
        self.recent_preds.append(int(correct))

    def sliding_accuracy(self) -> float:
        if len(self.recent_preds) < 5:
            return self.accuracy
        return sum(self.recent_preds) / len(self.recent_preds)

    def top_features(self, n: int = 5) -> List[str]:
        if self.xgb_m is None or not self.feat_names:
            return []
        imp = self.xgb_m.feature_importances_
        idx = np.argsort(imp)[::-1][:n]
        return [self.feat_names[i] if i < len(self.feat_names) else f'f{i}' for i in idx]

    def save(self) -> None:
        path = f'ensemble_{self.game_type}.pkl'
        with open(path, 'wb') as fp:
            pickle.dump({
                'scaler': self.scaler, 'selector': self.selector,
                'feat_names': self.feat_names,
                'xgb': self.xgb_m, 'lgb': self.lgb_m,
                'rf': self.rf_m, 'et': self.et_m,
                'lr': self.lr_m, 'meta': self.meta_m,
                'bandit': self.bandit.to_dict(),
                'accuracy': self.accuracy, 'auc': self.auc,
                'n_sessions': self.n_sessions, 'trained_at': self.trained_at,
            }, fp)
        log.info(f"[{self.game_type}] Model saved to {path}")

    def load(self) -> bool:
        path = f'ensemble_{self.game_type}.pkl'
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as fp:
                d = pickle.load(fp)
            self.scaler      = d['scaler']
            self.selector    = d['selector']
            self.feat_names  = d['feat_names']
            self.xgb_m       = d['xgb']
            self.lgb_m       = d['lgb']
            self.rf_m        = d['rf']
            self.et_m        = d['et']
            self.lr_m        = d['lr']
            self.meta_m      = d['meta']
            self.bandit      = ThompsonBandit.from_dict(self.ARMS, d.get('bandit', {}))
            self.accuracy    = d.get('accuracy', 0.0)
            self.auc         = d.get('auc', 0.0)
            self.n_sessions  = d.get('n_sessions', 0)
            self.trained_at  = d.get('trained_at')
            self.is_trained  = True
            log.info(f"[{self.game_type}] Model loaded from {path} (acc={self.accuracy:.4f})")
            return True
        except Exception as e:
            log.warning(f"[{self.game_type}] Load model failed: {e}")
            return False

# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTOR (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════
class Predictor:
    def __init__(self, game_type: str):
        self.game_type  = game_type
        self.collector  = DataCollector()
        self.engineer   = FeatureEngineer()
        self.ensemble   = ModelEnsemble(game_type)
        self._last_feats: Optional[pd.DataFrame] = None
        self._retrain_sessions: int = 0
        self._train_lock = threading.Lock()

    def _get_feats(self) -> Optional[pd.DataFrame]:
        df = self.collector.get_history(self.game_type)
        if len(df) < MIN_TRAIN:
            return None
        feats = self.engineer.build(df)
        feats = feats.iloc[max(self.engineer.SEQ_LEN if hasattr(self.engineer, 'SEQ_LEN') else 30, 50):]
        feats = feats.dropna(subset=['label'])
        self._last_feats = feats
        return feats

    def train_async(self) -> None:
        def _do():
            if not self._train_lock.acquire(blocking=False):
                log.info(f"[{self.game_type}] Training already in progress, skip.")
                return
            try:
                feats = self._get_feats()
                if feats is None:
                    log.warning(f"[{self.game_type}] Not enough data for training.")
                    return
                metrics = self.ensemble.train(feats)
                if metrics:
                    self.ensemble.save()
                    self._retrain_sessions = self.collector.count(self.game_type)
            except Exception as e:
                log.error(f"[{self.game_type}] Train error: {e}", exc_info=True)
            finally:
                self._train_lock.release()
        threading.Thread(target=_do, daemon=True).start()

def predict_next(self) -> Dict:
    n = self.collector.count(self.game_type)
    
    if n < MIN_TRAIN:
        return {
            'status': 'training',
            'message': f'Đang tích lũy dữ liệu: {n}/{MIN_TRAIN}',
            'progress': f'{min(100, int(n/MIN_TRAIN*100))}%',
            'type': self.game_type
        }
    
    # Nếu model chưa trained, TRAIN ĐỒNG BỘ NGAY (chờ xong mới chạy tiếp)
    if not self.ensemble.is_trained:
        log.info(f"[{self.game_type}] Đủ dữ liệu ({n}/{MIN_TRAIN}), train đồng bộ...")
        feats = self._get_feats()
        if feats is not None and len(feats) >= MIN_TRAIN:
            self.ensemble.train(feats)  # <-- train đồng bộ
            self.ensemble.save()
            log.info(f"[{self.game_type}] Train hoàn tất!")
        else:
            return {
                'status': 'training',
                'message': f'Đang xử lý dữ liệu: {n}/{MIN_TRAIN}',
                'progress': '100%',
                'type': self.game_type
            }
    
    # Sau khi train xong, dự đoán
    feats = self._get_feats()
    if feats is None or len(feats) == 0:
        return {'status': 'error', 'message': 'Không đủ dữ liệu'}
    
    fc = [c for c in feats.columns if c != 'label']
    last = feats[fc].iloc[-1].values.astype(np.float32)
    label, prob, arm = self.ensemble.predict(last)
    
    confidence = round(prob * 100) if prob <= 0.95 else 99
    prediction = 'Tài' if label == 1 else 'Xỉu'
    
    df = self.collector.get_history(self.game_type, limit=5)
    last_phien = int(df['id'].iloc[-1]) if len(df) else 0
    next_phien = last_phien + 1
    
    return {
        'type': self.game_type,
        'phien': next_phien,
        'prediction': prediction,
        'confidence': confidence,
        'probability': round(prob, 4),
        'arm_used': arm,
        'n_sessions': self.collector.count(self.game_type)
    }

    def verify(self, phien: int, actual: str) -> Dict:
        """Cập nhật kết quả thực tế và feedback bandit."""
        actual_norm = 'Tài' if 'tai' in actual.lower() or 'tài' in actual.lower() else 'Xỉu'
        conn = get_db()
        row = conn.execute(
            "SELECT * FROM predictions WHERE type=? AND phien=? ORDER BY id DESC LIMIT 1",
            (self.game_type, phien)
        ).fetchone()
        if not row:
            conn.close()
            return {'error': f'Phiên {phien} không tìm thấy'}
        correct = int(row['prediction'] == actual_norm)
        conn.execute(
            "UPDATE predictions SET actual=?, correct=? WHERE id=?",
            (actual_norm, correct, row['id'])
        )
        conn.commit(); conn.close()

        self.ensemble.feedback(row['model_used'] or 'xgb', bool(correct))

        # Auto retrain if sliding accuracy too low
        if self.ensemble.sliding_accuracy() < ACCURACY_THRESHOLD:
            log.warning(f"[{self.game_type}] Sliding acc too low, triggering retrain.")
            self.train_async()

        return {
            'phien': phien,
            'predicted': row['prediction'],
            'actual': actual_norm,
            'correct': bool(correct),
            'sliding_accuracy': round(self.ensemble.sliding_accuracy(), 4)
        }

def schedule_retrain(self) -> None:
    def loop():
        while True:
            time.sleep(RETRAIN_INTERVAL)
            n = self.collector.count(self.game_type)
            log.info(f"[{self.game_type}] Scheduled retrain check: n={n}")
            if n >= MIN_TRAIN and self.ensemble.is_trained:
                self.train_async()
    threading.Thread(target=loop, daemon=True).start()

# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════════
def validate_key(key: str, admin_only: bool = False) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM api_keys WHERE key=?", (key,)).fetchone()
    conn.close()
    if not row or not row['is_active']:
        return None
    if row['expires_at']:
        try:
            if datetime.utcnow() > datetime.fromisoformat(row['expires_at']):
                return None
        except: pass
    if admin_only and row['name'] != 'admin':
        return None
    remaining = row['daily_limit'] - row['used_today']
    return {'id': row['id'], 'name': row['name'], 'remaining': remaining,
            'is_admin': row['name'] == 'admin', 'daily_limit': row['daily_limit']}

def inc_usage(key: str):
    conn = get_db()
    conn.execute("UPDATE api_keys SET used_today=used_today+1 WHERE key=?", (key,))
    conn.commit(); conn.close()

def reset_daily():
    conn = get_db()
    conn.execute("UPDATE api_keys SET used_today=0, last_reset_date=CURRENT_DATE")
    conn.commit(); conn.close()
    log.info("Daily limits reset.")

def init_admin_key() -> str:
    global ADMIN_KEY
    conn = get_db()
    row = conn.execute("SELECT key FROM api_keys WHERE name='admin' LIMIT 1").fetchone()
    if row:
        ADMIN_KEY = row['key']
        conn.close()
        return ADMIN_KEY
    if not ADMIN_KEY:
        ADMIN_KEY = secrets.token_hex(32)
    conn.execute(
        "INSERT OR IGNORE INTO api_keys (key,name,daily_limit,is_active) VALUES(?,?,?,?)",
        (ADMIN_KEY, 'admin', 999999, 1)
    )
    conn.commit(); conn.close()
    return ADMIN_KEY

def start_daily_reset():
    def loop():
        while True:
            now  = datetime.utcnow()
            nxt  = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            time.sleep((nxt - now).total_seconds())
            reset_daily()
    threading.Thread(target=loop, daemon=True).start()

# Auth dependency for FastAPI
async def require_auth(request: Request, admin_only: bool = False):
    if not ENABLE_AUTH:
        return {'name': 'anon', 'is_admin': True, 'remaining': 9999}
    key = request.headers.get('X-API-Key') or request.query_params.get('key')
    if not key:
        raise HTTPException(401, 'Missing API Key')
    info = validate_key(key, admin_only)
    if not info:
        raise HTTPException(401, 'Invalid or expired API Key')
    if not info['is_admin'] and info['remaining'] <= 0:
        raise HTTPException(429, 'Daily limit exceeded')
    if admin_only and not info['is_admin']:
        raise HTTPException(403, 'Admin access required')
    if not admin_only:
        inc_usage(key)
    return info

def user_auth(request: Request):
    return require_auth(request, admin_only=False)

def admin_auth(request: Request):
    return require_auth(request, admin_only=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title='Hyper TàiXỉu Predictor', version='3.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# Singleton predictors
predictors: Dict[str, Predictor] = {}

def get_predictor(game_type: str) -> Predictor:
    if game_type not in predictors:
        raise HTTPException(400, f'Invalid type: {game_type}. Use hu or md5.')
    return predictors[game_type]

# ── Predict ──────────────────────────────────────────────────────────────────
@app.get('/predict')
async def predict(request: Request, type: str = 'hu', auth=Depends(user_auth)):
    p = get_predictor(type)
    return JSONResponse(p.predict_next())

# ── Verify ───────────────────────────────────────────────────────────────────
@app.post('/verify')
async def verify(request: Request, auth=Depends(user_auth)):
    body = await request.json()
    gt     = body.get('type', 'hu')
    phien  = body.get('phien')
    actual = body.get('actual', '')
    if phien is None:
        raise HTTPException(400, 'phien required')
    p = get_predictor(gt)
    return JSONResponse(p.verify(int(phien), str(actual)))

# ── Status ───────────────────────────────────────────────────────────────────
@app.get('/status')
async def status(request: Request, auth=Depends(user_auth)):
    result = {}
    for gt, p in predictors.items():
        n = p.collector.count(gt)
        result[gt] = {
            'n_sessions': n,
            'model_trained': p.ensemble.is_trained,
            'accuracy': round(p.ensemble.accuracy, 4),
            'auc': round(p.ensemble.auc, 4),
            'sliding_accuracy': round(p.ensemble.sliding_accuracy(), 4),
            'last_trained': p.ensemble.trained_at,
            'min_required': MIN_TRAIN,
            'ready': p.ensemble.is_trained,
            'deep_learning': HAS_TF,
            'bandit_weights': {k: round(v,3) for k,v in p.ensemble.bandit.get_probs().items()}
        }
    return JSONResponse({'status': 'ok', 'games': result, 'timestamp': datetime.utcnow().isoformat()+'Z'})

# ── Train (manual trigger) ────────────────────────────────────────────────────
@app.post('/train')
async def train(request: Request, auth=Depends(admin_auth)):
    body = await request.json() if request.headers.get('content-type','').startswith('application/json') else {}
    gt = body.get('type', 'all')
    types = ['hu', 'md5'] if gt == 'all' else [gt]
    for t in types:
        predictors[t].train_async()
    return JSONResponse({'status': 'training_started', 'types': types})

# ── Admin Key Management ──────────────────────────────────────────────────────
@app.get('/admin/keys')
async def admin_list_keys(request: Request, auth=Depends(admin_auth)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id,name,daily_limit,used_today,expires_at,is_active,created_at FROM api_keys"
    ).fetchall()
    conn.close()
    return JSONResponse({'keys': [dict(r) for r in rows]})

@app.post('/admin/keys')
async def admin_create_key(request: Request, auth=Depends(admin_auth)):
    body = await request.json()
    new_key = secrets.token_hex(32)
    conn = get_db()
    conn.execute(
        "INSERT INTO api_keys (key,name,daily_limit,expires_at) VALUES(?,?,?,?)",
        (new_key, body.get('name','user'), body.get('daily_limit', DEFAULT_DAILY), body.get('expires_at'))
    )
    conn.commit(); conn.close()
    return JSONResponse({'key': new_key, 'name': body.get('name','user')}, status_code=201)

@app.put('/admin/keys/{kid}')
async def admin_update_key(kid: int, request: Request, auth=Depends(admin_auth)):
    body = await request.json()
    fields = {k: v for k, v in body.items() if k in ('daily_limit','expires_at','is_active','note')}
    if not fields:
        raise HTTPException(400, 'No valid fields')
    set_clause = ', '.join(f"{k}=?" for k in fields)
    conn = get_db()
    conn.execute(f"UPDATE api_keys SET {set_clause} WHERE id=?", (*fields.values(), kid))
    conn.commit(); conn.close()
    return JSONResponse({'status': 'updated', 'id': kid})

@app.delete('/admin/keys/{kid}')
async def admin_delete_key(kid: int, request: Request, auth=Depends(admin_auth)):
    conn = get_db()
    conn.execute("UPDATE api_keys SET is_active=0 WHERE id=?", (kid,))
    conn.commit(); conn.close()
    return JSONResponse({'status': 'deactivated', 'id': kid})

@app.post('/admin/reset-daily')
async def admin_reset_daily(request: Request, auth=Depends(admin_auth)):
    reset_daily()
    return JSONResponse({'status': 'reset done'})

# ── Health ────────────────────────────────────────────────────────────────────
@app.get('/health')
async def health():
    return {'status': 'ok', 'timestamp': datetime.utcnow().isoformat()+'Z', 'deep_learning': HAS_TF}

# ── History / Learning ────────────────────────────────────────────────────────
@app.get('/history')
async def history(request: Request, type: str = 'hu', limit: int = 50, auth=Depends(user_auth)):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE type=? ORDER BY id DESC LIMIT ?",
        (type, limit)
    ).fetchall()
    conn.close()
    return JSONResponse({'type': type, 'history': [dict(r) for r in rows]})

# ── Compatibility routes (same as lc79b.js) ───────────────────────────────────
@app.get('/lc79-hu')
async def compat_hu(request: Request, auth=Depends(user_auth)):
    return JSONResponse(predictors['hu'].predict_next())

@app.get('/lc79-md5')
async def compat_md5(request: Request, auth=Depends(user_auth)):
    return JSONResponse(predictors['md5'].predict_next())

@app.get('/lc79-hu/lichsu')
async def lichsu_hu(request: Request, auth=Depends(user_auth)):
    conn = get_db()
    rows = conn.execute("SELECT * FROM predictions WHERE type='hu' ORDER BY id DESC LIMIT 100").fetchall()
    conn.close()
    return JSONResponse({'type':'hu','history':[dict(r) for r in rows],'total':len(rows)})

@app.get('/lc79-md5/lichsu')
async def lichsu_md5(request: Request, auth=Depends(user_auth)):
    conn = get_db()
    rows = conn.execute("SELECT * FROM predictions WHERE type='md5' ORDER BY id DESC LIMIT 100").fetchall()
    conn.close()
    return JSONResponse({'type':'md5','history':[dict(r) for r in rows],'total':len(rows)})

# ═══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════════════════════
@app.on_event('startup')
async def startup():
    log.info("═"*60)
    log.info("  HYPER TAIXIU PREDICTOR v3.0 — Starting")
    log.info("═"*60)

    init_db()
    admin = init_admin_key()

    # Init predictors
    for gt in ['hu', 'md5']:
        p = Predictor(gt)
        predictors[gt] = p
        loaded = p.ensemble.load()
        if not loaded:
            log.info(f"[{gt}] No saved model. Will train after data collection.")

    # Initial data fetch
    collector = DataCollector()
    for gt in ['hu', 'md5']:
        sessions = collector.fetch(gt)
        if sessions:
            n = collector.save(sessions)
            total = collector.count(gt)
            log.info(f"[{gt}] Initial fetch: +{n} new, total={total}")
        # Trigger training if enough data
        predictors[gt].train_async()

    # Background jobs
    collector.run_background()
    for gt in ['hu', 'md5']:
        predictors[gt].schedule_retrain()

    start_daily_reset()

    log.info(f"ADMIN KEY: {admin}")
    log.info(f"Auth enabled: {ENABLE_AUTH}")
    log.info(f"Deep Learning (TF): {HAS_TF}")
    log.info(f"Server port: {PORT}")
    log.info("Endpoints: /predict?type=hu|md5  /verify  /status  /train  /history")
    log.info("═"*60)

if __name__ == '__main__':
    uvicorn.run('hyper_taixiu_predictor:app', host='0.0.0.0', port=PORT,
                workers=1, log_level='info')
