# AnimalTower-RL (どうぶつタワー強化学習)

## 概要
このプロジェクトは、Deep Reinforcement Learningを使用して「どうぶつタワー」のゲームを学習するシステムです。Ray RLlibを使用した強化学習の実装を含みます。

## 機能
- 強化学習によるゲームプレイの自動化
- カスタム環境の実装
- 学習結果の可視化と統計分析
- 学習済みモデルの予測機能

## 必要条件
- Python 3.x
- OpenCV
- Appium-Python-Client
- Pyper
- Matplotlib
- Ray RLlib

## インストール方法
```bash
pip install -r requirements.txt
```

## 使用方法

### 学習の開始
```bash
python start_train.py
```

### 学習の再開
```bash
python resume_train.py
```

### 予測の実行
```bash
python predict.py
```

### 統計の計算
```bash
python calc_stat.py
```

### グラフの描画
```bash
python draw_graph.py
```

## プロジェクト構造
- `src/`: ソースコード
- `statistics/`: 統計データ
- `test/`: テストコード
- `log/`: ログファイル
- `env.py`: カスタム環境の実装
- `start_train.py`: 学習開始スクリプト
- `resume_train.py`: 学習再開スクリプト
- `predict.py`: 予測スクリプト
- `calc_stat.py`: 統計計算スクリプト
- `draw_graph.py`: グラフ描画スクリプト

## モデル設定
### 畳み込みフィルターについて
Ray RLlibのモデル設定では、環境の観測サイズに応じて適切な畳み込みフィルターを設定する必要があります。詳細は[Ray RLlibのドキュメント](https://docs.ray.io/en/ray-1.1.0/rllib-models.html)を参照してください。

## ライセンス
このプロジェクトはMITライセンスの下で公開されています。
