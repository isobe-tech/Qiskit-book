# ch11

第11章の実験コードを置く。

## sec11_01_noise_intro.py

ノイズを読み出し誤差とゲート誤差と緩和に分け、理想モデルとノイズモデルで結果がどう変わるかを確認する。さらに深さを増やすと成功率が下がる様子を図にする。

実行例は次のとおりである。

```bash
python sec11_01_noise_intro.py --outdir out --datadir data
```

図は `out/` に、数値データは `data/` に保存される。

