# ch12

第12章の実験コードを置く。

## sec12_01_readout_mitigation.py

読み出し誤差の校正から補正行列を推定し、測定結果の確率を読み出し補正で推定し直す。校正が一致している場合と、校正がずれた場合を並べ、補正が必ずしも改善にならないことも示す。

実行例は次のとおりである。

```bash
python sec12_01_readout_mitigation.py --outdir out --datadir data
```

図は `out/` に、数値データは `data/` に保存される。

