# 4. シミュレーションと可視化の使い方

MPCの実験には、参照入力を作る処理と、結果を可視化する処理が重要です。

このリポジトリでは、以下の2つがよく使われています。

- `PulseGenerator.sample_pulse(...)`: 規則的なパルス信号を作る
- `SimulationPlotterDash`: 時系列データをグラフで見る

## 1. パルス参照信号の作り方

`sample/lti_no_constraints/servo_motor.py` では次のように使っています。

```python
_, input_signal = PulseGenerator.sample_pulse(
    sampling_interval=dt,
    start_time=time[0],
    period=20.0,
    pulse_width=50.0,
    pulse_amplitude=1.0,
    duration=time[-1],
)
```

これは、MPCの参照入力として使う信号を作る処理です。

- `sampling_interval`: サンプリング間隔
- `period`: パルスの周期
- `pulse_width`: パルス幅の比率
- `pulse_amplitude`: パルスの高さ
- `duration`: 全体の時間長

この入力信号は、制御対象に対して「ステップやパルス状の変化」を与えるために使われます。

## 2. 波形の可視化

サンプルコードでは、最後に `SimulationPlotterDash` を使って結果を描画しています。

```python
plotter = SimulationPlotterDash()

plotter.append_name(reference, "reference")
plotter.append_name(U, "U")
plotter.append_name(y_measured, "y_measured")
plotter.append_name(X, "X")
```

その後、どの信号をどの位置に表示するかを `assign` で指定します。

```python
plotter.assign("reference", position=(0, 0),
               row=0, column=0, x_sequence=time)
plotter.assign("y_measured", position=(0, 0),
               row=0, column=0, x_sequence=time)

plotter.assign_all("U", position=(2, 0), x_sequence=time)
```

ここでの考え方は、次のどれだけ単純です。

- まず信号を追加する
- どのグラフのどこに置くかを決める
- `plot()` で描画する

## 3. 使い方の流れ

```mermaid
flowchart LR
    A[時系列データを準備] --> B[append_name / append]
    B --> C[assign でレイアウト指定]
    C --> D["plot() で表示"]
```

この流れを覚えると、MPCの結果の確認がしやすくなります。

## 4. よく使うメソッド

| メソッド | 役割 |
| --- | --- |
| `append_name(data, name)` | 名前付きの時系列データを登録する |
| `assign(name, position, row, column, x_sequence)` | グラフ内の位置を決める |
| `assign_all(name, position, x_sequence)` | 一連の値をまとめて配置する |
| `plot(title)` | グラフを表示する |

実際のサンプルでは、以下のような観点で描画しています。

- 基準値 `reference`
- 実際の出力 `y_measured`
- 状態 `X`
- 制御入力 `U`

これにより、MPCの「目標追従」「状態変化」「入力の大きさ」を一目で確認できます。

## 5. 何を見ればよいか

制御設計の初学者が最初に見るべきものは、次の3つです。

1. 参照値と出力がどれだけ追従しているか
2. 制御入力が不必要に大きくないか
3. 状態が振動していないか

`SimulationPlotterDash` は、このような確認を簡単にするための補助機能です。

## 6. まとめ

MPCの理解を深めるには、数式だけでなく「入力がどう変わり、出力がどう見えるか」を見ることが重要です。

- `PulseGenerator` で興味のある入力信号を作る
- `SimulationPlotterDash` で時間変化を視覚化する
- `reference` と `y_measured` の比較から、制御がうまく動いているかを見る

この繰り返しが、MPCの使い方を体で覚える近道になります。
