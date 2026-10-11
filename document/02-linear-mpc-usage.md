# 2. 線形MPCの使い方

線形MPCの最も基本的な使い方は、`sample/lti_no_constraints/servo_motor.py` で確認できます。

## 1. 状態空間モデルを作る

線形MPCでは、制御対象のモデルを `A, B, C` の状態空間として持ちます。ここでは `SymbolicStateSpace` を使って、離散時間の状態空間モデルをラップします。

```python
sys_d = sys.sample(Ts=dt, method='euler')

ideal_plant_model = SymbolicStateSpace(
    sys_d.A, sys_d.B, sys_d.C,
    delta_time=dt,
    Number_of_Delay=0,
)
```

このコードの意味は、次のとおりです。

- `sys_d.A`, `B`, `C`: 離散時間の状態空間行列
- `delta_time`: サンプリング時間
- `Number_of_Delay`: 出力遅れがある場合の遅延数

`SymbolicStateSpace` は、MPC内部で必要になる状態や入力の次元を整理し、計算に必要な行列を扱いやすくする役割があります。

## 2. 重みとホライズンを決める

MPCでは「今の入力をどれだけ重視するか」「出力の追従をどれだけ重視するか」を重みで決めます。

```python
Weight_U = np.diag([0.001])
Weight_Y = np.diag([1.0, 0.005])

Np = 20
Nc = 2
```

- `Weight_U`: 入力変化の大きさに対する重み
- `Weight_Y`: 出力誤差に対する重み
- `Np`: 予測ホライズン（何ステップ先まで予測するか）
- `Nc`: 制御ホライズン（何ステップまで入力を調整するか）

サンプルでは入力は1つ、出力は2つです。したがって、重みはそれに合わせて行列を作っています。

## 3. MPCコントローラを生成する

```python
lti_mpc = LTI_MPC_NoConstraints(
    ideal_plant_model,
    Np=Np,
    Nc=Nc,
    Weight_U=Weight_U,
    Weight_Y=Weight_Y,
)
```

ここでのポイントは、モデルと重みを渡せば、後は `update()` で制御入力が返るようにしてあることです。

`LTI_MPC_NoConstraints` は、線形システム向けの「制御器本体」であり、内部で以下を管理します。

- カルマンフィルタによる状態推定
- 予測行列の生成
- 目標値との誤差計算
- 最適な入力の更新

## 4. 参照値と測定値を渡す

実際の制御ループでは、次のように使います。

```python
reference = np.array([[input_signal[i, 0]], [0.0]])
y_measured = C @ X

U = lti_mpc.update(reference, y_measured)
```

`reference` は目標値です。今回の例では、出力が2次元で1つは角度、もう1つは速度として扱われているので、形状は `(2, 1)` になっています。

`y_measured` は現在の出力計測値です。MPCは、この値と目標値の差分をもとに最適な入力を計算します。

`U = lti_mpc.update(...)` の戻り値は、次の1ステップに適用する制御入力です。ここでは1入力のシステムなので、スカラーのような形で返ってきます。

## 5. 使うときに意識するポイント

### 参照値の形

MPCクラスの `update` は、参考入力を配列で渡す設計です。サンプルでは以下のように扱っています。

```python
reference = np.array([[input_signal[i, 0]], [0.0]])
```

- 1列目: 角度の目標
- 2列目: 速度の目標

実際には、目標を 1次元のスカラーにすることもできますが、出力が複数ある場合は各出力に対する参考値を並べる設計になります。

### 計測遅れ

`SymbolicStateSpace` の `Number_of_Delay` に遅延ステップ数を渡せます。`LTI_MPC_NoConstraints` は、内部で遅延分の補正を行うようになっています。

## 6. この流れが意味すること

`sample/lti_no_constraints/servo_motor.py` は、線形システムのMPCを使うときに必要な「最小限の流れ」を全部含んでいます。

- モデルを作る
- 重みとホライズンを決める
- コントローラを作る
- 参照値と現在出力を渡す
- 次の制御入力を受け取る

この順序を押さえておけば、他の線形制御対象にも応用できます。

## 7. まとめ

線形MPCの使い方は、概念的にはシンプルです。

```mermaid
flowchart LR
    A[状態空間モデル] --> B[SymbolicStateSpace]
    B --> C[LTI_MPC_NoConstraints]
    C --> D["update(reference, y_measured)"]
    D --> E[制御入力 U]
```

重要なのは、「モデルさえ用意できれば、後は `_update()` で制御入力が返る」という感覚を持つことです。

[03-nonlinear-mpc-usage.md](./03-nonlinear-mpc-usage.md)
