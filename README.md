# 概要
 - Transformerを用いて日英翻訳を行う

# requirements
 - python >= 3.6
 - tensorflow >= 2.0
 - mecab-python3==0.6

# ファイルの説明
 ## ノートブック
 - Train_on_GPU.ipynb：GPUによる学習
 - Train_on_TPU.ipynb：TPUによる学習
 - Predict_on_GPU.ipynb：GPUによる推論
 - Predict_on_TPU.ipynb：TPUによる推論
 
 ## Pythonファイル
 - preprocess_utils.py：データの前処理プログラム
 - weight_utils.py：モデル保存・読み込みのプログラム
 - model.py：Transformerのプログラム

 ## 学習済みモデル 
 - checkpoints/gpu/：GPUの学習済みモデル
 - checkpoints/tpu/：TPUの学習済みモデル

# 使い方
 ## 実行環境
 - Google Colaboratory上で実行するため、Googleアカウントを持っている必要があります
 - 本ファイルをGoogleドライブにアップロードしてください
 ![drive_1](https://github.com/kawasaki-kento/Transformer/blob/master/image/drive_1.png)

 ## ハードウェア アクセラレータの選択
 - GPUもしくはTPUを使う際に、次のようにハードウェア アクセラレータを選択してください
 - GPU：ランタイム → ランタイムの変更 → ハードウェア アクセラレータ → GPU
 - TPU：ランタイム → ランタイムの変更 → ハードウェア アクセラレータ → TPU
 ![hardware_1](https://github.com/kawasaki-kento/Transformer/blob/master/image/hardware_1.png)
 - 
 ![hardware_2](https://github.com/kawasaki-kento/Transformer/blob/master/image/hardware_2.png)

 ## 使用データ
 - http://www.manythings.org/anki/ の日英翻訳コーパスを使用します
 - このデータは、CC-BY 2.0 (France)に従い、各文の属性欄に帰属者の記載があります
 ![download_1](https://github.com/kawasaki-kento/Transformer/blob/master/image/download_1.png)
 
 ## ドライブのマウント
 - Googleドライブを使用するため、次のようにドライブのマウントを行ってください
 ![mount_1](https://github.com/kawasaki-kento/Transformer/blob/master/image/mount_1.png)
 - 
 ![mount_2](https://github.com/kawasaki-kento/Transformer/blob/master/image/mount_2.png)
 
 ## データの読み込み
 - データ件数は約5.3万件、その内、約5万件を学習、約3千件をテスト用のデータとしています
 - 学習とテストの割合はsplit_percentで調整できます
 - 以下のように、パラレルコーパスになっていることを確認してください
 ![load_1](https://github.com/kawasaki-kento/Transformer/blob/master/image/load_1.png)

 ## 前処理
 - 前処理は以下の図のように、文章をベクトルに変換しています
 - その際、文章の先頭に\<start\>、後尾に\<end\>を追加しています
 - また、不明な単語の場合は\<unk\>が割り当てられるようになっています
 ![preprocess_1](https://github.com/kawasaki-kento/Transformer/blob/master/image/preprocess_1.png)

 ## ハイパーパラメータの設定
 - num_layers：内部レイヤー数
 - d_model：中間層のユニット数
 - num_heads：Multi Head Attentionのヘッド数
 - dff=2048：Feed Forward Networkのユニット数
 - dropout_rate：ドロップアウト率
 ![parameter_1](https://github.com/kawasaki-kento/Transformer/blob/master/image/parameter_1.PNG)

 ## モデルの保存・読み込みについて
 - モデルの保存・読み込みは、本来GCSを使うべきですが、このコードではGoogleドライブで行っています
 - checkpoint_path = "/content/drive/My Drive/Transformer/checkpoints/gpu/model"と読み込み先の指定ができます
 - そのため、学習時もしくは予測時、以下のことに注意してください
 	 - 「Load checkpoints successfully.」となれば、学習済みモデルが読み込めています
   ![checkpoint_3](https://github.com/kawasaki-kento/Transformer/blob/master/image/checkpoint_3.PNG)
 	 - 「*** Failed to load model weights ***」、「*** Failed to load optimizer weights ***」となれば、学習済みモデルが読み込めていませんので、ランタイムを再起動してやり直してください
   ![checkpoint_2](https://github.com/kawasaki-kento/Transformer/blob/master/image/checkpoint_2.PNG)
 	 - 「No available checkpoints.」となれば、学習済みモデルがないので、そのまま学習を実行してモデルを作成してください
   ![checkpoint_1](https://github.com/kawasaki-kento/Transformer/blob/master/image/checkpoint_1.PNG)


 ## 学習
 - EPOCHSを設定して実行してください
 - 50バッチごとに、学習の進捗が表示されます
 - また、5Epochごとにモデルが保存されます
 ![train_1](https://github.com/kawasaki-kento/Transformer/blob/master/image/train_1.PNG)

 ## 推論
 - テストデータを用いて推論を行います
 - 推論を実行すると予測結果を見ることができます

 ## Attention Weight の見方
 - 横軸が入力文、翻訳元の文章、縦軸が翻訳後の文になっています
 - 色が明るいほど、単語間の関係性が強いことを意味します
 - 例えば以下の図の場合、「This」と「この」の関係が強く出ていることがわかります
 ![attention_weight_1](https://github.com/kawasaki-kento/Transformer/blob/master/image/attention_weight_1.PNG)
